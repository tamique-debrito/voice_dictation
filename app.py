"""v2 entry point — wires every module behind the EventBus + direct queues.

Modes:
  --selftest     publish one event per persisted topic, verify, exit
  --feed-wav F   feed a WAV file through the preprocessor (no PyAudio),
                 then exit. Useful for parity testing against v1.
  (default)      live mic mode: start the PersistentRecorder, run until
                 SIGINT, then shut down cleanly.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import struct
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from typing import Optional

from .audio_persister import AudioPersister
from .audio_preprocessor import AudioPreprocessor
from .audio_windower import AudioWindower
from .clock import next_seq
from .event_bus import EventBus
from .paste_executor import PasteExecutor
from .persistent_recorder import PersistentRecorder
from .runtime_config import RuntimeConfig, load_runtime_config
from .streams import EventStreamPersister
from .transcriber import Transcriber
from .transcript_stream_manager import TranscriptStreamManager
from .types import (
    Event,
    TOPIC_AUDIO_SILENCE, TOPIC_AUDIO_CHUNK_COMMITTED,
    TOPIC_WINDOWS_FAST, TOPIC_WINDOWS_HQ,
    TOPIC_SEGMENTS_FAST, TOPIC_SEGMENTS_HQ,
    TOPIC_PROGRESS_FAST, TOPIC_PROGRESS_HQ,
    TOPIC_USER_PRESSES, TOPIC_USER_ACTIONS,
    TOPIC_PASTE_ACTIONS, TOPIC_TRANSCRIPT_CANONICAL,
)
from .user_action_manager import UserActionManager
from .widget_server import WidgetServer


logger = logging.getLogger(__name__)


PERSIST_TOPICS: dict[str, str] = {
    TOPIC_AUDIO_SILENCE: "silence_annotations.jsonl",
    TOPIC_AUDIO_CHUNK_COMMITTED: "audio_chunks.jsonl",
    TOPIC_WINDOWS_FAST: "fast_windows.jsonl",
    TOPIC_WINDOWS_HQ: "hq_windows.jsonl",
    TOPIC_SEGMENTS_FAST: "fast_segments.jsonl",
    TOPIC_SEGMENTS_HQ: "hq_segments.jsonl",
    TOPIC_PROGRESS_FAST: "fast_progress.jsonl",
    TOPIC_PROGRESS_HQ: "hq_progress.jsonl",
    TOPIC_USER_PRESSES: "user_press_events.jsonl",
    TOPIC_USER_ACTIONS: "user_actions.jsonl",
    TOPIC_PASTE_ACTIONS: "paste_actions.jsonl",
    TOPIC_TRANSCRIPT_CANONICAL: "transcript_canonical.jsonl",
}


class App:
    def __init__(self, cfg: RuntimeConfig, session_dir: str) -> None:
        self.cfg = cfg
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)
        self.stop = threading.Event()
        self.bus = EventBus()

        # Persisters first so they're listening before any producer emits.
        self._persisters: list[EventStreamPersister] = [
            EventStreamPersister(
                self.bus,
                topic,
                os.path.join(session_dir, filename),
                max_queued=cfg.bus.persister_max_queued,
            )
            for topic, filename in PERSIST_TOPICS.items()
        ]

        # Preprocessor + persister.
        self.preprocessor = AudioPreprocessor(
            cfg.preprocessor, cfg.audio, self.bus, self.stop,
        )
        self.audio_persister = AudioPersister(
            cfg.audio_persist, cfg.audio, self.bus, self.preprocessor,
            session_dir, self.stop,
        )
        self.preprocessor.add_sink(self.audio_persister)

        # Per-stream windowers + transcribers.
        self.fast_windower = AudioWindower(
            cfg.fast, cfg.audio, self.bus, self.preprocessor, self.stop,
        )
        self.hq_windower = (
            AudioWindower(cfg.hq, cfg.audio, self.bus, self.preprocessor, self.stop)
            if cfg.hq.enabled else None
        )
        self.preprocessor.add_sink(self.fast_windower)
        if self.hq_windower is not None:
            self.preprocessor.add_sink(self.hq_windower)

        self.fast_transcriber = Transcriber(
            cfg.fast, cfg.audio, self.bus, self.preprocessor,
            self.fast_windower.window_q, self.stop,
        )
        self.hq_transcriber: Optional[Transcriber] = None
        if self.hq_windower is not None:
            self.hq_transcriber = Transcriber(
                cfg.hq, cfg.audio, self.bus, self.preprocessor,
                self.hq_windower.window_q, self.stop,
            )

        self.manager = TranscriptStreamManager(
            self.bus, self.preprocessor,
            self.fast_transcriber.segment_q,
            self.hq_transcriber.segment_q if self.hq_transcriber else None,
            self.stop,
            complete_tolerance_ms=cfg.manager.complete_tolerance_ms,
        )
        marker_keys = {m["key"] for m in cfg.markers}
        self.user_actions = UserActionManager(
            self.bus, self.preprocessor, self.preprocessor, self.stop,
            hotkey_cfg=cfg.hotkeys,
            app_cfg=cfg.app,
            marker_keys=marker_keys,
            listen=True,
        )
        self.paste_executor = PasteExecutor(self.bus, self.stop)

        # Recorder is created lazily so non-mic modes (selftest, feed-wav)
        # can skip PyAudio init.
        self.recorder: Optional[PersistentRecorder] = None

        # Widget server is also lazy (only spun up in live + offline modes,
        # not in selftest).
        self.widget: Optional[WidgetServer] = None

        logger.info("App wired; session_dir=%s", session_dir)

    # ------------------------------------------------------------------

    def _stats_modules(self) -> dict:
        d: dict = {
            "preprocessor": self.preprocessor,
            "audio_persister": self.audio_persister,
            "windower.fast": self.fast_windower,
            "transcriber.fast": self.fast_transcriber,
        }
        if self.hq_windower is not None:
            d["windower.hq"] = self.hq_windower
        if self.hq_transcriber is not None:
            d["transcriber.hq"] = self.hq_transcriber
        d["manager"] = self.manager
        return d

    def start_mic(self, with_widget: bool = True) -> None:
        self.recorder = PersistentRecorder(
            self.preprocessor.raw_input_q, self.cfg.audio,
        )
        self.recorder.start()
        if with_widget:
            self.widget = WidgetServer(
                self.bus, self.preprocessor, self.manager,
                port=self.cfg.app.widget_port,
                cfg=self.cfg,
                stats_modules=self._stats_modules(),
            )
            self.widget.start()
        self._start_processors()

    def start_offline(self, with_widget: bool = False) -> None:
        """Start everything except the mic recorder."""
        if with_widget:
            self.widget = WidgetServer(
                self.bus, self.preprocessor, self.manager,
                port=self.cfg.app.widget_port,
                cfg=self.cfg,
                stats_modules=self._stats_modules(),
            )
            self.widget.start()
        self._start_processors()

    def _start_processors(self) -> None:
        self.preprocessor.start()
        self.audio_persister.start()
        self.fast_windower.start()
        if self.hq_windower:
            self.hq_windower.start()
        self.fast_transcriber.start()
        if self.hq_transcriber:
            self.hq_transcriber.start()
        self.manager.start()
        self.user_actions.start()
        logger.info("App started")

    def shutdown(self) -> None:
        self.stop.set()
        # Order matters: upstream first so trailing data reaches downstream
        # before downstream stops.
        if self.widget is not None:
            self.widget.shutdown()
            self.widget = None
        if self.recorder is not None:
            self.recorder.stop()
        # Preprocessor next — its remaining audio reaches the windowers.
        self.preprocessor.shutdown()
        # Windowers next — their final flushes reach the transcribers.
        if self.hq_windower:
            self.hq_windower.shutdown()
        self.fast_windower.shutdown()
        # Transcribers next — their segments reach the manager.
        if self.hq_transcriber:
            self.hq_transcriber.shutdown()
        self.fast_transcriber.shutdown()
        # Manager + actions.
        self.manager.shutdown()
        self.user_actions.shutdown()
        # Persister (audio) is independent — drain it last.
        self.audio_persister.shutdown()
        # Bus consumers last.
        self.paste_executor.shutdown()
        for p in self._persisters:
            p.shutdown()
        self.bus.shutdown()
        logger.info("App shutdown complete")

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def selftest(self) -> bool:
        logger.info("running selftest")
        for topic in PERSIST_TOPICS.keys():
            self.bus.publish(Event(
                topic=topic,
                emit_accepted_ms=0,
                seq=next_seq(),
                payload={"selftest": True, "topic": topic},
            ))
        time.sleep(0.5)
        ok = True
        for topic, filename in PERSIST_TOPICS.items():
            path = os.path.join(self.session_dir, filename)
            if not os.path.exists(path):
                logger.error("selftest: %s missing", path)
                ok = False
                continue
            with open(path) as f:
                lines = [ln for ln in f if ln.strip()]
            if len(lines) != 1:
                logger.error("selftest: %s has %d lines, expected 1", path, len(lines))
                ok = False
        logger.info("selftest %s", "PASSED" if ok else "FAILED")
        return ok

    # ------------------------------------------------------------------
    # WAV feed — for offline parity testing against v1
    # ------------------------------------------------------------------

    def feed_wav(self, path: str, max_seconds: float = 0.0) -> None:
        """Read a 16 kHz mono int16 WAV and push it through the preprocessor
        in chunk-sized pieces. Pushes as fast as the queue accepts (no realtime
        pacing — for offline parity testing). ``max_seconds`` caps the feed."""
        with wave.open(path, "rb") as wf:
            if wf.getframerate() != self.cfg.audio.sample_rate:
                raise ValueError(
                    f"WAV sample rate {wf.getframerate()} != "
                    f"config {self.cfg.audio.sample_rate}"
                )
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError("WAV must be mono int16")
            total = wf.getnframes()
            if max_seconds > 0:
                total = min(total, int(max_seconds * wf.getframerate()))
            logger.info("feed_wav: %s (%.2fs)", path, total / wf.getframerate())
            sent = 0
            sample_rate = wf.getframerate()
            while sent < total:
                want = min(self.cfg.audio.chunk_size, total - sent)
                chunk = wf.readframes(want)
                if not chunk:
                    break
                ts = sent / sample_rate
                self.preprocessor.raw_input_q.put((ts, chunk))
                sent += want
            # Allow downstream to drain.
            time.sleep(1.0)


# ----------------------------------------------------------------------


def _build_session_dir(cfg: RuntimeConfig) -> str:
    base = cfg.app.sessions_dir
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, stamp)


def _apply_env(cfg: RuntimeConfig) -> None:
    if cfg.app.hf_hub_offline and "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"


def _startup_extras(cfg: RuntimeConfig, session_dir: str) -> list[tuple[str, str]]:
    keys = cfg.hotkeys.keys
    hotkey_parts = [
        f"{keys.get('toggle_recording', 'r')}×2 record",
        f"{keys.get('toggle_aside', 'a')}×2 aside",
        f"{keys.get('discard_recording', 'x')}×2 discard",
        "m×2 mute",
        "q×2 quit",
    ]
    models = f"fast={cfg.fast.fw.model}"
    if cfg.hq.enabled:
        models += f", hq={cfg.hq.fw.model}"
    return [
        ("session", session_dir),
        ("models", models),
        ("hotkeys", "  ".join(hotkey_parts)),
    ]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="voice_dictation entry point")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--feed-wav", default=None,
                        help="feed a WAV file instead of using the mic")
    parser.add_argument("--feed-max-seconds", type=float, default=0,
                        help="cap WAV feed to N seconds (0 = whole file)")
    parser.add_argument("--duration", type=float, default=0,
                        help="live mic: seconds to record (0 = until SIGINT)")
    parser.add_argument("--widget", action="store_true",
                        help="start the widget server (live mic mode starts it by default)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cfg = load_runtime_config()
    _apply_env(cfg)
    session_dir = _build_session_dir(cfg)
    app = App(cfg, session_dir)

    try:
        if args.selftest:
            app.start_offline()
            ok = app.selftest()
            return 0 if ok else 1

        if args.feed_wav:
            app.start_offline(with_widget=args.widget)
            app.feed_wav(args.feed_wav, max_seconds=args.feed_max_seconds)
            if args.widget:
                port = app.widget.actual_port if app.widget else 0
                from .widget_server import _print_url_banner
                _print_url_banner(
                    "widget",
                    f"http://127.0.0.1:{port}/",
                    extras=_startup_extras(cfg, session_dir),
                )
                logger.info("Ctrl-C to exit")
                signal.signal(signal.SIGINT, lambda *_: app.stop.set())
                app.stop.wait()
            return 0

        # Live mic mode.
        app.start_mic()
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            signal.signal(signal.SIGINT, lambda *_: app.stop.set())
            from .widget_server import _print_url_banner
            port = app.widget.actual_port if app.widget else 0
            _print_url_banner(
                "widget",
                f"http://127.0.0.1:{port}/",
                extras=_startup_extras(cfg, session_dir),
            )
            logger.info("recording — Ctrl-C or double-tap Q to stop")
            app.stop.wait()
        return 0
    finally:
        app.shutdown()
        if not args.selftest:
            from .widget_server import print_closing_banner
            print_closing_banner([("session", session_dir)])


if __name__ == "__main__":
    sys.exit(main())
