#!/usr/bin/env python3
"""Persistent always-on voice dictation with marker hotkeys.

Architecture (single process, multiple threads):

    PyAudio thread (PersistentRecorder)
        └─► raw frame queue (ts, bytes)
                │
                ▼
    Aggregator thread
        ├─ feeds bytes into webrtcvad (SilenceDetector)
        ├─ packages PCM into AudioWindow at silence boundaries
        │   (or after max_window_seconds), pushes into transcriber.in_queue
        ▼
    Transcriber worker (StreamingTranscriber, faster-whisper)
        └─► Segment out_queue
                │
                ▼
    Drain thread
        ├─ feeds Segments into SessionWriter
        └─ checks chunk-flush threshold

    pynput Listener thread
        └─ BareDoubleTap routes marker keys (1, 2), r, a, x, q

Audio is never written to disk. Markers are inserted inline in the
chunk_*.txt files, with the no-overlap rule enforced by SessionWriter.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime, timezone
from typing import Optional

# Skip HuggingFace's "is there a newer model?" network check on startup
# unless the user passed --check-updates OR set hf_hub_offline=false in
# local_config.json. Without HF_HUB_OFFLINE=1, a slow/blocked HF connection
# stalls model load for 60s+. We set the env var here because huggingface_hub
# reads HF_HUB_OFFLINE once at import time — must be in place BEFORE
# faster_whisper imports (transitively via streaming_transcriber).
_CHECK_UPDATES = "--check-updates" in sys.argv
from runtime_config import CFG as _CFG  # noqa: E402 — must precede faster_whisper
if not _CHECK_UPDATES and _CFG.persistent.hf_hub_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    # Explicitly clear so a stale env var from a parent shell doesn't
    # contradict the JSON setting.
    os.environ.pop("HF_HUB_OFFLINE", None)

# Tracker file recording when we last actually contacted HF for updates.
_HF_CHECK_FILE = os.path.join(os.path.dirname(__file__), ".last_hf_check")
_HF_CHECK_STALE_SECONDS = 30 * 24 * 3600  # 30 days

from rich.console import Console
from rich.panel import Panel

from audio_archiver import AudioArchiver
from audio_fanout import AudioFanout
from clipboard_manager import ClipboardManager
from clipboard_window import (
    ClipboardWindowManager,
    ClipboardWindowType,
)
from paste_executor import PasteExecutor
from config import (
    ASIDE_MARKER_TYPE,
    CHUNK_TOKEN_TARGET,
    DEFAULT_MARKERS,
    FW_COMPUTE,
    FW_DEVICE,
    FW_MODEL,
    MIN_VOICED_FRAC,
    MIN_VOICED_MS,
    RECORDING_MARKER_TYPE,
    SILENCE_MS,
    TRANSCRIPT_STREAM_PORT,
    VAD_AGGRESSIVENESS,
    WIDGET_PORT,
)
from transcript_stream_server import TranscriptStreamServer
from hotkeys import load_config
from persistent_recorder import PersistentRecorder
from runtime_config import CFG, apply_dict_to_config, save_runtime_config, to_dict
from session_state import SessionState
from transcript_timeline import MarkerType, TranscriptTimeline
from transcription_stream import TranscriptionStream
from user_action_producer import UserActionProducer
import widget as widget_module


# PASTE_DRAIN_TIMEOUT lives in PasteExecutor now (as DEFAULT_DRAIN_TIMEOUT).


class PersistentApp:
    def __init__(self, config: dict, model: str, device: str, compute: str,
                 enable_widget: bool = True, open_browser: bool = False,
                 save_audio_override: Optional[bool] = None):
        self.console = Console()
        self.config = config
        self._stop = threading.Event()
        self._enable_widget = enable_widget
        self._open_browser = open_browser
        self._widget: Optional[widget_module.StatusServer] = None
        self._started_at = datetime.now(tz=timezone.utc)
        self._device = device
        self._compute = compute
        # CLI flag wins over the config flag so a user can opt in without
        # editing local_config.json. None = follow config; True/False = override.
        self._save_audio = (
            save_audio_override
            if save_audio_override is not None
            else bool(CFG.persistent.save_audio)
        )
        self._audio_archiver: Optional[AudioArchiver] = None

        marker_cfgs = config.get("persistent", {}).get("markers", DEFAULT_MARKERS)
        self.marker_types = [
            MarkerType(
                key=m["key"],
                type=m["type"],
                description=m["description"],
                flag=bool(m.get("flag", False)),
            )
            for m in marker_cfgs
        ]
        self._marker_keys = {m.key for m in self.marker_types}

        self.raw_q: queue.Queue = queue.Queue(maxsize=200)
        self.recorder = PersistentRecorder(self.raw_q)
        self.clipboard = ClipboardManager()
        self._stream_server = TranscriptStreamServer(port=TRANSCRIPT_STREAM_PORT)

        self._state_lock = threading.Lock()

        # AudioFanout drains raw_q, owns mute, and broadcasts frames to per-
        # stream input queues. Mute transitions emit a MUTE_RESET sentinel
        # that downstream aggregators consume to reset partial windows.
        self.audio_fanout = AudioFanout(self.raw_q, self._stop)

        # The unified canonical transcript. Owns: per-stream segment
        # buffers, markers (one list, anchored to audio_time), chunk files
        # (canonical + per-stream raw). All transcription streams ingest
        # segments into this single object.
        self.timeline = TranscriptTimeline(
            marker_types=self.marker_types,
            on_canonical_flush=self._on_canonical_flush,
        )

        # SessionState: absorbs the legacy DebugLog ring buffer + the
        # paste-log / marker-open mirror state previously held directly on
        # PersistentApp. Every state-mutating event flows through
        # self._state.ingest() (or .log() for the legacy two-arg shape used
        # by stream callbacks). snapshot() builds the widget /status payload.
        self._state = SessionState(
            timeline=self.timeline,
            started_at=self._started_at,
            device=device,
            compute=compute,
        )

        # Optional per-chunk audio archiver. Subscribes to the fanout (same
        # frames the streams see) and is invoked by ``_on_canonical_flush``
        # when a chunk_NNN.txt is written.
        if self._save_audio:
            archive_q = self.audio_fanout.subscribe(maxsize=400)
            self._audio_archiver = AudioArchiver(
                audio_q=archive_q,
                stop_event=self._stop,
                sample_rate=CFG.audio.sample_rate,
            )

        # Fast / live transcription stream — drives live paste + the
        # WebSocket stream broadcast and the canonical chunk flush ticks.
        fast_cfg = CFG.persistent.fast
        self.fast_stream = TranscriptionStream(
            label="fast",
            audio_input_q=self.audio_fanout.subscribe(maxsize=200),
            stop_event=self._stop,
            model_name=model,
            device=device,
            compute_type=compute,
            max_window_seconds=fast_cfg.max_window_seconds,
            silence_ms=SILENCE_MS,
            vad_aggressiveness=VAD_AGGRESSIVENESS,
            min_voiced_ms=MIN_VOICED_MS,
            min_voiced_frac=MIN_VOICED_FRAC,
            beam_size=fast_cfg.fw.beam_size,
            condition_on_previous_text=fast_cfg.fw.condition_on_previous_text,
            window_q_maxsize=fast_cfg.window_q_maxsize,
            debug_log=self._state.log,
            on_segment=self._on_segment,
            on_silence_boundary=self._on_silence_boundary_fast,
        )
        self.timeline.register_stream(
            "fast", priority=0, model_label=self.fast_stream.model_label
        )
        self.streams: list[TranscriptionStream] = [self.fast_stream]

        # HQ / high-quality transcription stream — slower model, beam
        # search, cross-window prompt context. Subscribed to the same
        # fanout so it sees the same audio. Backpressure-tolerant: each
        # subscriber queue is independent with drop-oldest-on-full.
        self._hq_audio_q: Optional[queue.Queue] = None
        self.hq_stream: Optional[TranscriptionStream] = None
        if CFG.persistent.hq.enabled:
            self._build_hq_stream()

        # Note: _paste_log and _marker_open_since now live inside SessionState.

        clipboard_types = [
            ClipboardWindowType(
                key="r",
                label="r",
                marker_type=RECORDING_MARKER_TYPE,
                can_park_others=False,
                can_be_parked=True,
                schedule_end_marker=True,
            ),
            ClipboardWindowType(
                key="a",
                label="aside",
                marker_type=ASIDE_MARKER_TYPE,
                can_park_others=True,
                can_be_parked=False,
                schedule_end_marker=False,
            ),
        ]
        # ClipboardWindowManager used to emit press / cursor_capture /
        # marker-drain derived events into the debug log; with R2 those
        # events are no longer the source of truth (user_action is), so
        # we pass a no-op debug_log. The clipboard manager still tracks
        # its internal state and schedules timeline markers correctly;
        # it just doesn't pollute the event stream with redundant rows.
        self.clipboard_mgr = ClipboardWindowManager(
            types=clipboard_types,
            timeline=self.timeline,
            debug_log=lambda kind, data: None,
            request_force_flush=self._request_force_flush_all,
            cursor_time_at=self._cursor_time_at,
        )

        # UserActionProducer owns pynput + double-tap detection +
        # backspace-out + key-to-action resolution. Its one output is a
        # ``user_action`` event per committed press: the third input
        # stream alongside fast/hq segments. SessionState.ingest is the
        # sole consumer — there is no longer a parallel ``_on_action``
        # dispatch path. All state mutation flows through user_action.
        self._user_actions = UserActionProducer(
            marker_keys=set(self._marker_keys),
            clipboard_keys=self.clipboard_mgr.keys(),
            debug_flag_key=CFG.persistent.debug_flag_key,
            cursor_time_at=self._cursor_time_at,
            on_user_action=self._dispatch_user_action,
            on_ack=self._ack,
            stop_event=self._stop,
        )

        # Wire SessionState's user_action dispatch to the live owners.
        # The clipboard manager + audio fanout are mutated directly from
        # SessionState._apply_user_action; the paste callback kicks off
        # PasteExecutor consumes paste-output from SessionState when a
        # clipboard window closes; the
        # stream server broadcasts marker open/close over the WebSocket.
        self._state.attach_live_components(
            recorder=self.recorder,
            audio_fanout=self.audio_fanout,
            clipboard_mgr=self.clipboard_mgr,
        )
        self._state.attach_stream_server(self._stream_server)
        self._state.attach_stop_event(self._stop)

        # PasteExecutor consumes the paste-output side of SessionState.
        # When a clipboard window closes (from a user_action(clipboard_toggle)
        # ending an active window), SessionState fires this callback with
        # the closed ClipboardWindow; PasteExecutor runs the drain → slice
        # → emit-paste-event → OS-paste pipeline on its own daemon thread.
        self._paste_executor = PasteExecutor(
            timeline=self.timeline,
            clipboard=self.clipboard,
            log_event=self._state.log,
            cursor_time_at=self._cursor_time_at,
            on_done=lambda msg, style: self._done(msg, style=style),
        )
        self._state.attach_paste_callback(self._paste_executor.execute)

    def _request_force_flush_all(self) -> None:
        """Fire force-flush on every active transcription stream."""
        for s in self.streams:
            s.request_force_flush()

    def _on_segment(self, stream_id: str, seg) -> None:
        """Live-only callback wired into every TranscriptionStream's drain
        loop. The segment has *already* been logged as a "segment" event
        via TranscriptionStream._emit_segment's ``_debug`` call (which
        routes through SessionState.log → SessionState._apply →
        timeline.ingest_segment), so we MUST NOT call ingest_segment
        again here — doing so doubled every phrase in the canonical
        render and the raw chunk files. This callback now only handles
        the live-only side effect of broadcasting fast-stream segments
        on the transcript WebSocket.
        """
        if stream_id == "fast":
            try:
                self._stream_server.broadcast(seg.text, seg.end_time)
            except Exception:
                pass

    # _snapshot_writer_loop removed: replay rebuilds widget state by
    # re-running the event stream through SessionState.ingest, so a
    # periodically-sampled snapshot file is redundant. See refactor plan.

    def _on_silence_boundary_fast(self) -> None:
        """Fast-aggregator silence-boundary tick. Drives canonical chunk
        flushing (and per-stream raw chunk ticking) on the timeline."""
        try:
            path = self.timeline.tick(at_silence_boundary=True)
        except Exception:
            return
        if path:
            self.console.print(f"[dim]💾 chunk → {os.path.basename(path)}[/dim]")

    def _on_canonical_flush(self, event: dict) -> None:
        """Fired by TranscriptTimeline whenever a chunk_NNN.txt is written.

        When --save-audio is on, slice the archiver's ring buffer over the
        chunk's [t_start, t_end] window and write the matching chunk_NNN.wav
        plus a manifest row. Silently no-op when archiving is disabled.
        """
        if self._audio_archiver is None:
            return
        try:
            wav_path = self._audio_archiver.write_chunk(
                idx=int(event["idx"]),
                t_start=float(event["t_start"]),
                t_end=float(event["t_end"]),
                session_dir=self.timeline.session_dir,
                canonical_text=str(event.get("text", "")),
            )
        except Exception:
            return
        if wav_path:
            self.console.print(
                f"[dim]🎙️  audio → {os.path.basename(wav_path)}[/dim]"
            )

    # ------------------------------------------------------------------
    # HQ stream lifecycle (live enable / disable)
    # ------------------------------------------------------------------

    def _build_hq_stream(self) -> None:
        """Instantiate, register, and start the HQ stream. Called at init
        when CFG.persistent.hq.enabled is True, or at runtime from
        apply_config when the user enables it via the settings page.

        Idempotent: no-op if hq_stream is already running.
        """
        if self.hq_stream is not None:
            return
        hq_cfg = CFG.persistent.hq
        self._hq_audio_q = self.audio_fanout.subscribe(maxsize=400)
        self.hq_stream = TranscriptionStream(
            label="hq",
            audio_input_q=self._hq_audio_q,
            stop_event=self._stop,
            model_name=hq_cfg.fw.model,
            device=hq_cfg.fw.device or self._device,
            compute_type=hq_cfg.fw.compute,
            max_window_seconds=hq_cfg.max_window_seconds,
            silence_ms=max(SILENCE_MS, 1200),
            vad_aggressiveness=VAD_AGGRESSIVENESS,
            min_voiced_ms=MIN_VOICED_MS,
            min_voiced_frac=MIN_VOICED_FRAC,
            beam_size=hq_cfg.fw.beam_size,
            condition_on_previous_text=hq_cfg.fw.condition_on_previous_text,
            window_q_maxsize=hq_cfg.window_q_maxsize,
            debug_log=self._state.log,
            on_segment=self._on_segment,
        )
        self.streams.append(self.hq_stream)
        # Register with the timeline. If HQ is being enabled mid-session
        # (recorder already running), tag its coverage_start_time so the
        # canonical render preserves the pre-enable fast text.
        coverage_start = 0.0
        if self.recorder.started_monotonic is not None:
            now_session = self._cursor_time_at(time.monotonic())
            if now_session > 0:
                coverage_start = now_session
        self.timeline.register_stream(
            "hq",
            priority=1,
            model_label=self.hq_stream.model_label,
            coverage_start_time=coverage_start,
        )

    def _start_hq_stream(self) -> None:
        """Load the HQ model + start its threads. Separate from build so
        the heavy model load happens during ``run()`` (or apply_config)
        rather than in the constructor."""
        if self.hq_stream is not None:
            self.hq_stream.start()

    def _teardown_hq_stream(self) -> None:
        """Stop, drain, and discard the HQ stream. Leaves the fast stream
        untouched."""
        if self.hq_stream is None:
            return
        self.console.print("[yellow]Stopping HQ stream…[/yellow]")
        hq = self.hq_stream
        if self._hq_audio_q is not None:
            self.audio_fanout.unsubscribe(self._hq_audio_q)
            self._hq_audio_q = None
        # Unregister from timeline so no new segments are accepted; existing
        # HQ-contributed segments stay in the canonical render.
        self.timeline.unregister_stream("hq")
        hq.teardown()
        # Drain any leftover segments through on_segment → timeline.
        hq.drain_remaining_segments()
        try:
            self.streams.remove(hq)
        except ValueError:
            pass
        self.hq_stream = None
        self.console.print("[dim]HQ stream stopped.[/dim]")

    # ------------------------------------------------------------------
    # Live configuration (widget /config endpoints)
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return the running config as a JSON-safe dict.

        INVARIANT: every field exposed here MUST be editable in the widget
        settings modal (widget.py, the form fields with data-path
        attributes). If you add a new field to ``RuntimeConfig``, also add
        a matching form control in the modal so users never need to hand-
        edit local_config.json.
        """
        return to_dict(CFG)

    def apply_config(self, new_data: dict) -> dict:
        """Merge ``new_data`` into CFG, persist to local_config.json, and
        apply runtime changes where supported (model hot-swap for the fast
        / hq streams, live enable/disable for the HQ stream; other fields
        take effect on next launch).

        INVARIANT: every field accepted here corresponds to a control in
        the widget settings modal (see ``get_config``). When extending the
        config schema, update BOTH the modal form AND any live-apply logic
        below so the user can edit the new field without restarting.

        Returns a result dict describing what was applied vs. queued
        for next launch.
        """
        applied: list[str] = []
        deferred: list[str] = []

        old_fast_model = CFG.persistent.fast.fw.model
        old_fast_beam = CFG.persistent.fast.fw.beam_size
        old_fast_prev = CFG.persistent.fast.fw.condition_on_previous_text
        old_hq_model = CFG.persistent.hq.fw.model
        old_hq_beam = CFG.persistent.hq.fw.beam_size
        old_hq_prev = CFG.persistent.hq.fw.condition_on_previous_text
        old_hq_enabled = CFG.persistent.hq.enabled

        apply_dict_to_config(CFG, new_data)
        saved_path = save_runtime_config(CFG)

        # Fast stream model swap.
        fast_fw = CFG.persistent.fast.fw
        if (fast_fw.model != old_fast_model or fast_fw.beam_size != old_fast_beam
                or fast_fw.condition_on_previous_text != old_fast_prev):
            self.console.print(
                f"[yellow]Hot-swap fast → {fast_fw.model} "
                f"(beam={fast_fw.beam_size}, prev_text={fast_fw.condition_on_previous_text})[/yellow]"
            )
            self.fast_stream.swap_model(
                model_name=fast_fw.model,
                device=fast_fw.device or self._device,
                compute_type=fast_fw.compute,
                beam_size=fast_fw.beam_size,
                condition_on_previous_text=fast_fw.condition_on_previous_text,
            )
            self.timeline.update_model_label("fast", self.fast_stream.model_label)
            applied.append("fast stream model swap")

        # HQ stream model swap (only if it was enabled — enable/disable still
        # requires a restart).
        hq_fw = CFG.persistent.hq.fw
        if old_hq_enabled and self.hq_stream is not None:
            if (hq_fw.model != old_hq_model or hq_fw.beam_size != old_hq_beam
                    or hq_fw.condition_on_previous_text != old_hq_prev):
                self.console.print(
                    f"[yellow]Hot-swap hq → {hq_fw.model} "
                    f"(beam={hq_fw.beam_size}, prev_text={hq_fw.condition_on_previous_text})[/yellow]"
                )
                self.hq_stream.swap_model(
                    model_name=hq_fw.model,
                    device=hq_fw.device or self._device,
                    compute_type=hq_fw.compute,
                    beam_size=hq_fw.beam_size,
                    condition_on_previous_text=hq_fw.condition_on_previous_text,
                )
                self.timeline.update_model_label("hq", self.hq_stream.model_label)
                applied.append("hq stream model swap")
        # Live enable/disable of the HQ stream.
        if CFG.persistent.hq.enabled != old_hq_enabled:
            if CFG.persistent.hq.enabled:
                # Was off, now on: build + start.
                try:
                    self._build_hq_stream()
                    self._start_hq_stream()
                    applied.append("hq stream enabled")
                except Exception as e:
                    # Roll back any half-built state so a retry can succeed
                    # and so the widget's status / settings page reflects
                    # the true (still-disabled) state.
                    self.console.print(f"[red]HQ enable failed: {e}[/red]")
                    deferred.append(f"hq.enabled failed: {e}")
                    if self.hq_stream is not None:
                        try:
                            if self._hq_audio_q is not None:
                                self.audio_fanout.unsubscribe(self._hq_audio_q)
                                self._hq_audio_q = None
                            self.timeline.unregister_stream("hq")
                            self.streams.remove(self.hq_stream)
                        except Exception:
                            pass
                        self.hq_stream = None
                    CFG.persistent.hq.enabled = False
                    # Re-persist the rolled-back value so the saved file
                    # doesn't say "enabled" when the stream isn't running.
                    save_runtime_config(CFG)
            else:
                # Was on, now off: teardown.
                try:
                    self._teardown_hq_stream()
                    applied.append("hq stream disabled")
                except Exception as e:
                    self.console.print(f"[red]HQ disable failed: {e}[/red]")
                    deferred.append(f"hq.enabled failed: {e}")

        return {
            "saved_to": saved_path,
            "applied": applied,
            "deferred": deferred,
        }

    # ------------------------------------------------------------------
    # Hotkey actions (called from pynput listener thread)
    # ------------------------------------------------------------------

    def _ack(self, command: str) -> None:
        """Print an immediate 'recognized' line — separate from execution."""
        self.console.print(f"[dim]› recognized: {command}[/dim]")

    def _done(self, message: str, style: str = "bold green") -> None:
        """Print an 'executed' line for a command's actual effect."""
        self.console.print(f"[{style}]✓ executed:[/{style}] {message}")

    def _dispatch_user_action(
        self, action: str, key: str, session_time: float
    ) -> None:
        """Forward a committed user_action from UserActionProducer into
        SessionState. SessionState._apply_user_action does all the work
        — mutates timeline / clipboard manager / audio fanout, fires the
        paste callback. There is no separate per-action handler path."""
        self._state.log("user_action", {
            "action": action,
            "key": key,
            "session_time": round(float(session_time), 3),
        })

    def _cursor_time_at(self, monotonic_ts: float) -> float:
        """Estimate the session-relative time corresponding to a wall-clock press."""
        started = self.recorder.started_monotonic or monotonic_ts
        return max(0.0, monotonic_ts - started)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._print_banner()
        self.recorder.start()
        # Anchor debug timestamps to the same monotonic origin the recorder
        # uses, so all event ts values are session-relative seconds.
        self._state.bind_origin(self.recorder.started_monotonic or time.monotonic())
        # Per-stream JSONL persistence: every event is appended to one of
        # stream_fast / stream_hq / stream_user / stream_observability under
        # the session dir. Replay merges these by (ts, seq) and feeds them
        # back through SessionState to rebuild the widget state — so we
        # don't need a separate periodic status_snapshots.jsonl any more.
        if CFG.persistent.debug_recording:
            self._state.set_output_dir(self.timeline.session_dir)
            self.console.print(
                f"[dim]🎬 debug recording → stream_fast/hq/user/observability.jsonl[/dim]"
            )
        self.audio_fanout.start()
        if self._audio_archiver is not None:
            self._audio_archiver.start()
            self.console.print(
                f"[dim]🎙️ saving per-chunk audio → "
                f"{os.path.join(self.timeline.session_dir, 'audio')}[/dim]"
            )
        self._stream_server.start()
        # Start each stream individually so a single stream's startup
        # failure (most commonly an HQ model that isn't cached yet) is
        # surfaced as a one-line error instead of a 50-line traceback.
        # Fast-stream failure is fatal — the app can't function without
        # live transcription — so we exit cleanly. HQ-stream failure is
        # recoverable: drop it from self.streams and run fast-only.
        for s in list(self.streams):
            try:
                s.start()
            except RuntimeError as e:
                # Our own clean error (from StreamingTranscriber.start).
                if s is self.fast_stream:
                    self.console.print(
                        f"[bold red]✗ Failed to start fast stream:[/bold red]\n"
                        f"[red]{e}[/red]"
                    )
                    self._stop.set()
                    self._shutdown()
                    sys.exit(1)
                self.console.print(
                    f"[bold yellow]✗ Failed to start {s.label} stream — "
                    f"continuing without it:[/bold yellow]\n"
                    f"[yellow]{e}[/yellow]"
                )
                self.streams.remove(s)
                if s is self.hq_stream:
                    self.hq_stream = None
                    if self._hq_audio_q is not None:
                        self.audio_fanout.unsubscribe(self._hq_audio_q)
                        self._hq_audio_q = None
                    try:
                        self.timeline.unregister_stream("hq")
                    except Exception:
                        pass
            except Exception as e:
                # Unexpected error — still report cleanly, but include the
                # type so it's debuggable from the console.
                if s is self.fast_stream:
                    self.console.print(
                        f"[bold red]✗ Failed to start fast stream "
                        f"({e.__class__.__name__}):[/bold red] {e}"
                    )
                    self._stop.set()
                    self._shutdown()
                    sys.exit(1)
                self.console.print(
                    f"[bold yellow]✗ {s.label} stream startup failed "
                    f"({e.__class__.__name__}):[/bold yellow] {e}"
                )
                self.streams.remove(s)
                if s is self.hq_stream:
                    self.hq_stream = None

        if self._enable_widget:
            try:
                self._widget = widget_module.StatusServer(
                    self.get_status_snapshot,
                    config_provider=self.get_config,
                    config_setter=self.apply_config,
                )
                try:
                    host, port = self._widget.start(port=WIDGET_PORT)
                except OSError:
                    # Fixed port in use (e.g. another instance still running);
                    # fall back to an ephemeral port so the app still launches.
                    self.console.print(
                        f"[yellow]Widget port {WIDGET_PORT} in use; using ephemeral port[/yellow]"
                    )
                    host, port = self._widget.start(port=0)
                url = f"http://{host}:{port}"
                self.console.print(f"[bold]Widget:[/bold] {url}")
                if self._open_browser:
                    webbrowser.open(url)
            except Exception as e:
                self.console.print(f"[yellow]Widget failed to start: {e}[/yellow]")
                self._widget = None

        self._user_actions.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._stop.set()
        self._user_actions.stop()
        self._shutdown()

    def get_status_snapshot(self) -> dict:
        """Return the JSON payload served by the widget's /status endpoint.

        Just refreshes the two live-only counters (per-stream drop stats
        and hq-active flag) and asks SessionState to build the payload.
        Component attachments happen once at __init__."""
        stream_stats = {
            s.label: {
                "accepted": s.windows_accepted_count,
                "dropped_queue_full": s.dropped_queue_full_count,
                "dropped_silent": s.dropped_silent_count,
            }
            for s in self.streams
        }
        self._state.set_stream_stats(stream_stats)
        self._state.set_hq_active(self.hq_stream is not None)
        return self._state.snapshot()

    def _shutdown(self) -> None:
        self.console.print("\n[bold yellow]Shutting down — flushing...[/bold yellow]")
        if self._widget is not None:
            try:
                self._widget.stop()
                self.console.print("[dim]Widget: stopped[/dim]")
            except Exception:
                pass
            self._widget = None
        self.recorder.stop()
        # Give the aggregators a moment to drain remaining frames.
        time.sleep(0.5)
        self.audio_fanout.stop()
        for s in self.streams:
            s.stop()
        # Drain any final segments from each stream.
        time.sleep(0.5)
        for s in self.streams:
            s.drain_remaining_segments()
        # Final chunk flush + finalize via the unified timeline (writes
        # canonical + per-stream raw chunks, transcript_final.{txt,json},
        # and session_meta.json in one go).
        flushed = self.timeline.force_flush()
        for label, path in flushed.items():
            if path:
                self.console.print(
                    f"[dim]💾 final {label} chunk → {os.path.basename(path)}[/dim]"
                )
        finalize_paths = self.timeline.finalize()
        if "transcript_final_txt" in finalize_paths:
            self.console.print("[dim]💾 final transcript → transcript_final.txt[/dim]")
        # Close debug-recording JSONL handles cleanly so files end with
        # a newline-terminated last record.
        self._state.close_output()
        self.console.print(
            f"[bold green]✓ Session saved:[/bold green] {self.timeline.session_dir}"
        )

    def _format_stream_summary(self) -> str:
        parts = []
        for s in self.streams:
            tx = s.transcriber
            parts.append(
                f"{s.label}={tx.model_label}({tx.device}/{tx.compute_type}, "
                f"beam={tx.beam_size}, prev_text={tx.condition_on_previous_text})"
            )
        return " | ".join(parts)

    def _print_banner(self) -> None:
        marker_lines = "\n".join(
            (
                f"  [bold]{m.key} (x2)[/bold] - flag marker: {m.type}  [dim]({m.description})[/dim]"
                if m.flag else
                f"  [bold]{m.key} (x2)[/bold] - open/close marker: {m.type}  [dim]({m.description})[/dim]"
            )
            for m in self.marker_types
        )
        body = (
            "[bold cyan]Persistent Voice Dictation[/bold cyan]\n\n"
            "[dim]Always-on transcription. Audio is NOT saved.[/dim]\n\n"
            "Hotkeys (bare double-tap, no Ctrl/Cmd held):\n"
            f"{marker_lines}\n"
            "  [bold]r (x2)[/bold] - clipboard window: paste captured text on second tap\n"
            "  [bold]a (x2)[/bold] - aside clipboard window\n"
            "  [bold]x (x2)[/bold] - cancel current window (aside if active, else r)\n"
            "  [bold]e (x2)[/bold] - flag an error in this moment (visible in widget + jsonl)\n"
            "  [bold]q (x2)[/bold] - quit + flush\n\n"
            f"[dim]Session: {self.timeline.session_id}[/dim]\n"
            f"[dim]Streams: {self._format_stream_summary()}[/dim]\n"
            f"[dim]Output:  {self.timeline.session_dir}[/dim]"
        )
        self.console.print(Panel.fit(body, title="🎤 Persistent Mode",
                                      border_style="cyan"))


def main():
    parser = argparse.ArgumentParser(description="Persistent voice dictation")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config JSON (default: local_config.json)")
    parser.add_argument("--model", default=FW_MODEL,
                        help=f"faster-whisper model name (default: {FW_MODEL})")
    parser.add_argument("--device", default=FW_DEVICE,
                        help=f"compute device (default: {FW_DEVICE})")
    parser.add_argument("--compute", default=FW_COMPUTE,
                        help=f"compute type (default: {FW_COMPUTE})")
    parser.add_argument("--no-widget", action="store_true",
                        help="Disable the HTTP status widget server entirely.")
    parser.add_argument("--open-browser", action="store_true",
                        help="Auto-open the widget URL in the default browser. "
                             "Default is to print the URL only (open it manually, "
                             "e.g. in VS Code's Simple Browser).")
    parser.add_argument("--check-updates", action="store_true",
                        help="Allow HuggingFace Hub network check for newer "
                             "model versions on this run. Default is offline "
                             "load from cache for fast, network-independent "
                             "startup.")
    parser.add_argument(
        "--save-audio",
        dest="save_audio",
        default=None,
        action="store_const",
        const=True,
        help="Persist per-chunk audio (chunk_NNN.wav) under the session "
             "dir for later annotation / fine-tuning. Overrides the "
             "persistent.save_audio config flag for this run.",
    )
    parser.add_argument(
        "--no-save-audio",
        dest="save_audio",
        action="store_const",
        const=False,
        help="Disable per-chunk audio persistence for this run "
             "(overrides persistent.save_audio config flag).",
    )
    args = parser.parse_args()

    if not args.check_updates:
        # Print a one-line nudge if it's been a while since we last checked.
        try:
            mtime = os.path.getmtime(_HF_CHECK_FILE)
            stale = (time.time() - mtime) > _HF_CHECK_STALE_SECONDS
        except OSError:
            stale = True
        if stale:
            days = "never"
            try:
                age = int((time.time() - mtime) / 86400)
                days = f"{age}d ago"
            except Exception:
                pass
            print(
                f"[hf] model update check skipped (last: {days}). "
                f"Run with --check-updates to refresh.",
                file=sys.stderr,
            )

    config = load_config(args.config)
    app = PersistentApp(
        config=config,
        model=args.model,
        device=args.device,
        compute=args.compute,
        enable_widget=not args.no_widget,
        open_browser=args.open_browser,
        save_audio_override=args.save_audio,
    )
    if args.check_updates:
        # If the model loaded successfully (PersistentApp constructor builds
        # the StreamingTranscriber but doesn't load the model — that happens
        # in app.run() via transcriber.start). Touch the tracker after run()
        # would be too late (only on shutdown). Touch right before run() —
        # the model load is the next step and any failure there is loud.
        try:
            with open(_HF_CHECK_FILE, "w") as f:
                f.write(str(int(time.time())))
        except OSError:
            pass
    app.run()


if __name__ == "__main__":
    main()
