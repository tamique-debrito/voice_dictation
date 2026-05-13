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

from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

from audio_fanout import AudioFanout
from clipboard_manager import ClipboardManager
from clipboard_window import (
    ClipboardWindow,
    ClipboardWindowManager,
    ClipboardWindowType,
)
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
from hotkeys import BareDoubleTap, key_char, load_config
from persistent_recorder import PersistentRecorder
from runtime_config import CFG, apply_dict_to_config, save_runtime_config, to_dict
from status_snapshot import build_status
from transcript_timeline import MarkerType, TranscriptTimeline
from transcription_stream import TranscriptionStream
import widget as widget_module


# Maximum time to wait for the transcriber to catch up with a clipboard-window
# end-press before pasting whatever we have.
PASTE_DRAIN_TIMEOUT = 6.0


class DebugLog:
    """Thread-safe ring buffer of typed debug events.

    Each event is ``{"ts": session_seconds, "kind": str, "data": dict}``.
    The widget reads a snapshot via ``snapshot()`` from the status payload.
    """

    def __init__(self, maxlen: int = 2000):
        self._events: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._origin: Optional[float] = None
        # Optional append-only persistence — when ``set_output_path`` is
        # called, every event written to the ring is also flushed to a
        # JSONL file for post-session replay.
        self._out_path: Optional[str] = None
        self._out_file = None

    def bind_origin(self, origin_monotonic: float) -> None:
        self._origin = origin_monotonic

    def set_output_path(self, path: str) -> None:
        """Open a JSONL file at ``path`` to persist all events for replay."""
        try:
            self._out_file = open(path, "a", buffering=1, encoding="utf-8")
            self._out_path = path
        except OSError:
            self._out_file = None

    def close_output(self) -> None:
        if self._out_file is not None:
            try:
                self._out_file.close()
            except Exception:
                pass
            self._out_file = None

    def _now(self) -> float:
        if self._origin is None:
            return 0.0
        return time.monotonic() - self._origin

    def log(self, kind: str, data: dict, ts: Optional[float] = None) -> None:
        evt = {
            "ts": round(ts if ts is not None else self._now(), 3),
            "kind": kind,
            "data": data,
        }
        with self._lock:
            self._events.append(evt)
            if self._out_file is not None:
                try:
                    import json as _json
                    self._out_file.write(_json.dumps(evt) + "\n")
                except Exception:
                    pass

    def snapshot(self, limit: int = 500) -> list[dict]:
        with self._lock:
            if limit >= len(self._events):
                return list(self._events)
            return list(self._events)[-limit:]


class PersistentApp:
    def __init__(self, config: dict, model: str, device: str, compute: str,
                 enable_widget: bool = True, open_browser: bool = False):
        self.console = Console()
        self.config = config
        self._stop = threading.Event()
        self._enable_widget = enable_widget
        self._open_browser = open_browser
        self._widget: Optional[widget_module.StatusServer] = None
        self._started_at = datetime.now(tz=timezone.utc)
        self._device = device
        self._compute = compute

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

        # Granular debug-event ring buffer surfaced to the widget.
        self._debug = DebugLog(maxlen=2000)

        # AudioFanout drains raw_q, owns mute, and broadcasts frames to per-
        # stream input queues. Mute transitions emit a MUTE_RESET sentinel
        # that downstream aggregators consume to reset partial windows.
        self.audio_fanout = AudioFanout(self.raw_q, self._stop)

        # The unified canonical transcript. Owns: per-stream segment
        # buffers, markers (one list, anchored to audio_time), chunk files
        # (canonical + per-stream raw). All transcription streams ingest
        # segments into this single object.
        self.timeline = TranscriptTimeline(marker_types=self.marker_types)

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
            debug_log=self._debug.log,
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

        # Status-widget state. paste_log is most-recent-first when serialized.
        self._paste_log: deque = deque(maxlen=50)
        self._marker_open_since: Optional[str] = None

        self._kb = keyboard.Controller()

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
        self.clipboard_mgr = ClipboardWindowManager(
            types=clipboard_types,
            timeline=self.timeline,
            debug_log=self._debug.log,
            request_force_flush=self._request_force_flush_all,
            cursor_time_at=self._cursor_time_at,
        )

        # Bare double-tap dispatcher. `e` is the "error" / debug-flag key:
        # pressing it stamps a high-visibility event into the debug log
        # (and the persisted debug_events.jsonl) so post-session replay
        # can jump straight to "the moment something went wrong" instead
        # of grepping through the full timeline.
        keys = (
            set(self._marker_keys)
            | self.clipboard_mgr.keys()
            | {"x", "q", "m", CFG.persistent.debug_flag_key}
        )
        self._dt = BareDoubleTap(window=1.0, keys=keys, on_double_tap=self._on_action)

    def _request_force_flush_all(self) -> None:
        """Fire force-flush on every active transcription stream."""
        for s in self.streams:
            s.request_force_flush()

    def _on_segment(self, stream_id: str, seg) -> None:
        """Callback wired into every TranscriptionStream's drain loop.

        Forwards the segment to the timeline (which routes it into the
        per-stream buffer and re-renders canonical on demand) and, for the
        fast stream only, broadcasts on the live transcript WebSocket.
        """
        try:
            self.timeline.ingest_segment(
                stream_id,
                seg.text,
                seg.start_time,
                seg.end_time,
                seg.words,
            )
        except Exception:
            pass
        if stream_id == "fast":
            try:
                self._stream_server.broadcast(seg.text, seg.end_time)
            except Exception:
                pass

    def _snapshot_writer_loop(self) -> None:
        """Periodically write get_status_snapshot() to status_snapshots.jsonl
        under the session dir so the full widget state can be replayed.

        Trims ``debug_events`` to an empty list on the disk copy — the
        ring-buffer events are already being persisted line-by-line in
        ``debug_events.jsonl``; duplicating them in every snapshot would
        bloat the file by O(snapshots × events) for no gain.
        """
        import json as _json
        path = os.path.join(self.timeline.session_dir, "status_snapshots.jsonl")
        try:
            fh = open(path, "a", buffering=1, encoding="utf-8")
        except OSError:
            return
        try:
            interval = float(CFG.persistent.debug_snapshot_interval_s) or 2.0
            while not self._stop.is_set():
                try:
                    snap = self.get_status_snapshot()
                    snap_copy = dict(snap)
                    snap_copy["debug_events"] = []  # see docstring
                    fh.write(_json.dumps(snap_copy) + "\n")
                except Exception:
                    pass
                if self._stop.wait(interval):
                    break
        finally:
            try:
                fh.close()
            except Exception:
                pass

    def _on_silence_boundary_fast(self) -> None:
        """Fast-aggregator silence-boundary tick. Drives canonical chunk
        flushing (and per-stream raw chunk ticking) on the timeline."""
        try:
            path = self.timeline.tick(at_silence_boundary=True)
        except Exception:
            return
        if path:
            self.console.print(f"[dim]💾 chunk → {os.path.basename(path)}[/dim]")

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
            debug_log=self._debug.log,
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

    def _undo_keypresses(self, count: int = 2) -> None:
        """Send N backspaces to undo the double-tap that just landed in
        whichever app currently has keyboard focus. Best-effort — in most
        non-text contexts backspace is a no-op."""
        try:
            for _ in range(count):
                self._kb.tap(keyboard.Key.backspace)
        except Exception:
            pass

    def _done(self, message: str, style: str = "bold green") -> None:
        """Print an 'executed' line for a command's actual effect."""
        self.console.print(f"[{style}]✓ executed:[/{style}] {message}")

    def _on_action(self, char: str) -> None:
        # Immediately undo the two visible keystrokes in the focused app.
        # Done before printing so the cleanup happens before any user-visible
        # delay from the action itself.
        self._undo_keypresses(2)
        if char in self._marker_keys:
            self._ack(f"marker `{char}`")
            self._handle_marker(char)
        elif char in self.clipboard_mgr.keys():
            label = self.clipboard_mgr.type_for(char).label
            phase = f"{label} end" if self.clipboard_mgr.is_active(char) else f"{label} start"
            self._ack(phase)
            self._toggle_clipboard_window(char)
        elif char == "x":
            self._ack("cancel (x)")
            self._cancel_window()
        elif char == "m":
            self._ack("mute toggle (m)")
            self._toggle_mute()
        elif char == CFG.persistent.debug_flag_key:
            self._ack(f"error flag ({char})")
            self._flag_issue()
        elif char == "q":
            self._ack("quit (q)")
            self._stop.set()
            self._done("shutting down", style="bold yellow")

    def _handle_marker(self, key: str) -> None:
        events = self.timeline.insert_marker(key)
        if not events:
            return
        for action, type_name in events:
            self._debug.log("marker", {
                "type": type_name, "action": action, "key": key,
            })
        for action, type_name in events:
            if action == "flag":
                self._done(f"flagged [magenta]{type_name}[/magenta]")
                continue
            verb = "opened" if action == "open" else "closed"
            note = ""
            if len(events) > 1 and action == "close":
                # Cross-type switch — make the auto-close obvious.
                note = " [dim](auto, switching type)[/dim]"
            self._done(f"{verb} marker [magenta]{type_name}[/magenta]{note}")
        # Update widget-visible "open since" timestamp.
        with self._state_lock:
            self._marker_open_since = (
                datetime.now().strftime("%H:%M:%S")
                if self.timeline.open_marker_type() is not None else None
            )

    def _toggle_clipboard_window(self, key: str) -> None:
        result = self.clipboard_mgr.toggle(key)
        if result.action == "start_blocked":
            self._done(f"[yellow]{result.error_msg}[/yellow]", style="yellow")
            return
        if result.action == "started":
            self._done(f"started {result.type_label} capture")
            return
        if result.action == "ended" and result.closed_window is not None:
            threading.Thread(
                target=self._finish_paste,
                args=(result.closed_window,),
                daemon=True,
            ).start()

    def _flag_issue(self) -> None:
        """Stamp a high-visibility ``debug_flag`` event with the current
        session time. Captured by the debug ring buffer (so the widget
        timeline renders it immediately) and by ``debug_events.jsonl``
        (so post-session replay can jump straight to it).

        Useful workflow: when something feels off during a session — a
        missed word, a botched paste, an HQ stream lag spike — double-tap
        ``b`` to bookmark the exact moment. Then post-session you can
        replay or grep the JSONL for ``"kind": "debug_flag"`` to land at
        the right spot without combing through the whole timeline.
        """
        # Snapshot useful context so the flag is self-describing in JSONL.
        try:
            counters = {
                s.label: {
                    "accepted": s.windows_accepted_count,
                    "dropped_queue_full": s.dropped_queue_full_count,
                    "dropped_silent": s.dropped_silent_count,
                }
                for s in self.streams
            }
        except Exception:
            counters = {}
        try:
            hq_edge = self.timeline.hq_high_watermark()
            fast_edge = self.timeline.fast_high_watermark()
        except Exception:
            hq_edge = 0.0
            fast_edge = 0.0
        self._debug.log("debug_flag", {
            "wall_clock": datetime.now().strftime("%H:%M:%S"),
            "fast_watermark": round(fast_edge, 3),
            "hq_watermark": round(hq_edge, 3),
            "stream_stats": counters,
            "capture_mode": (
                "r" if self.clipboard_mgr.is_active("r") else
                "aside" if self.clipboard_mgr.is_active("a") else
                "passive"
            ),
        })
        self._done("flagged this moment for debug — see debug_events.jsonl",
                   style="bold magenta")

    def _toggle_mute(self) -> None:
        now_muted = self.audio_fanout.toggle_muted()
        if now_muted:
            self._debug.log("mute", {"state": "on"})
            self._done("muted — transcription paused", style="bold yellow")
        else:
            self._debug.log("mute", {"state": "off"})
            self._done("unmuted — transcription resumed")

    def _cancel_window(self) -> None:
        result = self.clipboard_mgr.cancel()
        if result.nothing:
            self._done("nothing to cancel", style="dim")
            return
        if result.resumed_labels:
            suffix = f" ({', '.join(result.resumed_labels)} resumed)"
        else:
            suffix = ""
        self._done(f"{result.cancelled_label} cancelled{suffix}")

    def _finish_paste(self, window: ClipboardWindow) -> None:
        # Wait for the transcriber to drain past the end-press timestamp.
        wait_started = time.monotonic()
        deadline = window.end_press_time + PASTE_DRAIN_TIMEOUT
        target = self._cursor_time_at(window.end_press_time)
        timed_out = True
        while time.monotonic() < deadline:
            if self.timeline.latest_segment_end() >= target:
                timed_out = False
                break
            time.sleep(0.1)
        drain_wait_ms = int((time.monotonic() - wait_started) * 1000)
        # Materialize any pending markers (notably the recording-end token
        # for THIS press) by re-anchoring them to the current fast
        # watermark, so they surface in the canonical render BEFORE we
        # slice for paste. Otherwise the end marker would only appear once
        # later audio crosses the press time — by then the slice is gone.
        drained = self.timeline.flush_pending_markers()
        if drained:
            self._debug.log("marker", {
                "type": "recording", "action": "drained_pending",
                "key": "-", "count": drained,
            })
        # Close the live span at the current cursor (in canonical render)
        # and paste.
        window.close(self.timeline.cursor())
        text = window.text(self.timeline).strip()
        self._debug.log("paste", {
            "label": window.label,
            "start_press_time": round(window.start_press_time, 3),
            "end_press_time": round(self._cursor_time_at(window.end_press_time), 3),
            "spans": [list(s) for s in window.spans],
            "drain_wait_ms": drain_wait_ms,
            "timed_out": timed_out,
            "char_count": len(text),
            "preview": text[:200],
        })
        if not text:
            self._done(f"{window.label} → nothing to paste", style="yellow")
            return
        self.clipboard.copy_and_paste(text)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        self._done(f"{window.label} pasted → [green]{preview}[/green]")
        # Record for the widget paste log (chronological — newest last so the
        # UI can simply scroll to the bottom to show the tail).
        entry = {
            "ts": datetime.now().strftime("%H:%M:%S"),
            "label": window.label,
            "preview": text[:120] + ("…" if len(text) > 120 else ""),
            "full": text,
        }
        with self._state_lock:
            self._paste_log.append(entry)

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
        self._debug.bind_origin(self.recorder.started_monotonic or time.monotonic())
        # Optional post-session replay persistence: write debug events as
        # they arrive to a JSONL under the session dir, plus periodic
        # status snapshots. Together these are enough to replay everything
        # the live widget showed.
        self._snapshot_thread: Optional[threading.Thread] = None
        if CFG.persistent.debug_recording:
            evt_path = os.path.join(self.timeline.session_dir, "debug_events.jsonl")
            self._debug.set_output_path(evt_path)
            self._snapshot_thread = threading.Thread(
                target=self._snapshot_writer_loop,
                name="SnapshotWriter",
                daemon=True,
            )
            self._snapshot_thread.start()
            self.console.print(
                f"[dim]🎬 debug recording → debug_events.jsonl, "
                f"status_snapshots.jsonl[/dim]"
            )
        self.audio_fanout.start()
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

        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._stop.set()
        listener.stop()
        self._shutdown()

    def get_status_snapshot(self) -> dict:
        """Return the JSON payload served by the widget's /status endpoint."""
        with self._state_lock:
            paste_log = list(self._paste_log)
            marker_open_since = self._marker_open_since
        stream_stats = {
            s.label: {
                "accepted": s.windows_accepted_count,
                "dropped_queue_full": s.dropped_queue_full_count,
                "dropped_silent": s.dropped_silent_count,
            }
            for s in self.streams
        }
        return build_status(
            timeline=self.timeline,
            hq_active=self.hq_stream is not None,
            device=self._device,
            compute=self._compute,
            started_at=self._started_at,
            audio_fanout=self.audio_fanout,
            clipboard_mgr=self.clipboard_mgr,
            paste_log=paste_log,
            marker_open_since=marker_open_since,
            recorder=self.recorder,
            debug=self._debug,
            stream_stats=stream_stats,
        )

    def _on_press(self, key) -> None:
        try:
            self._dt.feed(key_char(key))
        except Exception:
            pass

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
        self._debug.close_output()
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
