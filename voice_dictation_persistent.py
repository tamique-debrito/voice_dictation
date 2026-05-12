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

from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

from clipboard_manager import ClipboardManager
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
    SAMPLE_RATE,
    SILENCE_MS,
    TRANSCRIPT_STREAM_PORT,
    VAD_AGGRESSIVENESS,
)
from transcript_stream_server import TranscriptStreamServer
from hotkeys import BareDoubleTap, key_char, load_config
from persistent_recorder import PersistentRecorder
from session_writer import MarkerType, SessionWriter
from silence_detector import SilenceDetector
from streaming_transcriber import AudioWindow, Segment, StreamingTranscriber
import widget as widget_module


# How long an audio window can grow before we force-emit it for transcription
# even without a silence boundary. Keeps latency bounded for monologues.
MAX_WINDOW_SECONDS = 25.0
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

    def bind_origin(self, origin_monotonic: float) -> None:
        self._origin = origin_monotonic

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

    def snapshot(self, limit: int = 500) -> list[dict]:
        with self._lock:
            if limit >= len(self._events):
                return list(self._events)
            return list(self._events)[-limit:]


class _ClipboardWindow:
    """Tracks an active r/aside paste window across aside park/resume."""

    def __init__(
        self,
        label: str,
        start_cursor: Optional[int],
        end_press_time: float,
        start_press_time: float,
    ):
        self.label = label
        # spans: list of (start, end). start can be None until the scheduled
        # cursor-capture callback fires; end is None while the span is live.
        self.spans: list[tuple[Optional[int], Optional[int]]] = [(start_cursor, None)]
        self.end_press_time = end_press_time
        self.start_press_time = start_press_time

    def set_start_cursor(self, cursor: int) -> None:
        _, e = self.spans[0]
        self.spans[0] = (cursor, e)

    def park(self, cursor: int) -> None:
        s, _ = self.spans[-1]
        self.spans[-1] = (s, cursor)

    def resume(self, cursor: int) -> None:
        self.spans.append((cursor, None))

    def close(self, cursor: int) -> None:
        s, _ = self.spans[-1]
        self.spans[-1] = (s, cursor)

    def text(self, writer: SessionWriter) -> str:
        parts = []
        for i, (s, e) in enumerate(self.spans):
            if s is None:
                # Start cursor never landed (very fast end). Use writer cursor.
                s = writer.cursor()
                self.spans[i] = (s, e)
            # Apply word-level trimming only at the outer edges of the
            # full window (start_press_time on the first span, end_press_time
            # on the last). Internal park/resume spans use plain slicing.
            sp = self.start_press_time if i == 0 else None
            ep = self.end_press_time if i == len(self.spans) - 1 else None
            parts.append(writer.slice_for_paste_by_time(s, e, sp, ep))
        return " ".join(p for p in parts if p)


class PersistentApp:
    def __init__(self, config: dict, model: str, device: str, compute: str,
                 enable_widget: bool = True):
        self.console = Console()
        self.config = config
        self._stop = threading.Event()
        self._enable_widget = enable_widget
        self._widget: Optional[widget_module.StatusServer] = None
        self._started_at = datetime.now(tz=timezone.utc)
        self._device = device
        self._compute = compute

        marker_cfgs = config.get("persistent", {}).get("markers", DEFAULT_MARKERS)
        self.marker_types = [
            MarkerType(key=m["key"], type=m["type"], description=m["description"])
            for m in marker_cfgs
        ]
        self._marker_keys = {m.key for m in self.marker_types}

        self.raw_q: queue.Queue = queue.Queue(maxsize=200)
        self.window_q: queue.Queue = queue.Queue(maxsize=8)
        self.segment_q: queue.Queue = queue.Queue()

        self.recorder = PersistentRecorder(self.raw_q)
        self.transcriber = StreamingTranscriber(
            in_queue=self.window_q,
            out_queue=self.segment_q,
            model_name=model,
            device=device,
            compute_type=compute,
        )
        self.writer = SessionWriter(
            marker_types=self.marker_types,
            model_label=self.transcriber.model_label,
        )
        self.clipboard = ClipboardManager()
        self._stream_server = TranscriptStreamServer(port=TRANSCRIPT_STREAM_PORT)

        self._aggregator_thread: Optional[threading.Thread] = None
        self._drain_thread: Optional[threading.Thread] = None

        # Clipboard-window state (touched only from listener thread).
        self._r_window: Optional[_ClipboardWindow] = None
        self._aside_window: Optional[_ClipboardWindow] = None
        self._state_lock = threading.Lock()

        # Status-widget state. paste_log is most-recent-first when serialized.
        self._paste_log: deque = deque(maxlen=50)
        # Wall-clock HH:MM:SS string of when the currently-open marker opened.
        self._marker_open_since: Optional[str] = None

        # Mute toggle: when set, the aggregator drains raw audio without
        # feeding it to VAD/transcription. Toggled by double-tap `m`.
        self._muted = threading.Event()

        # Force-flush signal: set by clipboard-window end-press to ask the
        # aggregator to immediately flush its in-progress audio window so
        # trailing words reach the transcriber inside the paste-drain budget.
        self._force_flush = threading.Event()

        # Granular debug-event ring buffer surfaced to the widget.
        self._debug = DebugLog(maxlen=2000)

        # Bare double-tap dispatcher
        keys = set(self._marker_keys) | {"r", "a", "x", "q", "m"}
        self._dt = BareDoubleTap(window=1.0, keys=keys, on_double_tap=self._on_action)

        # Used to undo the two visible keypresses in whichever app currently
        # has keyboard focus when a double-tap is recognized.
        self._kb = keyboard.Controller()

    # ------------------------------------------------------------------
    # Aggregator: PCM -> windows
    # ------------------------------------------------------------------

    def _aggregator_loop(self) -> None:
        vad = SilenceDetector(silence_ms=SILENCE_MS, aggressiveness=VAD_AGGRESSIVENESS)
        window_pcm = bytearray()
        window_start: Optional[float] = None
        window_end: float = 0.0
        last_chunk_check = 0.0

        def flush_window(reason: str) -> None:
            nonlocal window_pcm, window_start, window_end
            if not window_pcm or window_start is None:
                window_pcm = bytearray()
                window_start = None
                window_end = 0.0
                vad.reset_counts()
                return
            voiced_ms = SilenceDetector.voiced_ms(vad.voiced_frames)
            voiced_frac = (
                vad.voiced_frames / vad.total_frames if vad.total_frames else 0.0
            )
            ws, we = window_start, window_end
            if voiced_ms < MIN_VOICED_MS or voiced_frac < MIN_VOICED_FRAC:
                self._debug.log("audio_window", {
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": "dropped_silent",
                    "trigger": reason,
                })
                window_pcm = bytearray()
                window_start = None
                window_end = 0.0
                vad.reset_counts()
                return
            try:
                self.window_q.put(
                    AudioWindow(
                        pcm=bytes(window_pcm),
                        start_time=ws,
                        end_time=we,
                    ),
                    timeout=2.0,
                )
                self._debug.log("audio_window", {
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": reason,
                })
            except queue.Full:
                self._debug.log("audio_window", {
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": "dropped_queue_full",
                    "trigger": reason,
                })
            window_pcm = bytearray()
            window_start = None
            window_end = 0.0
            vad.reset_counts()

        while not self._stop.is_set():
            try:
                ts, data = self.raw_q.get(timeout=0.5)
            except queue.Empty:
                # No audio for a moment — try a chunk flush opportunistically.
                self._maybe_flush_chunk(at_silence_boundary=False)
                continue

            if self._muted.is_set():
                # Drop the chunk and discard any in-progress window so we
                # don't flush a partial utterance straddling the mute edge.
                # VAD state is reset too, so re-arming only happens after
                # fresh voiced audio post-unmute.
                if window_pcm:
                    window_pcm = bytearray()
                    window_start = None
                    window_end = 0.0
                    vad.reset_counts()
                continue

            if window_start is None:
                window_start = ts
            window_pcm.extend(data)
            # Each PyAudio chunk is CHUNK_SIZE=1024 samples = 64 ms.
            window_end = ts + (len(data) / 2 / SAMPLE_RATE)

            saw_boundary = False
            for evt in vad.feed(data):
                if evt == "boundary":
                    saw_boundary = True
                    break

            window_seconds = window_end - window_start
            forced = self._force_flush.is_set()
            if forced:
                self._force_flush.clear()
            if forced or saw_boundary or window_seconds >= MAX_WINDOW_SECONDS:
                reason = (
                    "forced" if forced
                    else "silence" if saw_boundary
                    else "max_window"
                )
                flush_window(reason=reason)
                if saw_boundary or forced:
                    # Drain any pending chunk if size threshold is met.
                    self._maybe_flush_chunk(at_silence_boundary=True)

            # Periodic background chunk-flush check (covers MAX_WINDOW splits).
            now = time.monotonic()
            if now - last_chunk_check > 5.0:
                last_chunk_check = now
                self._maybe_flush_chunk(at_silence_boundary=False)

        flush_window(reason="shutdown")

    def _maybe_flush_chunk(self, at_silence_boundary: bool) -> None:
        path = self.writer.maybe_flush_chunk(at_silence_boundary=at_silence_boundary)
        if path:
            self.console.print(f"[dim]💾 chunk → {os.path.basename(path)}[/dim]")

    # ------------------------------------------------------------------
    # Drain: segments -> session writer
    # ------------------------------------------------------------------

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            try:
                seg: Segment = self.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.writer.feed_segment(seg.text, seg.end_time, words=seg.words)
            self._stream_server.broadcast(seg.text, seg.end_time)
            self._debug.log("segment", {
                "text": seg.text,
                "start": round(seg.start_time, 3),
                "end": round(seg.end_time, 3),
                "words": [
                    {"text": w.text, "s": round(w.start_time, 3),
                     "e": round(w.end_time, 3), "p": round(w.probability, 3)}
                    for w in seg.words
                ],
            })

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
        elif char == "r":
            # Distinguish start vs. end at recognition time so the user
            # knows which phase fired.
            phase = "r end" if self._r_window is not None else "r start"
            self._ack(phase)
            self._toggle_r()
        elif char == "a":
            phase = "aside end" if self._aside_window is not None else "aside start"
            self._ack(phase)
            self._toggle_aside()
        elif char == "x":
            self._ack("cancel (x)")
            self._cancel_window()
        elif char == "m":
            self._ack("mute toggle (m)")
            self._toggle_mute()
        elif char == "q":
            self._ack("quit (q)")
            self._stop.set()
            self._done("shutting down", style="bold yellow")

    def _handle_marker(self, key: str) -> None:
        events = self.writer.insert_marker(key)
        if not events:
            return
        for action, type_name in events:
            self._debug.log("marker", {
                "type": type_name, "action": action, "key": key,
            })
        for action, type_name in events:
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
                if self.writer.open_marker_type() is not None else None
            )

    def _toggle_r(self) -> None:
        with self._state_lock:
            if self._r_window is None:
                if self._aside_window is not None:
                    self._done(
                        "[yellow]aside is active; close it with `a` first[/yellow]",
                        style="yellow",
                    )
                    return
                press_ts = time.monotonic()
                press_session = self._cursor_time_at(press_ts)
                # Emit recording-start marker into the transcript.
                self.writer.append_marker(RECORDING_MARKER_TYPE, "start")
                # Schedule the start cursor to land AFTER segments whose
                # audio reaches the press moment, so post-press words don't
                # get buried behind the cursor by a late-arriving window.
                self._r_window = _ClipboardWindow(
                    "r", start_cursor=None,
                    end_press_time=press_ts, start_press_time=press_session,
                )
                w = self._r_window
                requested = time.monotonic()
                def _set_cursor(cursor: int, _w=w, _req=requested,
                                _pt=press_session):
                    _w.set_start_cursor(cursor)
                    self._debug.log("cursor_capture", {
                        "label": "r",
                        "press_time": round(_pt, 3),
                        "fired_latency_ms": int((time.monotonic() - _req) * 1000),
                        "cursor": cursor,
                    })
                self.writer.schedule_cursor_capture(press_session, _set_cursor)
                self._debug.log("press", {
                    "key": "r", "phase": "start",
                    "session_time": round(press_session, 3),
                })
                self._done("started r capture")
            else:
                # End r: defer marker insertion until transcription has
                # caught up to the press moment, so the token lands AFTER
                # any in-flight segments whose audio precedes the press.
                press_ts = time.monotonic()
                press_session = self._cursor_time_at(press_ts)
                # Force the aggregator to flush its in-progress audio so
                # trailing words reach the transcriber inside the
                # paste-drain budget, instead of waiting for VAD silence.
                self._force_flush.set()
                self.writer.schedule_marker(
                    RECORDING_MARKER_TYPE, "end", press_session
                )
                self._r_window.end_press_time = press_ts
                window = self._r_window
                self._r_window = None
                self._debug.log("press", {
                    "key": "r", "phase": "end",
                    "session_time": round(press_session, 3),
                })
                threading.Thread(
                    target=self._finish_paste, args=(window,), daemon=True
                ).start()

    def _toggle_aside(self) -> None:
        with self._state_lock:
            if self._aside_window is None:
                press_ts = time.monotonic()
                press_session = self._cursor_time_at(press_ts)
                self.writer.append_marker(ASIDE_MARKER_TYPE, "start")
                # Park the active r window at the (current) cursor; the
                # aside's own start cursor is scheduled by audio time.
                if self._r_window is not None:
                    self._r_window.park(self.writer.cursor())
                self._aside_window = _ClipboardWindow(
                    "aside", start_cursor=None,
                    end_press_time=press_ts, start_press_time=press_session,
                )
                w = self._aside_window
                requested = time.monotonic()
                def _set_cursor(cursor: int, _w=w, _req=requested,
                                _pt=press_session):
                    _w.set_start_cursor(cursor)
                    self._debug.log("cursor_capture", {
                        "label": "aside",
                        "press_time": round(_pt, 3),
                        "fired_latency_ms": int((time.monotonic() - _req) * 1000),
                        "cursor": cursor,
                    })
                self.writer.schedule_cursor_capture(press_session, _set_cursor)
                self._debug.log("press", {
                    "key": "a", "phase": "start",
                    "session_time": round(press_session, 3),
                })
                self._done("started aside capture")
            else:
                press_ts = time.monotonic()
                press_session = self._cursor_time_at(press_ts)
                self._force_flush.set()
                self.writer.append_marker(ASIDE_MARKER_TYPE, "end")
                self._aside_window.end_press_time = press_ts
                window = self._aside_window
                self._aside_window = None
                if self._r_window is not None:
                    self._r_window.resume(self.writer.cursor())
                self._debug.log("press", {
                    "key": "a", "phase": "end",
                    "session_time": round(press_session, 3),
                })
                threading.Thread(
                    target=self._finish_paste, args=(window,), daemon=True
                ).start()

    def _toggle_mute(self) -> None:
        if self._muted.is_set():
            self._muted.clear()
            self._debug.log("mute", {"state": "off"})
            self._done("unmuted — transcription resumed")
        else:
            self._muted.set()
            self._debug.log("mute", {"state": "on"})
            self._done("muted — transcription paused", style="bold yellow")

    def _cancel_window(self) -> None:
        with self._state_lock:
            if self._aside_window is not None:
                # Scrub the unfinished aside-start marker so the transcript
                # doesn't show a recording event for a cancelled capture.
                self.writer.remove_last_marker(ASIDE_MARKER_TYPE, "start")
                self._aside_window = None
                if self._r_window is not None:
                    self._r_window.resume(self.writer.cursor())
                self._done("aside cancelled (r resumed)" if self._r_window else "aside cancelled")
                return
            if self._r_window is not None:
                self.writer.remove_last_marker(RECORDING_MARKER_TYPE, "start")
                self._r_window = None
                self._done("r capture cancelled")
                return
            self._done("nothing to cancel", style="dim")

    def _finish_paste(self, window: _ClipboardWindow) -> None:
        # Wait for the transcriber to drain past the end-press timestamp.
        wait_started = time.monotonic()
        deadline = window.end_press_time + PASTE_DRAIN_TIMEOUT
        target = self._cursor_time_at(window.end_press_time)
        timed_out = True
        while time.monotonic() < deadline:
            if self.writer.latest_segment_end() >= target:
                timed_out = False
                break
            time.sleep(0.1)
        drain_wait_ms = int((time.monotonic() - wait_started) * 1000)
        # Materialize any pending markers (notably the recording-end token
        # for THIS press) at the current cursor. Otherwise, if the audio
        # right after the end-press is silent and the press_time is never
        # crossed by a segment, the pending end marker sits in the queue
        # and gets drained by the next recording's first segment — landing
        # in the middle of the new recording's content.
        drained = self.writer.flush_pending_markers()
        if drained:
            self._debug.log("marker", {
                "type": "recording", "action": "drained_pending",
                "key": "-", "count": drained,
            })
        # Close the live span at the current cursor and paste.
        window.close(self.writer.cursor())
        text = window.text(self.writer).strip()
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
        self._stream_server.start()
        self.transcriber.start()
        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop, name="Aggregator", daemon=True
        )
        self._aggregator_thread.start()
        self._drain_thread = threading.Thread(
            target=self._drain_loop, name="SegmentDrain", daemon=True
        )
        self._drain_thread.start()

        if self._enable_widget:
            try:
                self._widget = widget_module.StatusServer(self.get_status_snapshot)
                host, port = self._widget.start()
                url = f"http://{host}:{port}"
                self.console.print(f"[bold]Widget:[/bold] {url}")
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
            r_active = self._r_window is not None
            aside_active = self._aside_window is not None
            paste_log = list(self._paste_log)
            marker_open_since = self._marker_open_since
        open_marker = self.writer.open_marker_type()
        if r_active and aside_active:
            mode = "r+aside"
        elif r_active:
            mode = "r"
        elif aside_active:
            mode = "aside"
        else:
            mode = "passive"
        # If the marker state changed since last record (e.g. nothing is open
        # now), drop the stale "open since" string.
        if open_marker is None:
            marker_open_since = None
        return {
            "session_id": self.writer.session_id,
            "session_dir": self.writer.session_dir,
            "model": self.transcriber.model_label,
            "device": self._device,
            "compute": self._compute,
            "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
            "uptime_seconds": (datetime.now(tz=timezone.utc) - self._started_at).total_seconds(),
            "chunk_count": self.writer.chunk_count,
            "capture": {
                "mode": mode,
                "r_active": r_active,
                "aside_active": aside_active,
                "muted": self._muted.is_set(),
            },
            "open_marker": open_marker,
            "open_marker_since": marker_open_since,
            "paste_log": paste_log,
            "transcript_tail": self.writer.tail(2000),
            "now_seconds": round(
                time.monotonic() - (self.recorder.started_monotonic or time.monotonic()),
                3,
            ),
            "debug_events": self._debug.snapshot(limit=500),
        }

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
        # Give the aggregator a moment to drain remaining frames.
        time.sleep(0.5)
        self.transcriber.stop()
        # Drain any final segments.
        time.sleep(0.5)
        while True:
            try:
                seg: Segment = self.segment_q.get_nowait()
            except queue.Empty:
                break
            self.writer.feed_segment(seg.text, seg.end_time, words=seg.words)
        # Final chunk + meta.
        path = self.writer.force_flush()
        if path:
            self.console.print(f"[dim]💾 final chunk → {os.path.basename(path)}[/dim]")
        meta_path = self.writer.finalize()
        self.console.print(
            f"[bold green]✓ Session saved:[/bold green] {self.writer.session_dir}"
        )

    def _print_banner(self) -> None:
        marker_lines = "\n".join(
            f"  [bold]{m.key} (x2)[/bold] - open/close marker: {m.type}  [dim]({m.description})[/dim]"
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
            "  [bold]q (x2)[/bold] - quit + flush\n\n"
            f"[dim]Session: {self.writer.session_id}[/dim]\n"
            f"[dim]Model:   {self.transcriber.model_label} ({FW_DEVICE}, {FW_COMPUTE})[/dim]\n"
            f"[dim]Output:  {self.writer.session_dir}[/dim]"
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
                        help="Disable the HTTP status widget + browser auto-open.")
    args = parser.parse_args()

    config = load_config(args.config)
    app = PersistentApp(
        config=config,
        model=args.model,
        device=args.device,
        compute=args.compute,
        enable_widget=not args.no_widget,
    )
    app.run()


if __name__ == "__main__":
    main()
