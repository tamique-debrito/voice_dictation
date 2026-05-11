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
from datetime import datetime
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
    RECORDING_MARKER_TYPE,
    SAMPLE_RATE,
    SILENCE_MS,
)
from hotkeys import BareDoubleTap, key_char, load_config
from persistent_recorder import PersistentRecorder
from session_writer import MarkerType, SessionWriter
from silence_detector import SilenceDetector
from streaming_transcriber import AudioWindow, Segment, StreamingTranscriber


# How long an audio window can grow before we force-emit it for transcription
# even without a silence boundary. Keeps latency bounded for monologues.
MAX_WINDOW_SECONDS = 25.0
# Maximum time to wait for the transcriber to catch up with a clipboard-window
# end-press before pasting whatever we have.
PASTE_DRAIN_TIMEOUT = 6.0


class _ClipboardWindow:
    """Tracks an active r/aside paste window across aside park/resume."""

    def __init__(self, label: str, start_cursor: int, end_press_time: float):
        self.label = label
        # spans: list of (start, end). The last span has end=None while live.
        self.spans: list[tuple[int, Optional[int]]] = [(start_cursor, None)]
        self.end_press_time = end_press_time

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
        for s, e in self.spans:
            parts.append(writer.slice_for_paste(s, e))
        return " ".join(p for p in parts if p)


class PersistentApp:
    def __init__(self, config: dict, model: str, device: str, compute: str):
        self.console = Console()
        self.config = config
        self._stop = threading.Event()

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

        self._aggregator_thread: Optional[threading.Thread] = None
        self._drain_thread: Optional[threading.Thread] = None

        # Clipboard-window state (touched only from listener thread).
        self._r_window: Optional[_ClipboardWindow] = None
        self._aside_window: Optional[_ClipboardWindow] = None
        self._state_lock = threading.Lock()

        # Bare double-tap dispatcher
        keys = set(self._marker_keys) | {"r", "a", "x", "q"}
        self._dt = BareDoubleTap(window=1.0, keys=keys, on_double_tap=self._on_action)

        # Used to undo the two visible keypresses in whichever app currently
        # has keyboard focus when a double-tap is recognized.
        self._kb = keyboard.Controller()

    # ------------------------------------------------------------------
    # Aggregator: PCM -> windows
    # ------------------------------------------------------------------

    def _aggregator_loop(self) -> None:
        vad = SilenceDetector(silence_ms=SILENCE_MS, aggressiveness=2)
        window_pcm = bytearray()
        window_start: Optional[float] = None
        window_end: float = 0.0
        last_chunk_check = 0.0

        def flush_window(at_silence: bool) -> None:
            nonlocal window_pcm, window_start, window_end
            if not window_pcm or window_start is None:
                return
            try:
                self.window_q.put(
                    AudioWindow(
                        pcm=bytes(window_pcm),
                        start_time=window_start,
                        end_time=window_end,
                    ),
                    timeout=2.0,
                )
            except queue.Full:
                # Transcriber is way behind. Drop this window.
                pass
            window_pcm = bytearray()
            window_start = None
            window_end = 0.0

        while not self._stop.is_set():
            try:
                ts, data = self.raw_q.get(timeout=0.5)
            except queue.Empty:
                # No audio for a moment — try a chunk flush opportunistically.
                self._maybe_flush_chunk(at_silence_boundary=False)
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
            if saw_boundary or window_seconds >= MAX_WINDOW_SECONDS:
                flush_window(at_silence=saw_boundary)
                if saw_boundary:
                    # Drain any pending chunk if size threshold is met.
                    self._maybe_flush_chunk(at_silence_boundary=True)

            # Periodic background chunk-flush check (covers MAX_WINDOW splits).
            now = time.monotonic()
            if now - last_chunk_check > 5.0:
                last_chunk_check = now
                self._maybe_flush_chunk(at_silence_boundary=False)

        flush_window(at_silence=True)

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
            self.writer.feed_segment(seg.text, seg.end_time)

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
        elif char == "q":
            self._ack("quit (q)")
            self._stop.set()
            self._done("shutting down", style="bold yellow")

    def _handle_marker(self, key: str) -> None:
        events = self.writer.insert_marker(key)
        if not events:
            return
        for action, type_name in events:
            verb = "opened" if action == "open" else "closed"
            note = ""
            if len(events) > 1 and action == "close":
                # Cross-type switch — make the auto-close obvious.
                note = " [dim](auto, switching type)[/dim]"
            self._done(f"{verb} marker [magenta]{type_name}[/magenta]{note}")

    def _toggle_r(self) -> None:
        with self._state_lock:
            if self._r_window is None:
                if self._aside_window is not None:
                    self._done(
                        "[yellow]aside is active; close it with `a` first[/yellow]",
                        style="yellow",
                    )
                    return
                # Emit recording-start marker into the transcript.
                self.writer.append_marker(RECORDING_MARKER_TYPE, "start")
                cursor = self.writer.cursor()
                self._r_window = _ClipboardWindow("r", cursor, time.monotonic())
                self._done("started r capture")
            else:
                # End r: emit recording-end marker, drain, then paste.
                self.writer.append_marker(RECORDING_MARKER_TYPE, "end")
                self._r_window.end_press_time = time.monotonic()
                window = self._r_window
                self._r_window = None
                threading.Thread(
                    target=self._finish_paste, args=(window,), daemon=True
                ).start()

    def _toggle_aside(self) -> None:
        with self._state_lock:
            if self._aside_window is None:
                self.writer.append_marker(ASIDE_MARKER_TYPE, "start")
                cursor = self.writer.cursor()
                if self._r_window is not None:
                    self._r_window.park(cursor)
                self._aside_window = _ClipboardWindow("aside", cursor, time.monotonic())
                self._done("started aside capture")
            else:
                self.writer.append_marker(ASIDE_MARKER_TYPE, "end")
                self._aside_window.end_press_time = time.monotonic()
                window = self._aside_window
                self._aside_window = None
                if self._r_window is not None:
                    self._r_window.resume(self.writer.cursor())
                threading.Thread(
                    target=self._finish_paste, args=(window,), daemon=True
                ).start()

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
        deadline = window.end_press_time + PASTE_DRAIN_TIMEOUT
        target = self._cursor_time_at(window.end_press_time)
        while time.monotonic() < deadline:
            if self.writer.latest_segment_end() >= target:
                break
            time.sleep(0.1)
        # Close the live span at the current cursor and paste.
        window.close(self.writer.cursor())
        text = window.text(self.writer).strip()
        if not text:
            self._done(f"{window.label} → nothing to paste", style="yellow")
            return
        self.clipboard.copy_and_paste(text)
        preview = text[:80] + ("..." if len(text) > 80 else "")
        self._done(f"{window.label} pasted → [green]{preview}[/green]")

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
        self.transcriber.start()
        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop, name="Aggregator", daemon=True
        )
        self._aggregator_thread.start()
        self._drain_thread = threading.Thread(
            target=self._drain_loop, name="SegmentDrain", daemon=True
        )
        self._drain_thread.start()

        listener = keyboard.Listener(on_press=self._on_press)
        listener.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._stop.set()
        listener.stop()
        self._shutdown()

    def _on_press(self, key) -> None:
        try:
            self._dt.feed(key_char(key))
        except Exception:
            pass

    def _shutdown(self) -> None:
        self.console.print("\n[bold yellow]Shutting down — flushing...[/bold yellow]")
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
            self.writer.feed_segment(seg.text, seg.end_time)
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
    args = parser.parse_args()

    config = load_config(args.config)
    app = PersistentApp(
        config=config,
        model=args.model,
        device=args.device,
        compute=args.compute,
    )
    app.run()


if __name__ == "__main__":
    main()
