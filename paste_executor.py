"""PasteExecutor: consumes paste output events emitted by SessionState
and performs the actual OS paste.

This module owns the side-effecting tail of the paste pipeline:

1. Drain barrier — wait for the fast transcription stream's watermark
   to reach the press's session time, so the words spoken right up to
   the click moment have a chance to make it into the canonical render
   before we slice for paste.
2. Pending-marker flush — re-anchor any markers still queued past the
   watermark to the current cursor so the canonical text contains the
   recording-end token before the slice is computed.
3. Slice + emit paste event — call ``window.text(timeline)`` against the
   now-flushed canonical render and emit a ``paste`` event into
   ``SessionState`` carrying the resolved text, timing, and span data
   (the ``paste`` event drives the paste_log mirror).
4. OS paste — copy the text into the system clipboard and synthesize a
   ``Cmd+V`` to drop it into whichever app currently has focus.

Live mode wires a ``PasteExecutor`` into ``SessionState`` as the
paste callback. Replay mode leaves the callback unset — none of the
above happens, but the resulting recording is still complete because
the paste event isn't where state lives (the user_action stream is).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional

from clipboard_manager import ClipboardManager
from clipboard_window import ClipboardWindow


# Maximum time to wait for the transcriber to drain past a clipboard
# window's end-press before we paste whatever has rendered so far. Long
# enough that fast-stream lag during transcribe-time bursts doesn't
# clip the tail; short enough that the user doesn't feel a stall.
DEFAULT_DRAIN_TIMEOUT = 6.0


class PasteExecutor:
    def __init__(
        self,
        *,
        timeline,
        clipboard: ClipboardManager,
        log_event: Callable[[str, dict], None],
        cursor_time_at: Callable[[float], float],
        # Optional UI hooks — let the host print "pasted X" / "nothing to
        # paste" lines without coupling this module to a console.
        on_done: Optional[Callable[[str, str], None]] = None,
        drain_timeout: float = DEFAULT_DRAIN_TIMEOUT,
    ):
        self._timeline = timeline
        self._clipboard = clipboard
        self._log_event = log_event
        self._cursor_time_at = cursor_time_at
        self._on_done = on_done
        self._drain_timeout = drain_timeout

    # ------------------------------------------------------------------
    # Public: SessionState's paste_callback signature
    # ------------------------------------------------------------------

    def execute(self, window: ClipboardWindow) -> None:
        """Spawn a daemon thread that drains, slices, emits the paste
        event, and performs the OS paste. Returns immediately so the
        caller (SessionState.ingest) doesn't block on transcription
        latency."""
        threading.Thread(
            target=self._run,
            args=(window,),
            name="PasteExecutor",
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self, window: ClipboardWindow) -> None:
        # Wait for the fast watermark to cross the end-press time so
        # words spoken right up to the click are inside the canonical
        # render before we slice.
        wait_started = time.monotonic()
        deadline = window.end_press_time + self._drain_timeout
        target = self._cursor_time_at(window.end_press_time)
        timed_out = True
        while time.monotonic() < deadline:
            if self._timeline.latest_segment_end() >= target:
                timed_out = False
                break
            time.sleep(0.1)
        drain_wait_ms = int((time.monotonic() - wait_started) * 1000)

        # Re-anchor any pending markers (notably the recording-end
        # token for THIS press) to the current fast watermark so they
        # appear in the canonical render before we slice.
        try:
            drained = self._timeline.flush_pending_markers()
        except Exception:
            drained = 0
        if drained:
            try:
                self._log_event("marker", {
                    "type": "recording", "action": "drained_pending",
                    "key": "-", "count": drained,
                })
            except Exception:
                pass

        # Close the live span at the current cursor and compute the text.
        window.close(self._timeline.cursor())
        text = window.text(self._timeline).strip()

        # Emit the paste event. SessionState._apply reads this to update
        # the paste_log mirror; the event is also recorded into
        # stream_user.jsonl for the dashboard's debug events panel.
        try:
            self._log_event("paste", {
                "label": window.label,
                "start_press_time": round(window.start_press_time, 3),
                "end_press_time": round(
                    self._cursor_time_at(window.end_press_time), 3
                ),
                "spans": [list(s) for s in window.spans],
                "drain_wait_ms": drain_wait_ms,
                "timed_out": timed_out,
                "char_count": len(text),
                "preview": text[:200],
                "ts_wall": datetime.now().strftime("%H:%M:%S"),
                "full": text,
            })
        except Exception:
            pass

        if not text:
            if self._on_done is not None:
                try:
                    self._on_done(
                        f"{window.label} → nothing to paste",
                        "yellow",
                    )
                except Exception:
                    pass
            return

        # OS-level paste: clipboard.copy_and_paste handles save/restore
        # of the user's existing clipboard so dictation doesn't clobber
        # whatever they had.
        try:
            self._clipboard.copy_and_paste(text)
        except Exception:
            pass

        if self._on_done is not None:
            preview = text[:80] + ("..." if len(text) > 80 else "")
            try:
                self._on_done(
                    f"{window.label} pasted → [green]{preview}[/green]",
                    "bold green",
                )
            except Exception:
                pass
