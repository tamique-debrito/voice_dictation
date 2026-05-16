"""PasteExecutor — subscribes to paste.actions; performs the actual paste.

Live mode copies the text to the system clipboard via pyperclip and
synthesises Cmd+V (macOS/Linux) or Ctrl+V (Windows) via pynput.
``NullPasteExecutor`` (used in replay) just records what would have been
pasted without touching the OS.
"""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from typing import Optional

try:
    import pyperclip  # type: ignore
    _PYPERCLIP_OK = True
except Exception:
    pyperclip = None  # type: ignore
    _PYPERCLIP_OK = False

try:
    from pynput import keyboard  # type: ignore
    _PYNPUT_OK = True
except Exception:
    keyboard = None  # type: ignore
    _PYNPUT_OK = False

from .event_bus import EventBus
from .types import Event, TOPIC_PASTE_ACTIONS


logger = logging.getLogger(__name__)


_PASTE_MODIFIER = None
if _PYNPUT_OK:
    _PASTE_MODIFIER = (
        keyboard.Key.ctrl if platform.system() == "Windows" else keyboard.Key.cmd
    )


class PasteExecutor:
    def __init__(self, bus: EventBus, stop_event: threading.Event,
                 *, copy_paste_delay: float = 0.03) -> None:
        self._bus = bus
        self._stop = stop_event
        self._delay = copy_paste_delay
        self._kb: Optional["keyboard.Controller"] = None
        if _PYNPUT_OK:
            try:
                self._kb = keyboard.Controller()
            except Exception:
                logger.exception("PasteExecutor: keyboard.Controller() failed")
        # All output (pastes + stream_edits) runs on a single worker so
        # keystrokes from two close-in-time events never interleave. Two
        # concurrent ``keyboard.type()`` calls produce char-by-char
        # interleaving like "YTehsa,t 'yse..." for "Yes,..." + "That's..."
        # which is exactly what we saw during a live stream regression.
        self._q: "queue.Queue[Optional[Event]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop, name="paste-exec", daemon=True,
        )
        self._worker.start()
        self._sub = bus.subscribe(
            TOPIC_PASTE_ACTIONS, self._on_paste, name="paste_executor",
        )
        ready = _PYPERCLIP_OK and self._kb is not None
        logger.info(
            "PasteExecutor initialized (ready=%s pyperclip=%s pynput=%s)",
            ready, _PYPERCLIP_OK, _PYNPUT_OK,
        )

    def _on_paste(self, ev: Event) -> None:
        # Enqueue only; the worker decides what to do. The bus subscriber
        # loop must not block on clipboard / keyboard synthesis.
        self._q.put(ev)

    def _worker_loop(self) -> None:
        while True:
            ev = self._q.get()
            if ev is None:
                return
            try:
                self._handle(ev)
            except Exception:
                logger.exception("PasteExecutor: worker handler failed")

    def _handle(self, ev: Event) -> None:
        kind = ev.payload.get("kind", "paste")
        if kind == "stream_edit":
            backspaces = int(ev.payload.get("backspaces", 0) or 0)
            text = ev.payload.get("text", "") or ""
            final_text = ev.payload.get("final_text")
            if backspaces == 0 and not text and final_text is None:
                return
            if self._kb is None:
                logger.warning(
                    "PasteExecutor: pynput missing, would stream_edit "
                    "(bs=%d text=%r)", backspaces, text[:80],
                )
                return
            self._stream_emit_now(backspaces, text)
            # When the manager signals a stream close it includes the
            # whole concatenated text as ``final_text`` so we can sync
            # the system clipboard, matching the behavior of a regular
            # paste (which copies-then-Cmd+V's). Without this, a
            # live-streamed dictation leaves nothing on the clipboard.
            if final_text is not None and _PYPERCLIP_OK:
                try:
                    pyperclip.copy(final_text)
                    logger.info(
                        "PasteExecutor: clipboard set to live-stream "
                        "final_text (%d chars)", len(final_text),
                    )
                except Exception:
                    logger.exception("PasteExecutor: clipboard copy failed")
            return
        text = ev.payload.get("text", "") or ""
        if not text:
            logger.info("PasteExecutor: empty text, skipping")
            return
        if not _PYPERCLIP_OK or self._kb is None:
            logger.warning(
                "PasteExecutor: deps missing, would paste: %r", text[:80],
            )
            return
        self._paste_now(text)

    def _paste_now(self, text: str) -> None:
        try:
            pyperclip.copy(text)
            time.sleep(self._delay)
            with self._kb.pressed(_PASTE_MODIFIER):
                self._kb.tap("v")
            logger.info("PasteExecutor pasted %d chars", len(text))
        except Exception:
            logger.exception("PasteExecutor paste failed; text is on clipboard")

    def _stream_emit_now(self, backspaces: int, text: str) -> None:
        # Known-but-implausible reentrancy: synthesised keystrokes are
        # seen by UserActionManager's global keyboard listener. The
        # backspaces here can't match any action key (they're Key.backspace,
        # not a char), and ``keyboard.type(text)`` synthesises individual
        # presses spread across milliseconds — for a fast double-tap of an
        # action key to fall out of normal English transcription, the
        # transcribed text would need to contain two consecutive copies of
        # the action key within BareMultiTap.window (default 1.0s). For the
        # default keys (r/a/x/q/m/marker keys), this is vanishingly rare in
        # practice; flag here so a future bug report has a thread to pull.
        try:
            for _ in range(backspaces):
                self._kb.tap(keyboard.Key.backspace)
            if text:
                self._kb.type(text)
            logger.info(
                "PasteExecutor stream_edit (bs=%d +%d chars)",
                backspaces, len(text),
            )
        except Exception:
            logger.exception("PasteExecutor stream_edit failed")

    def shutdown(self) -> None:
        self._sub.shutdown()
        self._q.put(None)
        self._worker.join(timeout=2.0)


class NullPasteExecutor:
    """Replay-mode executor — records but does not paste."""

    def __init__(self, bus: EventBus) -> None:
        self.pastes: list[dict] = []
        self._sub = bus.subscribe(
            TOPIC_PASTE_ACTIONS, self._on_paste, name="null_paste",
        )

    def _on_paste(self, ev: Event) -> None:
        self.pastes.append({
            "emit_accepted_ms": ev.emit_accepted_ms,
            "seq": ev.seq,
            **ev.payload,
        })

    def shutdown(self) -> None:
        self._sub.shutdown()
