"""PasteExecutor — subscribes to paste.actions; performs the actual paste.

Live mode copies the text to the system clipboard via pyperclip and
synthesises Cmd+V (macOS/Linux) or Ctrl+V (Windows) via pynput.
``NullPasteExecutor`` (used in replay) just records what would have been
pasted without touching the OS.
"""

from __future__ import annotations

import logging
import platform
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
        self._sub = bus.subscribe(
            TOPIC_PASTE_ACTIONS, self._on_paste, name="paste_executor",
        )
        ready = _PYPERCLIP_OK and self._kb is not None
        logger.info(
            "PasteExecutor initialized (ready=%s pyperclip=%s pynput=%s)",
            ready, _PYPERCLIP_OK, _PYNPUT_OK,
        )

    def _on_paste(self, ev: Event) -> None:
        text = ev.payload.get("text", "") or ""
        if not text:
            logger.info("PasteExecutor: empty text, skipping")
            return
        if not _PYPERCLIP_OK or self._kb is None:
            logger.warning(
                "PasteExecutor: deps missing, would paste: %r", text[:80],
            )
            return
        # Spawn a thread so the bus subscriber loop isn't blocked by the
        # clipboard / keyboard synthesis. Pastes are independent.
        threading.Thread(
            target=self._paste_now, args=(text,), name="paste-exec", daemon=True,
        ).start()

    def _paste_now(self, text: str) -> None:
        try:
            pyperclip.copy(text)
            time.sleep(self._delay)
            with self._kb.pressed(_PASTE_MODIFIER):
                self._kb.tap("v")
            logger.info("PasteExecutor pasted %d chars", len(text))
        except Exception:
            logger.exception("PasteExecutor paste failed; text is on clipboard")

    def shutdown(self) -> None:
        self._sub.shutdown()


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
