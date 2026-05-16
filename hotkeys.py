"""BareDoubleTap + key_char — ported from voice_dictation/hotkeys.py.

Kept self-contained in v2 to avoid importing voice_dictation/ (the v1 module
imports its config singleton at module-load time).
"""

from __future__ import annotations

import time
from typing import Callable, Optional


def key_char(key) -> Optional[str]:
    """Decode a pynput key event into a lowercase character (or None)."""
    try:
        c = key.char
        if not c:
            return None
        if len(c) == 1 and ord(c) < 32:
            return chr(ord(c) + 96)
        return c.lower()
    except AttributeError:
        return None


class BareDoubleTap:
    """Per-key double-tap detector with cross-key reset and typing-guard.

    A double-tap is registered when the same tracked key is pressed twice
    within ``window`` seconds, with no other tracked key in between AND no
    untracked typing within ``quiet_before`` of the first tap.
    """

    def __init__(
        self,
        window: float,
        keys: set[str],
        on_double_tap: Callable[[str], None],
        quiet_before: float = 1.0,
    ) -> None:
        self.window = window
        self.keys = set(keys)
        self.on_double_tap = on_double_tap
        self.quiet_before = quiet_before
        self._last_press: dict[str, float] = {}
        self._last_untracked_time: float = 0.0

    def update_keys(self, keys: set[str]) -> None:
        self.keys = set(keys)
        self._last_press = {}

    def feed(self, char: Optional[str]) -> None:
        if char is None:
            return
        now = time.time()
        if char in self.keys:
            last = self._last_press.get(char, 0.0)
            if now - last <= self.window:
                self._last_press = {char: 0.0}
                if last - self._last_untracked_time >= self.quiet_before:
                    self.on_double_tap(char)
            else:
                self._last_press[char] = now
                for other in list(self._last_press):
                    if other != char:
                        self._last_press[other] = 0.0
        else:
            self._last_untracked_time = now
            if self._last_press:
                self._last_press.clear()
