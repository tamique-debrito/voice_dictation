"""Shared hotkey utilities used by both the regular and persistent apps.

Houses:
    * Modifier-key tracking helpers (originally inlined in voice_dictation.py)
    * Config loading
    * BareDoubleTap: per-key double-press detection without any modifiers held.
      Resets state when an interleaving (different) key is pressed inside the
      window, so accidental "rr" inside ordinary typing is unlikely to fire.
"""

from __future__ import annotations

import os
import platform as _platform
import time
from typing import Callable, Optional

from pynput import keyboard


# ---------------------------------------------------------------------------
# Modifier-aware hotkey support (mirrors the original voice_dictation.py)
# ---------------------------------------------------------------------------

if _platform.system() == "Windows":
    DEFAULT_MODIFIERS = ["cmd", "shift"]  # Win+Shift
else:  # macOS
    DEFAULT_MODIFIERS = ["ctrl", "cmd"]  # Ctrl+Cmd

DEFAULT_CONFIG = {
    "modifiers": DEFAULT_MODIFIERS,
    "keys": {
        "toggle_recording": "r",
        "discard_recording": "x",
        "toggle_aside": "a",
    },
    "no_modifiers": False,
}

_MODIFIER_MAP = {
    "ctrl":  {keyboard.Key.ctrl, keyboard.Key.ctrl_r, keyboard.Key.ctrl_l},
    "cmd":   {keyboard.Key.cmd, keyboard.Key.cmd_r},
    "alt":   {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_r},
}


def load_config(path: Optional[str], here: Optional[str] = None) -> dict:
    """Load configuration from JSON file, falling back to local_config.json.

    Returns the legacy shape ``{modifiers, keys, no_modifiers, persistent}``
    so existing callers keep working. The data is pulled from the unified
    ``runtime_config`` singleton (which itself reads from
    ``local_config.json`` with env-var overrides). The ``path`` arg, when
    given, is used to reload the unified config from a custom path —
    otherwise the already-loaded singleton is returned.

    The ``persistent.markers`` key remains supported as a place to put
    user-configured marker types.
    """
    from runtime_config import CFG, load_runtime_config

    if path is not None:
        cfg = load_runtime_config(path=path, module_dir=here)
    else:
        cfg = CFG
    return {
        "modifiers": list(cfg.hotkeys.modifiers),
        "keys": dict(cfg.hotkeys.keys),
        "no_modifiers": cfg.hotkeys.no_modifiers,
        "persistent": {"markers": list(cfg.markers)},
    }


def key_char(key) -> Optional[str]:
    """Decode a pynput key event into a lowercase character (or None).

    Handles control characters (e.g. Ctrl+R yields \x12) by mapping back to
    the plain letter.
    """
    try:
        c = key.char
        if not c:
            return None
        if len(c) == 1 and ord(c) < 32:
            return chr(ord(c) + 96)
        return c.lower()
    except AttributeError:
        return None


class ModifierTracker:
    """Tracks which named modifier groups are currently pressed."""

    def __init__(self, required: list[str]):
        self.required = list(required)
        self._pressed: set[str] = set()

    def update(self, key, pressed: bool) -> bool:
        """Update tracking for ``key``. Returns True if the key was a modifier."""
        for name, keys in _MODIFIER_MAP.items():
            if key in keys:
                if pressed:
                    self._pressed.add(name)
                else:
                    self._pressed.discard(name)
                return True
        return False

    def all_required_active(self) -> bool:
        return all(m in self._pressed for m in self.required)


# ---------------------------------------------------------------------------
# Bare double-tap (no modifiers required)
# ---------------------------------------------------------------------------

class BareDoubleTap:
    """Per-key double-tap detector with cross-key reset.

    Maintains a per-key "last press time" map. A double-tap is registered when
    the same key is pressed twice within ``window`` seconds, with no other
    relevant key pressed in between.

    ``quiet_before`` adds a typing-guard: if any untracked key was pressed
    within that many seconds before the first tap, the double-tap is suppressed.
    This prevents accidental triggers when the hotkey key appears at the end of
    normal typing (e.g. "…text r" then pressing r again quickly).

    Usage::

        tap = BareDoubleTap(window=1.0, keys={"r", "1", "q"},
                            on_double_tap=lambda k: ...)
        tap.feed(char)  # call from pynput on_press

    Only keys present in ``keys`` are tracked; unrelated keystrokes (typing
    prose) reset the tracking state for any pending double-tap, so accidental
    "rr" inside text is unlikely to fire.
    """

    def __init__(
        self,
        window: float,
        keys: set[str],
        on_double_tap: Callable[[str], None],
        quiet_before: float = 1.0,
    ):
        self.window = window
        self.keys = set(keys)
        self.on_double_tap = on_double_tap
        self.quiet_before = quiet_before
        self._last_press: dict[str, float] = {}
        self._last_untracked_time: float = 0.0

    def update_keys(self, keys: set[str]) -> None:
        """Replace the watched key set (used when marker config changes)."""
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
                # Suppress if untracked typing happened within quiet_before of
                # the first tap — guards against "...word r" + r firing.
                if last - self._last_untracked_time >= self.quiet_before:
                    self.on_double_tap(char)
            else:
                self._last_press[char] = now
                # Pressing a tracked key resets pending state on others.
                for other in list(self._last_press):
                    if other != char:
                        self._last_press[other] = 0.0
        else:
            # An untracked key inside an ordinary typing burst resets
            # all pending double-tap state.
            self._last_untracked_time = now
            if self._last_press:
                self._last_press.clear()
