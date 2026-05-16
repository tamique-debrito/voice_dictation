"""BareDoubleTap + BareMultiTap + key_char — ported from voice_dictation/hotkeys.py.

Kept self-contained in v2 to avoid importing voice_dictation/ (the v1 module
imports its config singleton at module-load time).
"""

from __future__ import annotations

import threading
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


class BareMultiTap:
    """Double- and triple-tap detector with disambiguation delay.

    Press accounting mirrors ``BareDoubleTap``. To allow a 3rd press to
    upgrade a double-tap into a triple-tap, the double-tap action callback
    is deferred by ``disambiguation_delay`` seconds via a Timer. The 3rd
    press, if it arrives in that window, cancels the timer and fires the
    triple-tap callback immediately.

    ``on_visual_cleanup(char, n_visible)`` runs *synchronously* on the
    listener thread the instant we detect an n-tap so the caller can
    backspace-out the visible chars before any deferred action fires.
    This prevents the user from seeing the typed key sit on screen for
    the disambiguation delay.

    Cross-key reset and typing-guard (``quiet_before``) match BareDoubleTap.
    """

    def __init__(
        self,
        window: float,
        keys: set[str],
        on_double_tap: Callable[[str], None],
        on_triple_tap: Callable[[str], None],
        on_visual_cleanup: Optional[Callable[[str, int], None]] = None,
        disambiguation_delay: float = 0.30,
        quiet_before: float = 1.0,
    ) -> None:
        self.window = window
        self.keys = set(keys)
        self.on_double_tap = on_double_tap
        self.on_triple_tap = on_triple_tap
        self.on_visual_cleanup = on_visual_cleanup
        self.disambiguation_delay = disambiguation_delay
        self.quiet_before = quiet_before
        # Per-key press history (timestamps within current sequence).
        self._presses: dict[str, list[float]] = {}
        self._last_untracked_time: float = 0.0
        # Deferred double-tap timer for the most recent 2-tap, keyed by char.
        self._pending_double: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def update_keys(self, keys: set[str]) -> None:
        with self._lock:
            self.keys = set(keys)
            self._presses = {}
            for t in self._pending_double.values():
                t.cancel()
            self._pending_double.clear()

    def _fire_double(self, char: str) -> None:
        with self._lock:
            self._pending_double.pop(char, None)
            # Sequence consumed.
            self._presses.pop(char, None)
        self.on_double_tap(char)

    def feed(self, char: Optional[str]) -> None:
        if char is None:
            return
        now = time.time()
        if char not in self.keys:
            self._last_untracked_time = now
            with self._lock:
                self._presses.clear()
                for t in self._pending_double.values():
                    t.cancel()
                self._pending_double.clear()
            return

        with self._lock:
            # Reset other-key sequences (cross-key reset).
            for k in list(self._presses):
                if k != char:
                    self._presses.pop(k, None)
            history = self._presses.get(char, [])
            # Drop presses outside the rolling window.
            history = [t for t in history if now - t <= self.window]
            history.append(now)
            self._presses[char] = history
            n = len(history)
            # Typing guard: only fire if quiet before the *first* press.
            quiet_ok = (history[0] - self._last_untracked_time
                        >= self.quiet_before)
            timer_to_start: Optional[threading.Timer] = None
            fire_triple = False
            if n == 2 and quiet_ok:
                if self.on_visual_cleanup is not None:
                    self.on_visual_cleanup(char, 2)
                # Defer double-tap to allow a 3rd press to upgrade.
                t = threading.Timer(
                    self.disambiguation_delay, self._fire_double, args=(char,),
                )
                t.daemon = True
                self._pending_double[char] = t
                timer_to_start = t
            elif n >= 3 and quiet_ok:
                # Cancel deferred double-tap; fire triple-tap.
                pending = self._pending_double.pop(char, None)
                if pending is not None:
                    pending.cancel()
                self._presses.pop(char, None)
                if self.on_visual_cleanup is not None:
                    # Only one extra char to clean — taps 1 and 2 were
                    # already cleaned when we detected the double.
                    self.on_visual_cleanup(char, 1)
                fire_triple = True
        if timer_to_start is not None:
            timer_to_start.start()
        if fire_triple:
            self.on_triple_tap(char)
