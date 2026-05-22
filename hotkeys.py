"""BareDoubleTap + key_char — ported from voice_dictation/hotkeys.py.

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
    """Per-key double-tap detector with typing-guard and post-detect settle window.

    Detection flow
    ==============
    1. **quiet_before guard**: a double-tap is only recognised if the first tap
       occurred at least ``quiet_before`` seconds after the last *untracked*
       keystroke.  This prevents mid-word repeated letters from firing a marker.

    2. **on_candidate** (optional): fires *immediately* on the second keypress,
       before the settle window starts.  Use this for instant side-effects such
       as backspace-undoing the two visible keystrokes, or capturing a real-time
       timestamp, without waiting for the settle delay.

    3. **settle_after window**: after a candidate is detected the detector waits
       ``settle_after`` seconds before calling ``on_double_tap``.  During this
       window:
         - additional presses of the **same** tracked key are silently absorbed
           (handles accidental triple-taps / stutter on the second press).
         - any press of a **different** character cancels the pending fire and
           resets the detector.
       Set ``settle_after=0`` to restore the original immediate-fire behaviour.
    """

    def __init__(
        self,
        window: float,
        keys: set[str],
        on_double_tap: Callable[[str], None],
        quiet_before: float = 1.0,
        settle_after: float = 0.5,
        on_candidate: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.window = window
        self.keys = set(keys)
        self.on_double_tap = on_double_tap
        self.quiet_before = quiet_before
        self.settle_after = settle_after
        self.on_candidate = on_candidate
        self._last_press: dict[str, float] = {}
        self._last_untracked_time: float = 0.0
        # Settle state — access serialised by _lock.
        self._lock = threading.Lock()
        self._settle_char: Optional[str] = None
        self._settle_timer: Optional[threading.Timer] = None

    def update_keys(self, keys: set[str]) -> None:
        self.keys = set(keys)
        self._last_press = {}
        self._cancel_settle()

    # ------------------------------------------------------------------
    # Settle helpers
    # ------------------------------------------------------------------

    def _cancel_settle(self) -> None:
        """Cancel any pending settle timer. Thread-safe."""
        with self._lock:
            timer, self._settle_timer = self._settle_timer, None
            self._settle_char = None
        if timer is not None:
            timer.cancel()

    def _arm_settle(self, char: str) -> None:
        """Fire on_candidate immediately, then arm the settle timer."""
        # on_candidate runs outside the lock — it may call pynput/backspace.
        if self.on_candidate is not None:
            try:
                self.on_candidate(char)
            except Exception:
                pass

        with self._lock:
            # Defensively cancel any stale timer (shouldn't exist here).
            old_timer, self._settle_timer = self._settle_timer, None
            self._settle_char = char
            if self.settle_after > 0:
                timer = threading.Timer(
                    self.settle_after, self._on_settle, args=(char,),
                )
                self._settle_timer = timer
            else:
                timer = None

        if old_timer is not None:
            old_timer.cancel()

        if timer is not None:
            timer.start()
        else:
            # settle_after == 0: fire synchronously (original behaviour).
            self.on_double_tap(char)

    def _on_settle(self, char: str) -> None:
        """Timer callback — calls on_double_tap unless cancelled."""
        with self._lock:
            if self._settle_char != char:
                return  # cancelled or superseded between arm and fire
            self._settle_char = None
            self._settle_timer = None
        self.on_double_tap(char)

    # ------------------------------------------------------------------
    # Public feed
    # ------------------------------------------------------------------

    def feed(self, char: Optional[str]) -> None:
        if char is None:
            return
        now = time.time()

        # ── Settle-window intercept ──────────────────────────────────────
        with self._lock:
            settle = self._settle_char

        if settle is not None:
            if char == settle:
                # Same tracked key: absorb. The timer continues; on_double_tap
                # will still fire when the settle expires.
                return
            else:
                # Different character: cancel the pending double-tap and treat
                # this keystroke as an untracked character so the next detection
                # cycle starts clean.
                self._cancel_settle()
                self._last_untracked_time = now
                self._last_press.clear()
                return

        # ── Normal double-tap detection ──────────────────────────────────
        if char in self.keys:
            last = self._last_press.get(char, 0.0)
            if now - last <= self.window:
                self._last_press = {char: 0.0}
                if last - self._last_untracked_time >= self.quiet_before:
                    self._arm_settle(char)
            else:
                self._last_press[char] = now
                for other in list(self._last_press):
                    if other != char:
                        self._last_press[other] = 0.0
        else:
            self._last_untracked_time = now
            if self._last_press:
                self._last_press.clear()
