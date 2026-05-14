"""UserActionProducer: turns raw keypresses into the third input stream.

This module owns the entire keypress-handling pipeline that used to live
on ``PersistentApp``:

* The ``pynput.keyboard.Listener`` that watches keystrokes.
* The ``BareDoubleTap`` filter that promotes a double-tap into a
  committed action (and rejects single presses + presses adjacent to
  normal typing).
* The backspace-out side effect that erases the two visible keystrokes
  the user already typed before the double-tap was detected.
* The mapping from a committed key to its canonical action name
  (``marker_press``, ``clipboard_toggle``, ``discard``, ``mute_toggle``,
  ``debug_flag``, ``quit``).

Its single output is a ``user_action`` event delivered to a callback the
host supplies. The event payload is ``{action, key, session_time}``
where ``session_time`` is the monotonic press timestamp converted to
session-relative seconds. ``action`` is the canonical name; the key
itself stays in ``key`` so configurable hotkeys (marker keys, clipboard
keys) survive in the recording.

There is no ``phase`` on the event — open vs close for a clipboard
window, and the marker's open/close decision, are determined by the
state manager downstream based on its current state. One event per
intent. This is the third input stream of the event-sourced replay
model (fast segments, hq segments, user_action).

OS coupling: this module is the only place that synthesizes keyboard
events (backspace-out). It is NOT involved in pasting — that lives in
``PasteExecutor`` so the manager → output pipeline stays clean.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Set

from pynput import keyboard

from hotkeys import BareDoubleTap, key_char


# Callback signature: ``(action: str, key: str, session_time: float) -> None``.
# ``session_time`` is monotonic seconds relative to the recorder's start.
OnUserAction = Callable[[str, str, float], None]


class UserActionProducer:
    def __init__(
        self,
        *,
        marker_keys: Set[str],
        clipboard_keys: Set[str],
        debug_flag_key: str,
        cursor_time_at: Callable[[float], float],
        on_user_action: OnUserAction,
        # Optional ack hook — lets the host print a "› recognized" line
        # synchronously with the user's action, before the downstream
        # state machine has a chance to run. Stays out of the event
        # stream (UI-only).
        on_ack: Optional[Callable[[str], None]] = None,
        # Optional stop signal — when set, the listener thread exits.
        # ``q`` (quit) flips this from inside _commit if present.
        stop_event: Optional[threading.Event] = None,
        # Double-tap detection window in seconds.
        double_tap_window: float = 1.0,
    ):
        self._marker_keys = set(marker_keys)
        self._clipboard_keys = set(clipboard_keys)
        self._debug_flag_key = debug_flag_key
        self._cursor_time_at = cursor_time_at
        self._on_user_action = on_user_action
        self._on_ack = on_ack
        self._stop = stop_event

        # The full set of keys the double-tap filter watches. Anything
        # not in this set passes straight through to the user's app.
        action_keys = (
            self._marker_keys
            | self._clipboard_keys
            | {"x", "q", "m", self._debug_flag_key}
        )
        self._dt = BareDoubleTap(
            window=double_tap_window,
            keys=action_keys,
            on_double_tap=self._commit,
        )

        # Keyboard controller for backspace-out only. Lives here, not in
        # PasteExecutor, because backspace-out is part of the input
        # filtering cleanup ("the user double-tapped, undo the two
        # visible keystrokes that already landed"). Each module that
        # touches the keyboard gets its own Controller instance —
        # they're cheap and avoid coupling.
        self._kb = keyboard.Controller()

        self._listener: Optional[keyboard.Listener] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    # ------------------------------------------------------------------
    # Internal: raw key → double-tap filter → committed action
    # ------------------------------------------------------------------

    def _on_press(self, key) -> None:
        try:
            char = key_char(key)
        except Exception:
            return
        if char:
            self._dt.feed(char)

    def _commit(self, char: str) -> None:
        """Called by BareDoubleTap when a key survives the filter."""
        # First: erase the two visible keystrokes the user already typed
        # before the double-tap was detected. Best-effort — in non-text
        # contexts the backspaces are no-ops.
        self._undo_keypresses(2)

        action = self._resolve_action(char)
        if action is None:
            return

        # ``q`` (quit) is special: it does have a state effect (signal
        # shutdown), but only if the host wired a stop_event. We still
        # emit the user_action so it shows up in the recording.
        if action == "quit" and self._stop is not None:
            self._stop.set()

        session_time = self._cursor_time_at(time.monotonic())

        if self._on_ack is not None:
            try:
                self._on_ack(self._ack_label(action, char))
            except Exception:
                pass

        try:
            self._on_user_action(action, char, session_time)
        except Exception:
            pass

    def _resolve_action(self, char: str) -> Optional[str]:
        if char in self._marker_keys:
            return "marker_press"
        if char in self._clipboard_keys:
            return "clipboard_toggle"
        if char == "x":
            return "discard"
        if char == "m":
            return "mute_toggle"
        if char == self._debug_flag_key:
            return "debug_flag"
        if char == "q":
            return "quit"
        return None

    @staticmethod
    def _ack_label(action: str, char: str) -> str:
        if action == "marker_press":
            return f"marker `{char}`"
        if action == "clipboard_toggle":
            return f"clipboard {char}"
        return {
            "discard": "cancel (x)",
            "mute_toggle": "mute toggle (m)",
            "debug_flag": f"error flag ({char})",
            "quit": "quit (q)",
        }.get(action, action)

    def _undo_keypresses(self, count: int = 2) -> None:
        """Send N backspaces to undo the visible keystrokes that landed
        in whichever app currently has keyboard focus before the
        double-tap was committed. Best-effort — in non-text contexts
        backspace is a no-op anyway."""
        try:
            for _ in range(count):
                self._kb.tap(keyboard.Key.backspace)
        except Exception:
            pass
