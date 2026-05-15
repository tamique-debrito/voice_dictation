"""UserActionManager — keystroke filter + accepted-ms stamping + action composer.

Pipeline (live):

  pynput on_press
   ├─ BareDoubleTap.feed(char)
   │   └─ on_double_tap → _commit(char, real_ms)
   │       ├─ preprocessor.stamp_accepted_ms(real_ms) → content_accepted_ms
   │       ├─ allocate press_idx
   │       ├─ backspace-undo the two visible keystrokes
   │       ├─ publish user.presses (raw committed press)
   │       └─ feed into state machine → publish user.actions for any
   │          composed event (paste_window_complete, cancel, marker, etc.)

State machine (composed actions):
  - clipboard_toggle: opens a recording window on first press; on the
    second press closes the window and emits paste_window_complete with
    start_press_idx, end_press_idx, start_accepted_ms, end_accepted_ms.
    user.presses for the closing press carries ``is_clipboard_end: True``
    so AudioWindowers can flush on it.
  - aside_toggle: same shape but emits aside_started / aside_ended.
  - discard: while a recording window is open, emit cancel and clear state.
  - marker_press / debug_flag / mute_toggle / quit: pass-through.

Replay-mode UserActionManager is provided by ``replay/mock_user_actions.py``
in Phase 6; this class only handles live keystrokes.

Test mode: ``listen=False`` skips pynput so unit tests can call
``_commit(...)`` directly without needing a keyboard.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Callable, Optional

try:
    from pynput import keyboard
    _PYNPUT_AVAILABLE = True
except Exception:  # pragma: no cover
    keyboard = None  # type: ignore
    _PYNPUT_AVAILABLE = False

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .hotkeys import BareDoubleTap, key_char
from .runtime_config import HotkeyConfig, AppConfig
from .types import Event, TOPIC_USER_PRESSES, TOPIC_USER_ACTIONS


logger = logging.getLogger(__name__)


class _SessionRealClock:
    """time.monotonic() since session start, expressed as int milliseconds."""

    def __init__(self) -> None:
        self._started: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started is None:
                self._started = time.monotonic()

    def now_ms(self) -> int:
        with self._lock:
            if self._started is None:
                return 0
            return int((time.monotonic() - self._started) * 1000)


class UserActionManager:
    def __init__(
        self,
        bus: EventBus,
        clock: AcceptedClock,
        preprocessor,
        stop_event: threading.Event,
        *,
        hotkey_cfg: HotkeyConfig,
        app_cfg: AppConfig,
        marker_keys: set[str],
        listen: bool = True,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._preprocessor = preprocessor
        self._stop = stop_event
        self._listen = listen and _PYNPUT_AVAILABLE
        self._hotkey_cfg = hotkey_cfg
        self._app_cfg = app_cfg

        self._marker_keys = set(marker_keys)
        # Map "toggle_recording" / "toggle_aside" / "discard_recording" from cfg
        # to single-character keys.
        self._clipboard_key = hotkey_cfg.keys.get("toggle_recording", "r")
        self._aside_key = hotkey_cfg.keys.get("toggle_aside", "a")
        self._discard_key = hotkey_cfg.keys.get("discard_recording", "x")
        self._debug_flag_key = app_cfg.debug_flag_key

        action_keys = (
            self._marker_keys
            | {self._clipboard_key, self._aside_key, self._discard_key}
            | {"q", "m", self._debug_flag_key}
        )
        self._dt = BareDoubleTap(
            window=1.0, keys=action_keys, on_double_tap=self._on_double_tap,
        )

        self._real_clock = _SessionRealClock()
        self._press_seq = itertools.count(1)
        self._listener = None
        self._kb: Optional["keyboard.Controller"] = None

        # State machine.
        self._open_recording: Optional[dict] = None
        self._open_aside: Optional[dict] = None
        # Per-marker-key open state (interval semantics: first press opens,
        # second press closes — like a parenthesis pair).
        self._open_markers: dict[str, dict] = {}

        logger.info(
            "UserActionManager initialized (listen=%s, clipboard=%r, aside=%r, "
            "discard=%r, debug_flag=%r, markers=%s)",
            self._listen, self._clipboard_key, self._aside_key,
            self._discard_key, self._debug_flag_key, sorted(self._marker_keys),
        )

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._real_clock.start()
        if not self._listen:
            logger.info("UserActionManager.start (listen=False; tests/replay only)")
            return
        if keyboard is None:
            logger.warning("pynput not available; UserActionManager will not listen")
            return
        self._kb = keyboard.Controller()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()
        logger.info("UserActionManager.start (listening)")

    def shutdown(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("listener.stop failed")
            self._listener = None
        logger.info("UserActionManager.shutdown")

    # ------------------------------------------------------------------
    # Raw key intake
    # ------------------------------------------------------------------

    def _on_press(self, key) -> None:
        char = key_char(key)
        if char is not None:
            self._dt.feed(char)

    def _on_double_tap(self, char: str) -> None:
        real_ms = self._real_clock.now_ms()
        # Backspace-out the two visible keystrokes; best-effort.
        if self._kb is not None:
            try:
                self._kb.tap(keyboard.Key.backspace)
                self._kb.tap(keyboard.Key.backspace)
            except Exception:
                logger.exception("backspace-undo failed")
        self._commit(char, real_ms)

    # ------------------------------------------------------------------
    # Press commit (test entry point)
    # ------------------------------------------------------------------

    def _commit(self, char: str, real_ms: int) -> None:
        """Commit a post-filter double-tap. Public for tests."""
        action = self._resolve_action(char)
        if action is None:
            return
        accepted_ms = self._preprocessor.stamp_accepted_ms(real_ms)
        press_idx = next(self._press_seq)
        # Both clipboard-close and aside-close presses end a paste window
        # (recording or aside). The audio windower flushes on this flag,
        # which lets the matching `paste_window_complete` / `aside_ended`
        # action complete promptly instead of waiting for more audio.
        is_clipboard_end = (
            (action == "clipboard_toggle" and self._open_recording is not None)
            or (action == "aside_toggle" and self._open_aside is not None)
        )
        # Publish the committed press.
        self._bus.publish(Event(
            topic=TOPIC_USER_PRESSES,
            emit_accepted_ms=accepted_ms,
            seq=next_seq(),
            payload={
                "press_idx": press_idx,
                "key": char,
                "action": action,
                "real_ms": real_ms,
                "content_accepted_ms": accepted_ms,
                "is_clipboard_end": is_clipboard_end,
            },
        ))
        # Compose downstream actions.
        self._apply_state_machine(
            action=action, char=char, press_idx=press_idx,
            accepted_ms=accepted_ms, real_ms=real_ms,
        )

    def _resolve_action(self, char: str) -> Optional[str]:
        if char in self._marker_keys:
            return "marker_press"
        if char == self._clipboard_key:
            return "clipboard_toggle"
        if char == self._aside_key:
            return "aside_toggle"
        if char == self._discard_key:
            return "discard"
        if char == "m":
            return "mute_toggle"
        if char == self._debug_flag_key:
            return "debug_flag"
        if char == "q":
            return "quit"
        return None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _publish_action(self, action: str, payload: dict, accepted_ms: int) -> None:
        self._bus.publish(Event(
            topic=TOPIC_USER_ACTIONS,
            emit_accepted_ms=accepted_ms,
            seq=next_seq(),
            payload={"action": action, **payload},
        ))

    def _apply_state_machine(
        self,
        *,
        action: str,
        char: str,
        press_idx: int,
        accepted_ms: int,
        real_ms: int,
    ) -> None:
        if action == "clipboard_toggle":
            if self._open_recording is None:
                self._open_recording = {
                    "start_press_idx": press_idx,
                    "start_accepted_ms": accepted_ms,
                    "start_real_ms": real_ms,
                    "asides": [],
                }
                self._publish_action(
                    "recording_started",
                    {
                        "start_press_idx": press_idx,
                        "start_accepted_ms": accepted_ms,
                    },
                    accepted_ms,
                )
            else:
                # Auto-close any dangling aside at recording end.
                if self._open_aside is not None:
                    a = self._open_aside
                    self._open_aside = None
                    self._publish_action(
                        "aside_ended",
                        {
                            "start_press_idx": a["start_press_idx"],
                            "end_press_idx": press_idx,
                            "start_accepted_ms": a["start_accepted_ms"],
                            "end_accepted_ms": accepted_ms,
                        },
                        accepted_ms,
                    )
                    self._open_recording["asides"].append({
                        "start_press_idx": a["start_press_idx"],
                        "end_press_idx": press_idx,
                        "start_accepted_ms": a["start_accepted_ms"],
                        "end_accepted_ms": accepted_ms,
                    })
                rec = self._open_recording
                self._open_recording = None
                self._publish_action(
                    "paste_window_complete",
                    {
                        "start_press_idx": rec["start_press_idx"],
                        "end_press_idx": press_idx,
                        "start_accepted_ms": rec["start_accepted_ms"],
                        "end_accepted_ms": accepted_ms,
                        "aside_intervals": rec["asides"],
                    },
                    accepted_ms,
                )
            return

        if action == "aside_toggle":
            if self._open_aside is None:
                self._open_aside = {
                    "start_press_idx": press_idx,
                    "start_accepted_ms": accepted_ms,
                }
                self._publish_action(
                    "aside_started",
                    {"start_press_idx": press_idx,
                     "start_accepted_ms": accepted_ms},
                    accepted_ms,
                )
            else:
                a = self._open_aside
                self._open_aside = None
                interval = {
                    "start_press_idx": a["start_press_idx"],
                    "end_press_idx": press_idx,
                    "start_accepted_ms": a["start_accepted_ms"],
                    "end_accepted_ms": accepted_ms,
                }
                if self._open_recording is not None:
                    self._open_recording["asides"].append(interval)
                self._publish_action("aside_ended", interval, accepted_ms)
            return

        if action == "discard":
            if self._open_recording is not None:
                rec = self._open_recording
                self._open_recording = None
                # Drop any open aside as well; cancellation supersedes it.
                self._open_aside = None
                self._publish_action(
                    "cancel",
                    {
                        "start_press_idx": rec["start_press_idx"],
                        "discard_press_idx": press_idx,
                        "start_accepted_ms": rec["start_accepted_ms"],
                        "end_accepted_ms": accepted_ms,
                    },
                    accepted_ms,
                )
            return

        if action == "marker_press":
            open_state = self._open_markers.get(char)
            if open_state is None:
                # Open the marker interval.
                self._open_markers[char] = {
                    "open_press_idx": press_idx,
                    "open_accepted_ms": accepted_ms,
                }
                self._publish_action(
                    "marker_opened",
                    {"press_idx": press_idx, "key": char,
                     "accepted_ms": accepted_ms},
                    accepted_ms,
                )
            else:
                # Close it.
                del self._open_markers[char]
                self._publish_action(
                    "marker_closed",
                    {
                        "key": char,
                        "open_press_idx": open_state["open_press_idx"],
                        "close_press_idx": press_idx,
                        "open_accepted_ms": open_state["open_accepted_ms"],
                        "close_accepted_ms": accepted_ms,
                    },
                    accepted_ms,
                )
            return

        if action == "debug_flag":
            self._publish_action(
                "debug_flag",
                {"press_idx": press_idx, "accepted_ms": accepted_ms},
                accepted_ms,
            )
            return

        if action == "mute_toggle":
            self._publish_action(
                "mute_toggle",
                {"press_idx": press_idx, "accepted_ms": accepted_ms},
                accepted_ms,
            )
            return

        if action == "quit":
            self._publish_action(
                "quit",
                {"press_idx": press_idx, "accepted_ms": accepted_ms},
                accepted_ms,
            )
            self._stop.set()
            return
