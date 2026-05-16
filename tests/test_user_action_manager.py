"""Validate UserActionManager state machine without going through pynput."""

from __future__ import annotations

import threading
import unittest

from voice_dictation.event_bus import EventBus
from voice_dictation.runtime_config import AppConfig, HotkeyConfig
from voice_dictation.types import (
    Event, TOPIC_USER_ACTIONS, TOPIC_USER_PRESSES,
)
from voice_dictation.user_action_manager import UserActionManager


class _StubPreprocessor:
    """Minimal preprocessor stand-in. Returns the press's real_ms verbatim
    as accepted_ms (so tests can drive the state machine deterministically)."""

    def stamp_accepted_ms(self, real_ms: int, timeout: float = 1.0) -> int:
        return real_ms

    def now_accepted_ms(self) -> int:
        return 0


def _build():
    stop = threading.Event()
    bus = EventBus()
    pp = _StubPreprocessor()
    presses: list[Event] = []
    actions: list[Event] = []
    bus.subscribe(TOPIC_USER_PRESSES, presses.append, name="cap-presses")
    bus.subscribe(TOPIC_USER_ACTIONS, actions.append, name="cap-actions")
    m = UserActionManager(
        bus, pp, pp, stop,
        hotkey_cfg=HotkeyConfig(),
        app_cfg=AppConfig(),
        marker_keys={"1", "2"},
        listen=False,
    )
    m.start()
    return m, bus, presses, actions, stop


def _wait_for(predicate, timeout: float = 1.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


class TestClipboardPair(unittest.TestCase):
    def test_two_clipboard_toggles_compose_paste_window(self):
        m, bus, presses, actions, _ = _build()
        m._commit("r", real_ms=1000)
        m._commit("r", real_ms=5000)
        _wait_for(lambda: len(actions) >= 2)
        self.assertEqual([e.payload["action"] for e in actions],
                         ["recording_started", "paste_window_complete"])
        completed = actions[1]
        self.assertEqual(completed.payload["start_press_idx"], 1)
        self.assertEqual(completed.payload["end_press_idx"], 2)
        self.assertEqual(completed.payload["start_accepted_ms"], 1000)
        self.assertEqual(completed.payload["end_accepted_ms"], 5000)
        # Second press carries the is_clipboard_end flag.
        self.assertTrue(presses[1].payload["is_clipboard_end"])
        self.assertFalse(presses[0].payload["is_clipboard_end"])
        m.shutdown()
        bus.shutdown()

    def test_discard_cancels_open_recording(self):
        m, bus, _presses, actions, _ = _build()
        m._commit("r", real_ms=1000)
        m._commit("x", real_ms=2500)
        _wait_for(lambda: any(e.payload["action"] == "cancel" for e in actions))
        cancel_events = [e for e in actions if e.payload["action"] == "cancel"]
        self.assertEqual(len(cancel_events), 1)
        self.assertEqual(cancel_events[0].payload["start_accepted_ms"], 1000)
        self.assertEqual(cancel_events[0].payload["end_accepted_ms"], 2500)
        # State cleared: a fresh r should start a new recording, not close one.
        m._commit("r", real_ms=3000)
        _wait_for(lambda: any(e.payload["action"] == "recording_started"
                              and e.payload["start_accepted_ms"] == 3000
                              for e in actions))
        m.shutdown()
        bus.shutdown()

    def test_discard_with_no_open_recording_is_noop(self):
        m, bus, _presses, actions, _ = _build()
        m._commit("x", real_ms=1000)
        _wait_for(lambda: True, timeout=0.1)
        self.assertEqual(actions, [])
        m.shutdown()
        bus.shutdown()


class TestAside(unittest.TestCase):
    def test_aside_pair(self):
        m, bus, _presses, actions, _ = _build()
        m._commit("a", real_ms=1000)
        m._commit("a", real_ms=3000)
        _wait_for(lambda: len(actions) >= 2)
        self.assertEqual([e.payload["action"] for e in actions],
                         ["aside_started", "aside_ended"])
        m.shutdown()
        bus.shutdown()


class TestMarker(unittest.TestCase):
    def test_marker_press_opens_then_closes_interval(self):
        m, bus, _presses, actions, _ = _build()
        # First press opens.
        m._commit("1", real_ms=2000)
        _wait_for(lambda: len(actions) >= 1)
        self.assertEqual(actions[0].payload["action"], "marker_opened")
        self.assertEqual(actions[0].payload["key"], "1")
        self.assertEqual(actions[0].payload["accepted_ms"], 2000)
        # Second press of the same key closes.
        m._commit("1", real_ms=5000)
        _wait_for(lambda: len(actions) >= 2)
        self.assertEqual(actions[1].payload["action"], "marker_closed")
        self.assertEqual(actions[1].payload["key"], "1")
        self.assertEqual(actions[1].payload["open_accepted_ms"], 2000)
        self.assertEqual(actions[1].payload["close_accepted_ms"], 5000)
        m.shutdown()
        bus.shutdown()

    def test_two_marker_keys_are_independent(self):
        m, bus, _presses, actions, _ = _build()
        m._commit("1", real_ms=1000)  # open 1
        m._commit("2", real_ms=2000)  # open 2
        m._commit("1", real_ms=3000)  # close 1
        # 2 is still open.
        _wait_for(lambda: len(actions) >= 3)
        kinds = [a.payload["action"] for a in actions]
        self.assertEqual(
            kinds, ["marker_opened", "marker_opened", "marker_closed"],
        )
        m.shutdown()
        bus.shutdown()


class TestQuit(unittest.TestCase):
    def test_quit_sets_stop_event(self):
        m, bus, _presses, actions, stop = _build()
        self.assertFalse(stop.is_set())
        m._commit("q", real_ms=5000)
        _wait_for(stop.is_set)
        self.assertTrue(stop.is_set())
        m.shutdown()
        bus.shutdown()


if __name__ == "__main__":
    unittest.main()
