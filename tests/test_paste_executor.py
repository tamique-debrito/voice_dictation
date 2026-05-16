"""PasteExecutor — ordering and serialization.

Regression test for the live-stream interleave bug: two close-in-time
stream_edit events used to spawn concurrent threads that called
``keyboard.type()`` simultaneously, producing char-by-char interleaved
output (e.g. "YTehsa,t 'yse..." for "Yes,..." + "That's..."). The fix
serializes all paste.actions through a single worker thread."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from voice_dictation.event_bus import EventBus
from voice_dictation.paste_executor import PasteExecutor
from voice_dictation.types import Event, TOPIC_PASTE_ACTIONS


class _FakeKb:
    """Stand-in for pynput.keyboard.Controller that records calls and
    sleeps inside ``type()`` to surface any interleaving."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.log: list[tuple[str, object]] = []

    def tap(self, key) -> None:
        with self.lock:
            self.log.append(("tap", key))

    def type(self, text: str) -> None:
        # Mimic pynput's per-char latency. If two type() calls run
        # concurrently, individual chars from each would interleave in
        # ``self.log`` — which is what the serialization fix prevents.
        for ch in text:
            time.sleep(0.005)
            with self.lock:
                self.log.append(("type", ch))


def _stream(text: str) -> Event:
    return Event(
        topic=TOPIC_PASTE_ACTIONS,
        emit_accepted_ms=0,
        seq=0,
        payload={
            "kind": "stream_edit",
            "backspaces": 0,
            "text": text,
            "start_press_idx": 1,
        },
    )


class TestPasteSerialization(unittest.TestCase):
    def test_concurrent_stream_edits_do_not_interleave(self):
        bus = EventBus()
        stop = threading.Event()
        fake_kb = _FakeKb()
        # PasteExecutor checks _PYNPUT_OK at construction; bypass by
        # patching the keyboard.Controller call site.
        with mock.patch(
            "voice_dictation.paste_executor.keyboard.Controller",
            return_value=fake_kb,
        ):
            pe = PasteExecutor(bus, stop)
        try:
            # Fire two stream_edits back-to-back. Without serialization
            # the chars would interleave in fake_kb.log.
            bus.publish(_stream("hello"))
            bus.publish(_stream("world"))
            # Wait for both type() calls to drain.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with fake_kb.lock:
                    typed = "".join(c for op, c in fake_kb.log if op == "type")
                if typed == "helloworld":
                    break
                time.sleep(0.01)
            with fake_kb.lock:
                typed = "".join(c for op, c in fake_kb.log if op == "type")
            self.assertEqual(typed, "helloworld",
                             f"keystrokes interleaved: log={fake_kb.log}")
        finally:
            pe.shutdown()
            bus.shutdown()


if __name__ == "__main__":
    unittest.main()
