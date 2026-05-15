"""Smoke tests for the widget HTTP/SSE server."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request

import queue

from voice_dictation.clock import next_seq
from voice_dictation.event_bus import EventBus
from voice_dictation.transcript_stream_manager import TranscriptStreamManager
from voice_dictation.types import Event, Segment, TOPIC_AUDIO_SILENCE
from voice_dictation.widget_server import WidgetServer


class _Clock:
    def __init__(self):
        self.ms = 0

    def now_accepted_ms(self) -> int:
        return self.ms


class TestWidgetServer(unittest.TestCase):
    def setUp(self) -> None:
        self.bus = EventBus()
        self.clock = _Clock()
        self.fast_q: queue.Queue[Segment] = queue.Queue()
        self.stop = threading.Event()
        self.manager = TranscriptStreamManager(
            self.bus, self.clock, self.fast_q, None, self.stop,
        )
        self.manager.start()
        self.widget = WidgetServer(self.bus, self.clock, self.manager, port=0)
        self.widget.start()
        self.base = f"http://127.0.0.1:{self.widget.actual_port}"

    def tearDown(self) -> None:
        self.widget.shutdown()
        self.manager.shutdown()
        self.bus.shutdown()

    def _get(self, path: str, timeout: float = 2.0) -> tuple[int, str]:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")

    def test_index_serves_html(self) -> None:
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("voice_dictation", body)
        self.assertIn("EventSource", body)

    def test_snapshot_returns_state_and_transcript(self) -> None:
        # Publish a paste-action so the paste log has content.
        self.bus.publish(Event(
            topic="paste.actions", emit_accepted_ms=1000, seq=next_seq(),
            payload={"paste_idx": 7, "start_accepted_ms": 0,
                     "end_accepted_ms": 1000, "text": "hi"},
        ))
        self.bus.wait_idle(timeout=1.0)
        status, body = self._get("/snapshot")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("state", data)
        self.assertIn("pastes", data)
        self.assertIn("transcript", data)
        self.assertEqual(len(data["pastes"]), 1)
        self.assertEqual(data["pastes"][0]["paste_idx"], 7)
        self.assertEqual(data["state"]["capture"], "passive")

    def test_sse_delivers_live_event(self) -> None:
        # Open SSE in a thread; publish an event; verify it arrives.
        received: list[str] = []
        ready = threading.Event()

        def listener():
            req = urllib.request.Request(self.base + "/events")
            with urllib.request.urlopen(req, timeout=3.0) as r:
                ready.set()
                buf = b""
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not received:
                    chunk = r.read(256)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        msg, buf = buf.split(b"\n\n", 1)
                        msg_str = msg.decode("utf-8", errors="replace")
                        if "audio.silence" in msg_str:
                            received.append(msg_str)
                            return

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        self.assertTrue(ready.wait(timeout=2.0))
        # Give the server a beat to register the SSE client before publish.
        time.sleep(0.1)
        self.bus.publish(Event(
            topic=TOPIC_AUDIO_SILENCE,
            emit_accepted_ms=42,
            seq=next_seq(),
            payload={"boundary_at_accepted_ms": 42, "duration_real_ms": 500},
        ))
        t.join(timeout=3.0)
        self.assertEqual(len(received), 1, "expected one SSE event")
        self.assertIn("\"emit_accepted_ms\": 42", received[0])


if __name__ == "__main__":
    unittest.main()
