"""Replay determinism tests.

Strategy: build a synthetic session by directly publishing events into a
bus + persisters, then run the ReplayCoordinator against the persisted
JSONL files and verify the manager emits paste_actions that match the
original ones."""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import time
import unittest

from voice_dictation.clock import next_seq
from voice_dictation.event_bus import EventBus
from voice_dictation.replay.coordinator import TOPIC_FILES
from voice_dictation.replay.harness import (
    _load_live_pastes, _normalize_pastes, run_replay,
)
from voice_dictation.streams import EventStreamPersister
from voice_dictation.transcript_stream_manager import TranscriptStreamManager
from voice_dictation.types import (
    Event, Segment, WindowCompletion,
    TOPIC_PASTE_ACTIONS, TOPIC_PROGRESS_FAST,
    TOPIC_SEGMENTS_FAST, TOPIC_USER_ACTIONS, TOPIC_USER_PRESSES,
)


class _Clock:
    def __init__(self):
        self.ms = 0

    def now_accepted_ms(self) -> int:
        return self.ms


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def _segment_payload(seg: Segment) -> dict:
    return {
        "window_id": seg.window_id,
        "text": seg.text,
        "content_accepted_ms_start": seg.content_accepted_ms_start,
        "content_accepted_ms_end": seg.content_accepted_ms_end,
        "ends_on_marker_idx": seg.ends_on_marker_idx,
        "words": [],
    }


class TestReplayDeterminism(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="vd2-replay-")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def _live_session(self) -> list[dict]:
        """Record a tiny synthetic session: one paste window with one segment."""
        bus = EventBus()
        clock = _Clock()
        fast_q: queue.Queue[Segment] = queue.Queue()
        stop = threading.Event()
        manager = TranscriptStreamManager(bus, clock, fast_q, None, stop)
        manager.start()
        # Persisters for the topics we'll need in replay.
        persisters = [
            EventStreamPersister(
                bus, topic, os.path.join(self.dir, filename),
            )
            for topic, filename in TOPIC_FILES.items()
        ]

        # Synthetic flow: press r (open), then a 4000ms segment from fast,
        # then press r again (close → paste_window_complete).
        # Press 1 (open):
        clock.ms = 1000
        bus.publish(Event(
            topic=TOPIC_USER_PRESSES, emit_accepted_ms=1000, seq=next_seq(),
            payload={"press_idx": 1, "key": "r", "is_clipboard_end": False,
                     "real_ms": 1000, "content_accepted_ms": 1000},
        ))

        # Segment arrives at accepted_ms 5000, annotated with marker_idx=2
        # so the paste completes via the marker-end shortcut (the new rule).
        clock.ms = 5000
        seg = Segment(
            text="hello there",
            content_accepted_ms_start=1000, content_accepted_ms_end=5000,
            words=[], window_id=1, ends_on_marker_idx=2,
        )
        fast_q.put(seg)
        # Persist a matching bus event so replay sees it (in live, the
        # Transcriber publishes this; here we publish manually since we
        # bypassed the Transcriber).
        bus.publish(Event(
            topic=TOPIC_SEGMENTS_FAST, emit_accepted_ms=5000, seq=next_seq(),
            payload=_segment_payload(seg),
        ))
        # WindowCompletion advances the manager's watermark past the
        # paste window's end_accepted_ms — the real Transcriber emits
        # one of these after every window.
        fast_q.put(WindowCompletion(
            stream_label="fast", window_id=1, end_accepted_ms=5500,
        ))
        bus.publish(Event(
            topic=TOPIC_PROGRESS_FAST, emit_accepted_ms=5500, seq=next_seq(),
            payload={"window_id": 1,
                     "transcribed_up_to_accepted_ms": 5500,
                     "dropped": False},
        ))

        # Press 2 (close) and paste_window_complete action.
        clock.ms = 5500
        bus.publish(Event(
            topic=TOPIC_USER_PRESSES, emit_accepted_ms=5500, seq=next_seq(),
            payload={"press_idx": 2, "key": "r", "is_clipboard_end": True,
                     "real_ms": 5500, "content_accepted_ms": 5500},
        ))
        bus.publish(Event(
            topic=TOPIC_USER_ACTIONS, emit_accepted_ms=5500, seq=next_seq(),
            payload={"action": "paste_window_complete",
                     "start_press_idx": 1, "end_press_idx": 2,
                     "start_accepted_ms": 1000, "end_accepted_ms": 5500},
        ))

        _wait_for(lambda: os.path.exists(
            os.path.join(self.dir, "paste_actions.jsonl")
        ) and os.path.getsize(
            os.path.join(self.dir, "paste_actions.jsonl")
        ) > 0)
        time.sleep(0.2)  # flush

        manager.shutdown()
        for p in persisters:
            p.shutdown()
        bus.shutdown()

        return _load_live_pastes(self.dir)

    def test_replay_emits_matching_paste(self):
        live = self._live_session()
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["text"], "hello there")
        # Now run the replay against the same dir.
        out_path = os.path.join(self.dir, "replay_paste_actions.jsonl")
        n, replay = run_replay(self.dir, out_paste_actions=out_path)
        self.assertGreater(n, 0)
        replay_norm = _normalize_pastes(replay)
        self.assertEqual(live, replay_norm)


if __name__ == "__main__":
    unittest.main()
