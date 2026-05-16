"""Validate TranscriptStreamManager: merge + paste completion rules."""

from __future__ import annotations

import queue
import threading
import time
import unittest

from voice_dictation.event_bus import EventBus
from voice_dictation.transcript_stream_manager import TranscriptStreamManager
from voice_dictation.types import (
    Event, Segment, WindowCompletion,
    TOPIC_PASTE_ACTIONS, TOPIC_USER_ACTIONS,
)


class _Clock:
    def __init__(self) -> None:
        self.ms = 0

    def now_accepted_ms(self) -> int:
        return self.ms


def _build(with_hq: bool = True):
    bus = EventBus()
    clock = _Clock()
    stop = threading.Event()
    fast_q: queue.Queue[Segment] = queue.Queue()
    hq_q: queue.Queue[Segment] = queue.Queue() if with_hq else None
    pastes: list[Event] = []
    bus.subscribe(TOPIC_PASTE_ACTIONS, pastes.append, name="cap-pastes")
    m = TranscriptStreamManager(bus, clock, fast_q, hq_q, stop)
    m.start()
    return m, bus, clock, fast_q, hq_q, pastes


def _seg(text: str, start: int, end: int, *, marker: int = None, window_id: int = 1) -> Segment:
    return Segment(
        text=text,
        content_accepted_ms_start=start,
        content_accepted_ms_end=end,
        words=[],
        window_id=window_id,
        ends_on_marker_idx=marker,
    )


def _wait_for(predicate, timeout: float = 1.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)


def _publish_paste_window(bus: EventBus, **kw) -> None:
    bus.publish(Event(
        topic=TOPIC_USER_ACTIONS,
        emit_accepted_ms=kw.get("end_accepted_ms", 0),
        seq=999,
        payload={"action": "paste_window_complete", **kw},
    ))


class TestTimestampCompletion(unittest.TestCase):
    def test_paste_emits_when_fast_covers_range(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("hello world", 0, 5000))
        time.sleep(0.05)
        _publish_paste_window(
            bus,
            start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "hello world")
        self.assertEqual(pastes[0].payload["paste_idx"], 1)
        m.shutdown()
        bus.shutdown()

    def test_paste_waits_for_segment_when_range_not_covered(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        time.sleep(0.05)
        self.assertEqual(pastes, [])
        # Now a segment arrives covering the range.
        fast_q.put(_seg("late hello", 0, 5200))
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "late hello")
        m.shutdown()
        bus.shutdown()

    def test_hq_wins_over_fast_when_both_cover(self):
        m, bus, _, fast_q, hq_q, pastes = _build(with_hq=True)
        fast_q.put(_seg("fast text", 0, 5000))
        hq_q.put(_seg("HQ text", 0, 5000))
        time.sleep(0.05)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "HQ text")
        m.shutdown()
        bus.shutdown()


class TestWatermarkCompletion(unittest.TestCase):
    """With the new watermark logic, completion is gated on
    ``transcribed_up_to + tolerance >= end_accepted_ms``. Marker-idx is
    no longer a separate code path — it's just metadata on segments."""

    def test_segment_end_within_tolerance_fires(self):
        # Segment ends at 4800, press at 5000, default tolerance 200 → fires.
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("almost done", 0, 4800))
        time.sleep(0.05)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "almost done")
        m.shutdown()
        bus.shutdown()

    def test_segment_end_beyond_tolerance_waits(self):
        # Segment ends at 4500, press at 5000, tolerance 200 → 4700 < 5000.
        # Paste should NOT fire.
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("too short", 0, 4500))
        time.sleep(0.05)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        time.sleep(0.1)
        self.assertEqual(pastes, [])
        m.shutdown()
        bus.shutdown()

    def test_window_completion_advances_watermark_to_fire(self):
        # Segment ends at 4500 (too short alone), but a WindowCompletion
        # for window end=5100 advances the watermark above the press.
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("partial", 0, 4500))
        fast_q.put(WindowCompletion(
            stream_label="fast", window_id=1, end_accepted_ms=5100,
        ))
        time.sleep(0.05)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "partial")
        m.shutdown()
        bus.shutdown()


class TestMultiSegmentConcat(unittest.TestCase):
    def test_overlapping_range_segments_are_joined(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("hello", 0, 1500))
        fast_q.put(_seg("world", 1500, 3000))
        fast_q.put(_seg("done", 3000, 5000))
        time.sleep(0.05)
        _publish_paste_window(
            bus, start_press_idx=1, end_press_idx=2,
            start_accepted_ms=0, end_accepted_ms=5000,
        )
        _wait_for(lambda: len(pastes) >= 1)
        self.assertEqual(pastes[0].payload["text"], "hello world done")
        m.shutdown()
        bus.shutdown()


if __name__ == "__main__":
    unittest.main()
