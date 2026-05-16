"""Live-stream dictation mode: triple-tap opens a streaming window where
text is emitted via (backspaces, append) edits as transcription arrives.

These tests exercise the TranscriptStreamManager half of the feature —
the UserActionManager half (triple-tap detection, is_live_stream flag on
recording_started / paste_window_complete) is verified separately."""

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
    fast_q: queue.Queue = queue.Queue()
    hq_q: queue.Queue = queue.Queue() if with_hq else None
    pastes: list[Event] = []
    bus.subscribe(TOPIC_PASTE_ACTIONS, pastes.append, name="cap-pastes")
    m = TranscriptStreamManager(bus, clock, fast_q, hq_q, stop)
    m.start()
    return m, bus, clock, fast_q, hq_q, pastes


def _seg(text: str, start: int, end: int, *, marker: int = None,
         window_id: int = 1) -> Segment:
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


def _open_live(bus: EventBus, *, start_ms: int = 0,
               start_press_idx: int = 1) -> None:
    bus.publish(Event(
        topic=TOPIC_USER_ACTIONS,
        emit_accepted_ms=start_ms,
        seq=1,
        payload={
            "action": "recording_started",
            "start_press_idx": start_press_idx,
            "start_accepted_ms": start_ms,
            "is_live_stream": True,
        },
    ))


def _close_live(bus: EventBus, *, end_ms: int, end_press_idx: int = 2,
                start_press_idx: int = 1, start_ms: int = 0) -> None:
    bus.publish(Event(
        topic=TOPIC_USER_ACTIONS,
        emit_accepted_ms=end_ms,
        seq=99,
        payload={
            "action": "paste_window_complete",
            "start_press_idx": start_press_idx,
            "end_press_idx": end_press_idx,
            "start_accepted_ms": start_ms,
            "end_accepted_ms": end_ms,
            "aside_intervals": [],
            "is_live_stream": True,
        },
    ))


def _stream_edits(pastes: list[Event]) -> list[tuple[int, str]]:
    out = []
    for ev in pastes:
        if ev.payload.get("kind") == "stream_edit":
            out.append((
                int(ev.payload.get("backspaces", 0) or 0),
                ev.payload.get("text", "") or "",
            ))
    return out


def _apply(edits: list[tuple[int, str]]) -> str:
    buf = ""
    for bs, text in edits:
        if bs:
            buf = buf[: max(0, len(buf) - bs)]
        buf += text
    return buf


class TestLiveStream(unittest.TestCase):
    def test_fast_only_appends(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _open_live(bus)
        # Wait for the recording_started to be processed.
        time.sleep(0.05)
        fast_q.put(_seg("hello", 0, 1500))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        fast_q.put(_seg("world", 1500, 3000))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 2)
        edits = _stream_edits(pastes)
        # Pure appends — no backspaces.
        for bs, _t in edits:
            self.assertEqual(bs, 0)
        self.assertEqual(_apply(edits), "hello world")
        m.shutdown()
        bus.shutdown()

    def test_hq_overwrites_fast(self):
        m, bus, _, fast_q, hq_q, pastes = _build(with_hq=True)
        _open_live(bus)
        time.sleep(0.05)
        fast_q.put(_seg("helo wrld", 0, 3000))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        # HQ catches up with a corrected version covering the same range.
        hq_q.put(_seg("hello world", 0, 3000))
        _wait_for(lambda: _apply(_stream_edits(pastes)) == "hello world",
                  timeout=2.0)
        edits = _stream_edits(pastes)
        # At least one edit must have used backspaces (to wipe the Fast text).
        self.assertTrue(any(bs > 0 for bs, _t in edits),
                        f"expected a backspace edit, got {edits}")
        self.assertEqual(_apply(edits), "hello world")
        m.shutdown()
        bus.shutdown()

    def test_aside_pauses_emit(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _open_live(bus)
        time.sleep(0.05)
        fast_q.put(_seg("hello", 0, 1500))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        before = len(_stream_edits(pastes))
        # Aside opens at 1500.
        bus.publish(Event(
            topic=TOPIC_USER_ACTIONS, emit_accepted_ms=1500, seq=10,
            payload={
                "action": "aside_started",
                "start_press_idx": 5,
                "start_accepted_ms": 1500,
            },
        ))
        time.sleep(0.05)
        # Segments arriving during the aside should NOT trigger new emits.
        fast_q.put(_seg("aside content", 1500, 3000))
        time.sleep(0.1)
        self.assertEqual(len(_stream_edits(pastes)), before,
                         "no emit should fire while aside is open")
        # Aside closes; the interval is added to exclude.
        bus.publish(Event(
            topic=TOPIC_USER_ACTIONS, emit_accepted_ms=3000, seq=11,
            payload={
                "action": "aside_ended",
                "start_press_idx": 5,
                "end_press_idx": 6,
                "start_accepted_ms": 1500,
                "end_accepted_ms": 3000,
            },
        ))
        # Real content after the aside should now emit, excluding the aside.
        fast_q.put(_seg("after", 3000, 4500))
        _wait_for(lambda: _apply(_stream_edits(pastes)) == "hello after",
                  timeout=2.0)
        m.shutdown()
        bus.shutdown()

    def test_discard_backspaces_everything(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _open_live(bus)
        time.sleep(0.05)
        fast_q.put(_seg("hello world", 0, 3000))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        emitted_so_far = _apply(_stream_edits(pastes))
        self.assertEqual(emitted_so_far, "hello world")
        # Discard the live recording.
        bus.publish(Event(
            topic=TOPIC_USER_ACTIONS, emit_accepted_ms=3500, seq=20,
            payload={
                "action": "cancel",
                "start_press_idx": 1,
                "discard_press_idx": 7,
                "start_accepted_ms": 0,
                "end_accepted_ms": 3500,
                "is_live_stream": True,
            },
        ))
        _wait_for(lambda: _apply(_stream_edits(pastes)) == "", timeout=2.0)
        # Subsequent segments should NOT trigger any more emits.
        before = len(_stream_edits(pastes))
        fast_q.put(_seg("late", 3500, 5000))
        time.sleep(0.1)
        self.assertEqual(len(_stream_edits(pastes)), before)
        m.shutdown()
        bus.shutdown()

    def test_close_does_not_emit_separate_paste(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _open_live(bus)
        time.sleep(0.05)
        fast_q.put(_seg("hello world", 0, 3000))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        # Close the live stream.
        _close_live(bus, end_ms=3000)
        # Drive a WindowCompletion past the close so the watermark covers.
        fast_q.put(WindowCompletion(
            stream_label="fast", window_id=1, end_accepted_ms=3200,
        ))
        time.sleep(0.1)
        # No "paste"-kind event should have been emitted — everything is
        # stream_edits. The text seen by the OS should be "hello world".
        kinds = [ev.payload.get("kind", "paste") for ev in pastes]
        self.assertNotIn("paste", kinds,
                         f"expected only stream_edits, got kinds={kinds}")
        self.assertEqual(_apply(_stream_edits(pastes)), "hello world")
        m.shutdown()
        bus.shutdown()

    def test_close_carries_final_text_for_clipboard_sync(self):
        # Regression: live streams used to leave the system clipboard
        # untouched. The closing stream_edit now carries ``final_text``
        # so PasteExecutor can pyperclip.copy() it.
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        _open_live(bus)
        time.sleep(0.05)
        fast_q.put(_seg("hello world", 0, 3000))
        _wait_for(lambda: len(_stream_edits(pastes)) >= 1)
        _close_live(bus, end_ms=3000)
        fast_q.put(WindowCompletion(
            stream_label="fast", window_id=1, end_accepted_ms=3200,
        ))
        _wait_for(lambda: any(
            ev.payload.get("final_text") is not None for ev in pastes
        ), timeout=2.0)
        closing = [ev for ev in pastes
                   if ev.payload.get("final_text") is not None]
        self.assertEqual(len(closing), 1)
        self.assertEqual(closing[0].payload["final_text"], "hello world")
        m.shutdown()
        bus.shutdown()


class TestRegularPasteStillWorks(unittest.TestCase):
    """The unified _compute_window_text path must not regress regular
    (non-live) paste-window completion."""

    def test_regular_paste_still_fires(self):
        m, bus, _, fast_q, _, pastes = _build(with_hq=False)
        fast_q.put(_seg("hello world", 0, 5000))
        time.sleep(0.05)
        bus.publish(Event(
            topic=TOPIC_USER_ACTIONS, emit_accepted_ms=5000, seq=30,
            payload={
                "action": "paste_window_complete",
                "start_press_idx": 1, "end_press_idx": 2,
                "start_accepted_ms": 0, "end_accepted_ms": 5000,
                "is_live_stream": False,
            },
        ))
        _wait_for(lambda: any(
            ev.payload.get("kind", "paste") == "paste" for ev in pastes
        ))
        paste_evs = [ev for ev in pastes
                     if ev.payload.get("kind", "paste") == "paste"]
        self.assertEqual(len(paste_evs), 1)
        self.assertEqual(paste_evs[0].payload["text"], "hello world")
        m.shutdown()
        bus.shutdown()


if __name__ == "__main__":
    unittest.main()
