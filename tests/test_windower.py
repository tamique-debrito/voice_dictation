"""Validate AudioWindower flush rules end-to-end with a real EventBus."""

from __future__ import annotations

import queue
import threading
import time
import unittest

from voice_dictation.audio_windower import AudioWindower
from voice_dictation.event_bus import EventBus
from voice_dictation.runtime_config import (
    AudioCaptureConfig, FasterWhisperConfig, StreamConfig,
)
from voice_dictation.types import (
    AcceptedAudioSegment, AudioWindow, Event,
    TOPIC_AUDIO_SILENCE, TOPIC_USER_PRESSES,
)


SR = 16000
BYTES_PER_MS = SR * 2 // 1000   # 32 bytes per ms at 16k mono int16


class _FixedClock:
    def __init__(self) -> None:
        self.ms = 0

    def now_accepted_ms(self) -> int:
        return self.ms


def _build(cfg_overrides: dict = None):
    audio_cfg = AudioCaptureConfig()
    stream_cfg = StreamConfig(
        label="fast",
        max_window_seconds=2.0,        # short for fast tests
        min_window_seconds=0.5,
        flush_on_markers=True,
        window_q_maxsize=4,
        fw=FasterWhisperConfig(),
    )
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(stream_cfg, k, v)
    bus = EventBus()
    clock = _FixedClock()
    stop = threading.Event()
    w = AudioWindower(stream_cfg, audio_cfg, bus, clock, stop)
    w.start()
    return w, bus, clock


def _seg(start_ms: int, dur_ms: int) -> AcceptedAudioSegment:
    pcm = b"\x00" * (BYTES_PER_MS * dur_ms)
    return AcceptedAudioSegment(
        pcm=pcm,
        start_accepted_ms=start_ms,
        end_accepted_ms=start_ms + dur_ms,
    )


class TestMaxWindow(unittest.TestCase):
    def test_flush_on_max_window(self):
        w, bus, _ = _build()
        # Feed 2.5s in 500ms chunks. max=2s, so after 2s a flush should fire.
        for i in range(5):
            w.put(_seg(i * 500, 500))
        try:
            win = w.window_q.get(timeout=1.0)
        except queue.Empty:
            self.fail("expected a window on max_window flush")
        self.assertEqual(win.flush_reason, "max_window")
        self.assertEqual(win.start_accepted_ms, 0)
        self.assertGreaterEqual(win.end_accepted_ms, 2000)
        self.assertIsNone(win.ends_on_marker_idx)
        w.shutdown()
        bus.shutdown()


class TestMarkerFlush(unittest.TestCase):
    def test_silence_marker_flushes_after_min(self):
        w, bus, _ = _build()
        # Min=500ms, accumulate 600ms then publish silence boundary → flush.
        w.put(_seg(0, 300))
        w.put(_seg(300, 300))
        bus.publish(Event(
            topic=TOPIC_AUDIO_SILENCE,
            emit_accepted_ms=600,
            seq=1,
            payload={"boundary_at_accepted_ms": 600, "duration_real_ms": 500},
        ))
        # Subscriber dispatch is async — wait briefly.
        try:
            win = w.window_q.get(timeout=1.0)
        except queue.Empty:
            self.fail("expected a window on silence-marker flush")
        self.assertEqual(win.flush_reason, "marker")
        self.assertEqual(win.end_accepted_ms, 600)
        self.assertIsNone(win.ends_on_marker_idx)
        w.shutdown()
        bus.shutdown()

    def test_marker_before_min_is_queued(self):
        w, bus, _ = _build()
        # Min=500ms. Marker arrives at 100ms — should NOT flush.
        w.put(_seg(0, 100))
        bus.publish(Event(
            topic=TOPIC_AUDIO_SILENCE,
            emit_accepted_ms=100, seq=1,
            payload={"boundary_at_accepted_ms": 100, "duration_real_ms": 500},
        ))
        time.sleep(0.1)  # allow subscriber to process
        # No window yet.
        self.assertTrue(w.window_q.empty())
        # Accumulate past min — the queued marker should now fire.
        w.put(_seg(100, 500))
        try:
            win = w.window_q.get(timeout=1.0)
        except queue.Empty:
            self.fail("expected queued marker to flush once min reached")
        self.assertEqual(win.flush_reason, "marker")
        self.assertEqual(win.end_accepted_ms, 600)
        w.shutdown()
        bus.shutdown()

    def test_clipboard_end_marker_carries_idx(self):
        w, bus, _ = _build()
        w.put(_seg(0, 700))
        bus.publish(Event(
            topic=TOPIC_USER_PRESSES,
            emit_accepted_ms=700, seq=2,
            payload={"press_idx": 17, "is_clipboard_end": True},
        ))
        try:
            win = w.window_q.get(timeout=1.0)
        except queue.Empty:
            self.fail("expected a window on clipboard-end flush")
        self.assertEqual(win.flush_reason, "marker_end_clipboard")
        self.assertEqual(win.ends_on_marker_idx, 17)
        w.shutdown()
        bus.shutdown()

    def test_flush_on_markers_disabled(self):
        # HQ-like config: don't flush on markers, only on max.
        w, bus, _ = _build({"flush_on_markers": False})
        w.put(_seg(0, 1000))
        bus.publish(Event(
            topic=TOPIC_AUDIO_SILENCE,
            emit_accepted_ms=1000, seq=3,
            payload={"boundary_at_accepted_ms": 1000, "duration_real_ms": 500},
        ))
        time.sleep(0.15)
        self.assertTrue(w.window_q.empty())
        # Continue until max=2s.
        w.put(_seg(1000, 1100))
        try:
            win = w.window_q.get(timeout=1.0)
        except queue.Empty:
            self.fail("expected max_window flush in flush_on_markers=False mode")
        self.assertEqual(win.flush_reason, "max_window")
        w.shutdown()
        bus.shutdown()


class TestShutdownFlush(unittest.TestCase):
    def test_shutdown_flushes_trailing_audio(self):
        w, bus, _ = _build()
        w.put(_seg(0, 200))
        w.shutdown()
        try:
            win = w.window_q.get(timeout=0.5)
        except queue.Empty:
            self.fail("expected trailing audio to be flushed on shutdown")
        self.assertEqual(win.flush_reason, "shutdown")
        bus.shutdown()


if __name__ == "__main__":
    unittest.main()
