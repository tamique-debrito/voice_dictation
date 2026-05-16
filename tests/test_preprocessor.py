"""Validate AudioPreprocessor with deterministic VAD classifications.

Patches ``webrtcvad.Vad.is_speech`` to return a scripted voiced/silent
sequence so we can assert exact behavior of the sliding-window gate,
accepted_ms cursor advancement, silence-boundary emission, and segment
fan-out — without depending on real human-speech audio.
"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from voice_dictation.audio_preprocessor import (
    AudioPreprocessor, VAD_FRAME_MS, BYTES_PER_SAMPLE,
)
from voice_dictation.event_bus import EventBus
from voice_dictation.runtime_config import (
    AudioCaptureConfig, PreprocessorConfig,
)
from voice_dictation.types import (
    AcceptedAudioSegment, Event, TOPIC_AUDIO_SILENCE,
)


SAMPLE_RATE = 16000
SUB_FRAME_BYTES = SAMPLE_RATE * VAD_FRAME_MS // 1000 * BYTES_PER_SAMPLE  # 640


class CaptureSink:
    def __init__(self) -> None:
        self.segments: list[AcceptedAudioSegment] = []

    def put(self, seg: AcceptedAudioSegment, timeout: float = 0.5) -> None:
        self.segments.append(seg)


class ScriptedVAD:
    """Yields the next voiced/silent boolean each call."""

    def __init__(self, sequence: list[bool]) -> None:
        self.sequence = list(sequence)
        self.idx = 0

    def is_speech(self, frame, sample_rate) -> bool:
        if self.idx >= len(self.sequence):
            return False
        v = self.sequence[self.idx]
        self.idx += 1
        return v


class _CapturingBus:
    """Stand-in for EventBus that records publishes synchronously, so tests
    don't have to wait for subscriber worker threads."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)

    def shutdown(self) -> None:
        pass


def _build_preprocessor(seq: list[bool], cfg_overrides: dict = None):
    audio_cfg = AudioCaptureConfig()
    pp_cfg = PreprocessorConfig(
        voiced_gate_frames=5,
        voiced_gate_min_voiced=2,
        silence_boundary_ms=100,   # short for fast tests: 5 frames
    )
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(pp_cfg, k, v)
    bus = _CapturingBus()
    stop = threading.Event()
    pp = AudioPreprocessor(pp_cfg, audio_cfg, bus, stop)
    sink = CaptureSink()
    pp.add_sink(sink)
    pp._vad = ScriptedVAD(seq)
    silence_events = [e for e in bus.events if e.topic == TOPIC_AUDIO_SILENCE]
    # Return a closure over bus.events so tests see fresh state each access.
    return pp, sink, bus


def _feed(pp: AudioPreprocessor, n_sub_frames: int) -> None:
    """Feed n sub-frames of zero PCM (content is irrelevant — VAD is mocked)."""
    pcm = b"\x00" * (SUB_FRAME_BYTES * n_sub_frames)
    pp._feed(0.0, pcm)


class TestGate(unittest.TestCase):
    def test_all_voiced_admits_all_after_warmup(self):
        # Gate is N=5,M=2: needs ≥2 voiced in last 5. First voiced frame
        # only sees itself → 1 voiced → drop. Second sees 2 → admit.
        pp, sink, bus = _build_preprocessor([True] * 10)
        _feed(pp, 10)
        admitted_frames = sum(len(s.pcm) // SUB_FRAME_BYTES for s in sink.segments)
        # First frame dropped (only 1/5 voiced); frames 2..10 admitted = 9.
        self.assertEqual(admitted_frames, 9)
        # accepted_ms advances exactly by 20ms per admitted frame.
        self.assertEqual(pp.now_accepted_ms(), 9 * VAD_FRAME_MS)
        bus.shutdown()

    def test_all_silent_admits_nothing_no_boundary(self):
        # No admit → gate never armed → no silence boundary fires.
        pp, sink, bus = _build_preprocessor([False] * 20)
        _feed(pp, 20)
        self.assertEqual(sink.segments, [])
        self.assertEqual(bus.events, [])
        self.assertEqual(pp.now_accepted_ms(), 0)
        bus.shutdown()

    def test_silence_boundary_fires_once_after_admit(self):
        # 6 voiced (warm up gate), then 10 silent. silence_boundary_ms=100ms=5 frames.
        # Trace with N=5, M=2:
        #   T1 voiced=1 → DROP
        #   T2..T6 → ADMIT (5 admits)
        #   F7 voiced=4 → ADMIT (tail)
        #   F8 voiced=3 → ADMIT
        #   F9 voiced=2 → ADMIT
        #   F10 voiced=1 → DROP, dropped_run=1
        #   F11..F13 → DROP, dropped_run=4
        #   F14 → DROP, dropped_run=5 → BOUNDARY at accepted_ms=8*20=160
        #   F15..F16 → DROP, disarmed
        # Total admits: 8.
        seq = [True] * 6 + [False] * 10
        pp, sink, bus = _build_preprocessor(seq)
        _feed(pp, len(seq))
        admitted_frames = sum(len(s.pcm) // SUB_FRAME_BYTES for s in sink.segments)
        self.assertEqual(admitted_frames, 8)
        silence_events = [e for e in bus.events if e.topic == TOPIC_AUDIO_SILENCE]
        self.assertEqual(len(silence_events), 1)
        ev = silence_events[0]
        self.assertEqual(ev.payload["boundary_at_accepted_ms"], 8 * VAD_FRAME_MS)
        self.assertEqual(ev.payload["duration_real_ms"], 100)
        bus.shutdown()

    def test_boundary_fires_per_silence_run(self):
        # Two silence runs separated by enough voiced to re-admit.
        # 6 voiced + 20 silent + 6 voiced + 20 silent → 2 boundaries.
        seq = [True] * 6 + [False] * 20 + [True] * 6 + [False] * 20
        pp, _sink, bus = _build_preprocessor(seq)
        _feed(pp, len(seq))
        silence_events = [e for e in bus.events if e.topic == TOPIC_AUDIO_SILENCE]
        self.assertEqual(len(silence_events), 2)
        self.assertLess(
            silence_events[0].payload["boundary_at_accepted_ms"],
            silence_events[1].payload["boundary_at_accepted_ms"],
        )
        bus.shutdown()


class TestAcceptedMsClock(unittest.TestCase):
    def test_now_accepted_ms_advances_per_admit(self):
        pp, _sink, bus = _build_preprocessor([True] * 10)
        _feed(pp, 10)
        # 9 admits × 20ms = 180ms.
        self.assertEqual(pp.now_accepted_ms(), 180)
        bus.shutdown()

    def test_real_ms_processed_advances_per_chunk(self):
        pp, _sink, bus = _build_preprocessor([False] * 50)
        # Feed in two batches of 25 sub-frames (500ms each).
        _feed(pp, 25)
        self.assertEqual(pp._real_ms_processed, 500)
        _feed(pp, 25)
        self.assertEqual(pp._real_ms_processed, 1000)
        bus.shutdown()


class TestSegmentEmission(unittest.TestCase):
    def test_segment_carries_correct_accepted_ms_range(self):
        pp, sink, bus = _build_preprocessor([True] * 10)
        _feed(pp, 10)
        # One emit per fed chunk: all 9 admits coalesce into one segment.
        self.assertEqual(len(sink.segments), 1)
        seg = sink.segments[0]
        self.assertEqual(seg.start_accepted_ms, 0)
        self.assertEqual(seg.end_accepted_ms, 9 * VAD_FRAME_MS)
        self.assertEqual(len(seg.pcm), 9 * SUB_FRAME_BYTES)
        bus.shutdown()


class TestMuteGate(unittest.TestCase):
    def test_muted_chunks_drop_audio_freeze_accepted_advance_real(self):
        # 10 voiced frames while unmuted, then mute, then 10 more voiced
        # while muted, then unmute and 10 more voiced. Expected:
        #   - segments only from the unmuted spans
        #   - accepted_ms frozen across the muted span
        #   - real_ms_processed advances throughout
        pp, sink, bus = _build_preprocessor([True] * 30)
        _feed(pp, 10)
        accepted_before_mute = pp.now_accepted_ms()
        real_before_mute = pp._real_ms_processed
        self.assertGreater(accepted_before_mute, 0)

        pp.set_muted(True)
        _feed(pp, 10)
        self.assertEqual(pp.now_accepted_ms(), accepted_before_mute)
        self.assertEqual(
            pp._real_ms_processed, real_before_mute + 10 * VAD_FRAME_MS
        )

        pp.set_muted(False)
        _feed(pp, 10)
        self.assertGreater(pp.now_accepted_ms(), accepted_before_mute)
        # All PCM emitted must come from the two unmuted spans only.
        total_admitted_frames = sum(
            len(s.pcm) // SUB_FRAME_BYTES for s in sink.segments
        )
        # Pre-mute span: 9 admits (first frame drops by warmup).
        # Post-mute span: gate was reseeded all-silence; same warmup → 9 admits.
        self.assertEqual(total_admitted_frames, 18)
        bus.shutdown()

    def test_mute_toggle_action_drives_mute_state(self):
        # Real EventBus subscription path: publish mute_toggle and assert
        # the preprocessor flips _muted.
        from voice_dictation.event_bus import EventBus
        from voice_dictation.types import TOPIC_USER_ACTIONS
        from voice_dictation.clock import next_seq

        audio_cfg = AudioCaptureConfig()
        pp_cfg = PreprocessorConfig(
            voiced_gate_frames=5,
            voiced_gate_min_voiced=2,
            silence_boundary_ms=100,
        )
        bus = EventBus()
        stop = threading.Event()
        pp = AudioPreprocessor(pp_cfg, audio_cfg, bus, stop)
        try:
            pp.start()
            self.assertFalse(pp._muted)
            bus.publish(Event(
                topic=TOPIC_USER_ACTIONS,
                emit_accepted_ms=0,
                seq=next_seq(),
                payload={"action": "mute_toggle"},
            ))
            # Bus delivers on a worker thread — poll briefly.
            import time
            for _ in range(50):
                if pp._muted:
                    break
                time.sleep(0.01)
            self.assertTrue(pp._muted)
            bus.publish(Event(
                topic=TOPIC_USER_ACTIONS,
                emit_accepted_ms=0,
                seq=next_seq(),
                payload={"action": "mute_toggle"},
            ))
            for _ in range(50):
                if not pp._muted:
                    break
                time.sleep(0.01)
            self.assertFalse(pp._muted)
        finally:
            stop.set()
            pp.shutdown()
            bus.shutdown()


if __name__ == "__main__":
    unittest.main()
