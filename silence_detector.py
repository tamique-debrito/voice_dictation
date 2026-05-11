"""Silence-boundary detection on top of webrtcvad.

webrtcvad operates on exact 10/20/30 ms frames at 8/16/32/48 kHz. We feed it
20 ms frames at SAMPLE_RATE (16 kHz = 320 samples = 640 bytes/frame) and
maintain a rolling window of voice/silence decisions.

The detector emits a "silence boundary" whenever ``silence_ms`` of continuous
non-voice frames has elapsed since the last voiced frame. Boundaries are
edge-triggered: one fires per silence run, after which the detector waits
for voice to resume before arming again.
"""

from __future__ import annotations

from typing import Iterator, Optional

import webrtcvad

from config import SAMPLE_RATE


VAD_FRAME_MS = 20  # 10/20/30 are the only legal values
BYTES_PER_SAMPLE = 2  # int16
SAMPLES_PER_FRAME = SAMPLE_RATE * VAD_FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE


class SilenceDetector:
    """Stream-fed VAD that emits boundary events.

    Usage::

        det = SilenceDetector(silence_ms=500, aggressiveness=2)
        for evt in det.feed(pcm_bytes):
            if evt == "boundary":
                ...
    """

    def __init__(self, silence_ms: int = 500, aggressiveness: int = 2):
        # webrtcvad aggressiveness: 0 (least aggressive) .. 3 (most aggressive).
        # 2 is a reasonable default for indoor dictation.
        self._vad = webrtcvad.Vad(aggressiveness)
        self._silence_frames_target = max(1, silence_ms // VAD_FRAME_MS)
        self._buf = bytearray()
        self._silent_run = 0
        self._armed = False  # only fire a boundary after at least one voiced frame
        # Per-window frame counters. Reset by the caller via reset_counts()
        # after consuming each window so it can gate transcription on the
        # fraction/duration of voiced audio in that window.
        self.voiced_frames = 0
        self.total_frames = 0

    def reset_counts(self) -> None:
        self.voiced_frames = 0
        self.total_frames = 0

    @staticmethod
    def voiced_ms(voiced_frames: int) -> int:
        return voiced_frames * VAD_FRAME_MS

    def feed(self, pcm: bytes) -> Iterator[str]:
        """Feed raw int16 PCM bytes; yield 'boundary' on each silence edge."""
        self._buf.extend(pcm)
        while len(self._buf) >= BYTES_PER_FRAME:
            frame = bytes(self._buf[:BYTES_PER_FRAME])
            del self._buf[:BYTES_PER_FRAME]
            try:
                voiced = self._vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                voiced = False
            self.total_frames += 1
            if voiced:
                self.voiced_frames += 1
                self._silent_run = 0
                self._armed = True
            else:
                self._silent_run += 1
                if self._armed and self._silent_run == self._silence_frames_target:
                    self._armed = False
                    yield "boundary"
