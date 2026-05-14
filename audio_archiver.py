"""AudioArchiver: optional per-chunk WAV persistence aligned to canonical chunks.

When ``persistent.save_audio`` is true, ``PersistentApp`` subscribes an
``AudioArchiver`` to ``AudioFanout`` and registers ``write_chunk`` as the
canonical-flush callback on ``TranscriptTimeline``. The archiver keeps the
most recent N seconds of timestamped PCM frames in memory; when the
canonical writer flushes ``chunk_NNN.txt`` for monotonic time range
``[t_start, t_end]``, the archiver slices the matching frames out of its
ring buffer and writes ``audio/chunk_NNN.wav`` plus a row to
``audio/manifest.jsonl``.

Frame timestamps are session-relative monotonic seconds (from
``PersistentRecorder``), matching the coordinates used by segment
``start_time``/``end_time`` everywhere else. So the slice is just
``[f for f in frames if t_start <= f.ts <= t_end]`` (with a little
boundary slack to cover frame quantization).

The buffer is bounded (default 120s) so memory stays flat regardless of
session length. Chunks typically flush every ~10-30s, so 120s is plenty
of headroom for transcriber lag.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import wave
from collections import deque
from typing import Optional


# Hard safety ceiling for how far back the buffer can hold frames if no
# canonical chunk flushes ever fire (e.g. the user mumbles for an hour
# without crossing the token-target threshold). Under normal operation
# the chunk-boundary trim (see _trim_locked) keeps the buffer to roughly
# one chunk's worth of audio.
_DEFAULT_BUFFER_SECONDS = 1800.0


class AudioArchiver:
    def __init__(
        self,
        audio_q: "queue.Queue",
        stop_event: threading.Event,
        sample_rate: int = 16000,
        buffer_seconds: float = _DEFAULT_BUFFER_SECONDS,
    ):
        self._q = audio_q
        self._stop = stop_event
        self._sample_rate = sample_rate
        self._buffer_seconds = buffer_seconds

        # Frames are stored as (ts, bytes) tuples; ts is session-monotonic.
        self._frames: deque[tuple[float, bytes]] = deque()
        # The audio-time end of the most recently archived canonical chunk.
        # Frames with ts < this can be discarded — they've already been
        # written out and won't belong to any later chunk. Updated by
        # write_chunk() after each successful slice.
        self._chunk_boundary_t: float = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="AudioArchiver", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            # MUTE_RESET is a tuple sentinel ("__mute_reset__",) — ignore.
            if not (isinstance(item, tuple) and len(item) == 2
                    and isinstance(item[0], (int, float))
                    and isinstance(item[1], (bytes, bytearray))):
                continue
            ts, data = item
            with self._lock:
                self._frames.append((float(ts), bytes(data)))
                self._trim_locked(float(ts))

    def _trim_locked(self, now_ts: float) -> None:
        # Two cutoffs, take the LATER (= aggressive trim):
        #   - chunk_boundary_t: anything older than the last archived chunk's
        #     end is already on disk and can never be needed again.
        #   - now_ts - buffer_seconds: safety ceiling for the case where
        #     chunks never flush (default 30 min — a session ought to
        #     produce a canonical chunk well before then).
        cutoff = max(now_ts - self._buffer_seconds, self._chunk_boundary_t)
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    # ------------------------------------------------------------------
    # Write a chunk WAV + manifest row
    # ------------------------------------------------------------------

    # Maximum gap between consecutive frames' timestamps before we treat
    # them as belonging to different ``audio_segments`` rather than the
    # same continuous span. 0.25 s matches typical sub-VAD jitter: real
    # speech-on/speech-off pauses come in 0.5 s+ blocks via the silence
    # detector, and consecutive captured frames are ~64 ms apart, so any
    # gap above 0.25 s is almost certainly a dropout or mute interval
    # worth marking as a discontinuity in the wav.
    _SEGMENT_GAP_THRESHOLD = 0.25

    def write_chunk(
        self,
        idx: int,
        t_start: float,
        t_end: float,
        session_dir: str,
        canonical_text: str = "",
    ) -> Optional[str]:
        """Write the wav for the canonical chunk covering ``[t_start, t_end]``.

        The wav contains **only real captured audio** — no silence padding.
        Frames that arrived contiguously (gap < 0.25 s) are concatenated
        into one ``audio_segment``; longer gaps (mute periods, dropouts)
        produce separate segments. The manifest row carries an
        ``audio_segments`` list with the wav-time offset and the
        timeline-time span for each segment, so playback can map
        ``now_seconds`` → ``audio.currentTime`` per-segment and pause in
        gaps.

        Returns the wav path on success, or ``None`` if no frames at all
        fell inside the window.
        """
        if t_end <= t_start:
            return None
        audio_dir = os.path.join(session_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        sr = self._sample_rate
        bytes_per_sample = 2  # int16 mono

        # Small slack on the window edges so a frame straddling the
        # boundary is still kept (any out-of-window bytes are clipped
        # before writing).
        slack = 0.05
        lo = t_start - slack
        hi = t_end + slack

        # Pass 1: walk frames in ts order, group contiguous runs into
        # segments. Each frame's actual audio coverage is [ts, ts+frame_dur).
        with self._lock:
            in_window = [(ts, bytes(data)) for (ts, data) in self._frames
                         if lo <= ts <= hi]
        if not in_window:
            return None

        # Estimate per-frame duration from data length. Each int16 mono
        # sample is 2 bytes, and the recorder produces fixed-size frames.
        # Using the actual frame's bytes is robust to config changes.
        def frame_dur(data: bytes) -> float:
            return (len(data) // bytes_per_sample) / sr

        # Group into segments by gap-threshold.
        segments_raw: list[list[tuple[float, bytes]]] = [[]]
        for (ts, data) in in_window:
            seg = segments_raw[-1]
            if seg:
                prev_ts, prev_data = seg[-1]
                prev_end = prev_ts + frame_dur(prev_data)
                if ts - prev_end > self._SEGMENT_GAP_THRESHOLD:
                    segments_raw.append([(ts, data)])
                    continue
            seg.append((ts, data))

        # Pass 2: serialize segments into the wav body + build the
        # manifest's ``audio_segments`` list. wav_offset_s is the
        # cumulative duration of all preceding segments.
        wav_body = bytearray()
        audio_segments: list[dict] = []
        for seg in segments_raw:
            if not seg:
                continue
            seg_t_start = seg[0][0]
            seg_t_end = seg[-1][0] + frame_dur(seg[-1][1])
            # Clamp to the chunk's [t_start, t_end] for accurate timeline
            # alignment — the slack frames at the edges shouldn't claim
            # to cover time they don't actually represent in the chunk.
            seg_t_start_clamped = max(seg_t_start, t_start)
            seg_t_end_clamped = min(seg_t_end, t_end)
            if seg_t_end_clamped <= seg_t_start_clamped:
                continue
            wav_offset_s = len(wav_body) / (sr * bytes_per_sample)
            for (_ts, data) in seg:
                wav_body.extend(data)
            audio_segments.append({
                "wav_offset_s": round(wav_offset_s, 3),
                "t_start": round(float(seg_t_start_clamped), 3),
                "t_end": round(float(seg_t_end_clamped), 3),
                "duration_s": round(
                    (len(wav_body) / (sr * bytes_per_sample)) - wav_offset_s,
                    3,
                ),
            })

        if not wav_body:
            return None

        wav_path = os.path.join(audio_dir, f"chunk_{idx:03d}.wav")
        tmp_path = wav_path + ".tmp"
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(bytes_per_sample)
            wf.setframerate(sr)
            wf.writeframes(bytes(wav_body))
        os.replace(tmp_path, wav_path)

        # Advance the trim boundary so frames belonging to this chunk can
        # be released on the next ingest. Frames at ts == t_end belong to
        # the NEXT chunk (per the canonical writer's word boundaries) so
        # we can safely discard everything strictly before t_end.
        with self._lock:
            if t_end > self._chunk_boundary_t:
                self._chunk_boundary_t = float(t_end)

        duration_s = len(wav_body) / (sr * bytes_per_sample)
        manifest_row = {
            "idx": idx,
            "audio_path": os.path.relpath(wav_path, session_dir),
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "duration_s": round(duration_s, 3),
            "audio_segments": audio_segments,
            "canonical_text": canonical_text,
            "frames_placed": sum(len(s) for s in segments_raw),
            "written_at": round(time.time(), 3),
        }
        manifest_path = os.path.join(audio_dir, "manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

        return wav_path
