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


_DEFAULT_BUFFER_SECONDS = 120.0


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
        cutoff = now_ts - self._buffer_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    # ------------------------------------------------------------------
    # Write a chunk WAV + manifest row
    # ------------------------------------------------------------------

    def write_chunk(
        self,
        idx: int,
        t_start: float,
        t_end: float,
        session_dir: str,
        canonical_text: str = "",
    ) -> Optional[str]:
        """Slice buffered PCM in ``[t_start, t_end]`` and write a wav.

        Returns the wav path on success, ``None`` if no frames cover the
        requested window (e.g. the chunk's audio aged out of the buffer
        because the transcriber lagged longer than ``buffer_seconds``).
        """
        if t_end <= t_start:
            return None
        audio_dir = os.path.join(session_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        # Slight slack on both ends: frames are ~tens of ms each, and the
        # canonical chunk's first-word-start / last-word-end may land
        # inside a frame rather than at its boundary. 50 ms is generous
        # without leaking into adjacent chunks (silence boundary is
        # >= 500 ms by default).
        slack = 0.05
        lo = t_start - slack
        hi = t_end + slack

        with self._lock:
            payload = b"".join(
                data for (ts, data) in self._frames
                if lo <= ts <= hi
            )

        if not payload:
            return None

        wav_path = os.path.join(audio_dir, f"chunk_{idx:03d}.wav")
        tmp_path = wav_path + ".tmp"
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self._sample_rate)
            wf.writeframes(payload)
        os.replace(tmp_path, wav_path)

        duration_s = len(payload) / (self._sample_rate * 2)  # 2 bytes/sample
        manifest_row = {
            "idx": idx,
            "audio_path": os.path.relpath(wav_path, session_dir),
            "t_start": round(float(t_start), 3),
            "t_end": round(float(t_end), 3),
            "duration_s": round(duration_s, 3),
            "canonical_text": canonical_text,
            "written_at": round(time.time(), 3),
        }
        manifest_path = os.path.join(audio_dir, "manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

        return wav_path
