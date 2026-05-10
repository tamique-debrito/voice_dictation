"""Background worker that turns audio windows into text segments.

Wraps ``faster_whisper.WhisperModel``. The aggregator decides when to emit a
window (typically after a silence boundary or after enough audio has piled
up) and pushes it into ``in_queue``. This worker pulls windows, transcribes
them, and pushes ``Segment`` records into ``out_queue``.

Segments include a session-relative time range so the session writer can
later associate marker insertions with the right region of text.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:  # pragma: no cover - runtime dependency
    WhisperModel = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from config import FW_COMPUTE, FW_DEVICE, FW_MODEL, SAMPLE_RATE


@dataclass
class AudioWindow:
    """A contiguous span of int16 PCM samples with session-relative timing."""
    pcm: bytes
    start_time: float
    end_time: float


@dataclass
class Segment:
    """Transcribed text covering ``start_time``..``end_time``."""
    text: str
    start_time: float
    end_time: float


class StreamingTranscriber:
    """faster-whisper-backed worker thread."""

    def __init__(
        self,
        in_queue: "queue.Queue[Optional[AudioWindow]]",
        out_queue: "queue.Queue[Segment]",
        model_name: str = FW_MODEL,
        device: str = FW_DEVICE,
        compute_type: str = FW_COMPUTE,
    ):
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run `pip install -r voice_dictation/requirements.txt`."
            ) from _IMPORT_ERROR
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def model_label(self) -> str:
        return f"faster-whisper:{self.model_name}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )
        self._thread = threading.Thread(target=self._loop, name="StreamingTranscriber",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                # Sentinel: drain and exit.
                break
            try:
                self._transcribe_window(item)
            except Exception as e:
                print(f"[StreamingTranscriber] transcription error: {e}")

    def _transcribe_window(self, win: AudioWindow) -> None:
        # int16 -> float32 in [-1, 1]
        audio = np.frombuffer(win.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return
        # vad_filter=False — we already gate on silence at the aggregator
        # level, and keeping faster-whisper's own VAD off avoids it dropping
        # short utterances.
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        window_duration = win.end_time - win.start_time
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            # Map per-segment relative timing into session-relative timing.
            # faster-whisper returns times relative to the audio it received.
            t0 = win.start_time + float(seg.start)
            t1 = win.start_time + float(seg.end)
            # Clamp in case of any drift past the window edge.
            t1 = min(t1, win.start_time + window_duration)
            self.out_queue.put(Segment(text=text, start_time=t0, end_time=t1))

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=10.0)
