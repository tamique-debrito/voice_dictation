"""Background worker that turns audio windows into text segments.

Wraps ``faster_whisper.WhisperModel``. The aggregator decides when to emit a
window (typically after a silence boundary or after enough audio has piled
up) and pushes it into ``in_queue``. This worker pulls windows, transcribes
them, and pushes ``Segment`` records into ``out_queue``.

Segments include a session-relative time range so the session writer can
later associate marker insertions with the right region of text.
"""

from __future__ import annotations

import os
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

from config import (
    FW_COMPUTE,
    FW_DEVICE,
    FW_LOG_PROB_THRESHOLD,
    FW_MODEL,
    FW_NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
)


@dataclass
class AudioWindow:
    """A contiguous span of int16 PCM samples with session-relative timing."""
    pcm: bytes
    start_time: float
    end_time: float


@dataclass
class Word:
    """A single transcribed word with session-relative timing."""
    text: str
    start_time: float
    end_time: float
    probability: float


@dataclass
class Segment:
    """Transcribed text covering ``start_time``..``end_time``."""
    text: str
    start_time: float
    end_time: float
    words: list["Word"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.words is None:
            self.words = []


class StreamingTranscriber:
    """faster-whisper-backed worker thread.

    Per-instance config:
      * ``beam_size`` — 1 for greedy (fast stream); 5+ for HQ.
      * ``condition_on_previous_text`` — when True, maintains a rolling
        prompt across sequential windows by feeding the previous window's
        decoded text to the next call as ``initial_prompt``. This is the
        cross-window context that ``small.en`` lacks today; the HQ stream
        in Phase 3 turns it on. ``faster_whisper``'s own
        ``condition_on_previous_text`` flag only conditions within a
        single ``transcribe()`` call across its internally-emitted
        segments — the rolling prompt covers what happens BETWEEN calls.
      * ``initial_prompt`` — optional seed text injected before the first
        window. Useful for biasing toward a domain vocabulary.
      * ``label`` — short name used in log lines (e.g. "fast", "hq").
    """

    def __init__(
        self,
        in_queue: "queue.Queue[Optional[AudioWindow]]",
        out_queue: "queue.Queue[Segment]",
        model_name: str = FW_MODEL,
        device: str = FW_DEVICE,
        compute_type: str = FW_COMPUTE,
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        initial_prompt: str = "",
        label: str = "stream",
        rolling_prompt_max_chars: int = 1000,
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
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.label = label
        self._rolling_prompt_max_chars = rolling_prompt_max_chars
        self._rolling_prompt = initial_prompt
        self._model: Optional[WhisperModel] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def model_label(self) -> str:
        return f"faster-whisper:{self.model_name}"

    def reset_prompt(self) -> None:
        """Clear the rolling prompt — used on model hot-swap so a new
        tokenizer doesn't see stale text from the previous model."""
        self._rolling_prompt = ""

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type,
            )
        except Exception as e:
            msg = str(e)
            cache_miss = (
                e.__class__.__name__ in (
                    "OfflineModeIsEnabled", "LocalEntryNotFoundError",
                )
                or "OfflineModeIsEnabled" in msg
                or "LocalEntryNotFoundError" in msg
                or "outgoing traffic has been disabled" in msg
            )
            if cache_miss:
                raise RuntimeError(
                    f"faster-whisper can't load model '{self.model_name}' "
                    f"for the '{self.label}' stream: it isn't in the local "
                    f"HuggingFace cache, and HF cache lookup is offline. "
                    f"To fix:\n"
                    f"  • Set \"persistent.hf_hub_offline\": false in "
                    f"local_config.json and restart (one-time download "
                    f"will run; you can flip it back to true afterward), "
                    f"OR\n"
                    f"  • Run once with the --check-updates CLI flag, OR\n"
                    f"  • Pre-download via "
                    f"`huggingface-cli download Systran/faster-whisper-"
                    f"{self.model_name}`.\n"
                    f"Original error: {e}"
                ) from e
            raise
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
        # no_speech_threshold + log_prob_threshold tighten faster-whisper's
        # own rejection of low-confidence decodes, suppressing hallucinations
        # like "okay okay okay" / "thank you thank you" on near-silent audio.
        # compression_ratio_threshold stays at the default 2.4 (catches
        # degenerate repetition loops).
        initial_prompt = (
            self._rolling_prompt if self.condition_on_previous_text else None
        )
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            no_speech_threshold=FW_NO_SPEECH_THRESHOLD,
            log_prob_threshold=FW_LOG_PROB_THRESHOLD,
            word_timestamps=True,
        )
        window_duration = win.end_time - win.start_time
        window_end_abs = win.start_time + window_duration
        decoded_pieces: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            decoded_pieces.append(text)
            # Map per-segment relative timing into session-relative timing.
            # faster-whisper returns times relative to the audio it received.
            t0 = win.start_time + float(seg.start)
            t1 = min(win.start_time + float(seg.end), window_end_abs)
            words: list[Word] = []
            for w in (getattr(seg, "words", None) or []):
                w_start = win.start_time + float(w.start)
                w_end = min(win.start_time + float(w.end), window_end_abs)
                words.append(Word(
                    text=w.word,
                    start_time=w_start,
                    end_time=w_end,
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                ))
            self.out_queue.put(Segment(
                text=text, start_time=t0, end_time=t1, words=words
            ))
        if self.condition_on_previous_text and decoded_pieces:
            # Update rolling prompt: keep the previous prompt's tail plus
            # this window's decoded text, then truncate to the configured
            # char budget so the prompt doesn't grow unbounded.
            combined = (self._rolling_prompt + " " + " ".join(decoded_pieces)).strip()
            if len(combined) > self._rolling_prompt_max_chars:
                combined = combined[-self._rolling_prompt_max_chars:]
            self._rolling_prompt = combined

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=10.0)
