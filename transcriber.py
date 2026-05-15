"""Transcriber — runs faster-whisper on each AudioWindow.

One instance per stream (fast / hq). Same class — only ``StreamConfig``
differs (model, beam_size, condition_on_previous_text, etc.).

Inputs:
  - Direct queue of ``AudioWindow`` from the AudioWindower.

Outputs:
  - Direct queue of ``Segment`` to TranscriptStreamManager (ordered delivery).
  - Bus publication of the same segments on ``segments.<label>`` (fan-out
    to UI + persister + replay).

Segments carry ``ends_on_marker_idx`` from the source window so the
TranscriptStreamManager can complete a paste window as soon as it sees the
matching marker-annotated segment, without waiting for a later segment.

Adapted from voice_dictation/streaming_transcriber.py:181-241.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Optional

import numpy as np

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .runtime_config import AudioCaptureConfig, StreamConfig
from .types import (
    AudioWindow, Event, Segment, Word, WindowCompletion,
    TOPIC_SEGMENTS_FAST, TOPIC_SEGMENTS_HQ,
    TOPIC_PROGRESS_FAST, TOPIC_PROGRESS_HQ,
)


logger = logging.getLogger(__name__)


try:
    from faster_whisper import WhisperModel  # type: ignore
    _FW_IMPORT_ERROR: Optional[BaseException] = None
except Exception as e:  # pragma: no cover
    WhisperModel = None
    _FW_IMPORT_ERROR = e


def _topic_for(label: str) -> str:
    if label == "fast":
        return TOPIC_SEGMENTS_FAST
    if label == "hq":
        return TOPIC_SEGMENTS_HQ
    return f"segments.{label}"


def _progress_topic_for(label: str) -> str:
    if label == "fast":
        return TOPIC_PROGRESS_FAST
    if label == "hq":
        return TOPIC_PROGRESS_HQ
    return f"progress.{label}"


class Transcriber:
    def __init__(
        self,
        config: StreamConfig,
        audio_cfg: AudioCaptureConfig,
        bus: EventBus,
        clock: AcceptedClock,
        in_q: queue.Queue[AudioWindow],
        stop_event: threading.Event,
    ) -> None:
        self._cfg = config
        self._audio = audio_cfg
        self._bus = bus
        self._clock = clock
        self._in_q = in_q
        self._stop = stop_event
        self.label = config.label

        # Direct ordered-delivery queue for TranscriptStreamManager.
        self.segment_q: queue.Queue[Segment] = queue.Queue()

        self._model: Optional[WhisperModel] = None
        self._rolling_prompt = ""
        self._rolling_prompt_max_chars = 800
        self._thread: Optional[threading.Thread] = None

        self.windows_processed = 0
        self.inference_seconds = 0.0

        logger.info(
            "Transcriber(%s) initialized (model=%s, beam=%d, prompt=%s)",
            self.label, config.fw.model, config.fw.beam_size,
            config.fw.condition_on_previous_text,
        )

    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "label": self.label,
            "model": self._cfg.fw.model,
            "model_loaded": self._model is not None,
            "windows_processed": self.windows_processed,
            "inference_seconds": round(self.inference_seconds, 2),
            "in_q_size": self._in_q.qsize(),
            "in_q_maxsize": self._in_q.maxsize,
            "segment_q_size": self.segment_q.qsize(),
            "rolling_prompt_chars": len(self._rolling_prompt),
        }

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if WhisperModel is None:
            logger.error(
                "Transcriber(%s): faster_whisper not importable (%s)",
                self.label, _FW_IMPORT_ERROR,
            )
            return False
        if self._cfg.fw.device == "" or self._cfg.fw.device == "cpu":
            device = "cpu"
        else:
            device = self._cfg.fw.device
        compute = self._cfg.fw.compute or "int8"
        t0 = time.monotonic()
        # HF_HUB_OFFLINE is honored to skip the hub-version check on cold
        # boot when the model is already cached.
        logger.info(
            "Transcriber(%s) loading model=%s device=%s compute=%s",
            self.label, self._cfg.fw.model, device, compute,
        )
        try:
            self._model = WhisperModel(
                self._cfg.fw.model, device=device, compute_type=compute,
            )
        except Exception:
            logger.exception("Transcriber(%s) model load failed", self.label)
            return False
        logger.info(
            "Transcriber(%s) model loaded in %.2fs",
            self.label, time.monotonic() - t0,
        )
        return True

    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name=f"Transcriber-{self.label}", daemon=True,
        )
        self._thread.start()
        logger.info("Transcriber(%s).start", self.label)

    def shutdown(self) -> None:
        if self._thread is None:
            return
        # Push a None sentinel to wake the get().
        try:
            self._in_q.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)
        self._thread = None
        logger.info(
            "Transcriber(%s).shutdown (windows=%d, infer_total=%.2fs)",
            self.label, self.windows_processed, self.inference_seconds,
        )

    def _loop(self) -> None:
        # Load model lazily so processes that never get a window (selftest,
        # idle session) don't pay model-load cost or fail offline.
        while True:
            try:
                item = self._in_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            # Tombstones from the windower (dropped due to queue-full)
            # pass straight through so the downstream watermark advances.
            if isinstance(item, WindowCompletion):
                self.segment_q.put(item)
                self._bus.publish(Event(
                    topic=_progress_topic_for(self.label),
                    emit_accepted_ms=self._clock.now_accepted_ms(),
                    seq=next_seq(),
                    payload={
                        "window_id": item.window_id,
                        "transcribed_up_to_accepted_ms": item.end_accepted_ms,
                        "dropped": True,
                    },
                ))
                continue
            if self._model is None and not self._ensure_model():
                return
            try:
                self._transcribe(item)
            except Exception:
                logger.exception("Transcriber(%s) failed on window", self.label)

    # ------------------------------------------------------------------

    def _emit_completion(self, win: AudioWindow, dropped: bool) -> None:
        """Push a WindowCompletion sentinel + publish a progress event.

        The manager's segment-drain loop consumes the sentinel atomically
        AFTER all segments for this window have been added to its buffer,
        so it can safely advance ``transcribed_up_to_accepted_ms`` to
        ``win.end_accepted_ms`` without ordering races."""
        self.segment_q.put(WindowCompletion(
            stream_label=self.label,
            window_id=win.window_id,
            end_accepted_ms=win.end_accepted_ms,
            dropped=dropped,
        ))
        self._bus.publish(Event(
            topic=_progress_topic_for(self.label),
            emit_accepted_ms=self._clock.now_accepted_ms(),
            seq=next_seq(),
            payload={
                "window_id": win.window_id,
                "transcribed_up_to_accepted_ms": win.end_accepted_ms,
                "dropped": dropped,
            },
        ))

    def _transcribe(self, win: AudioWindow) -> None:
        audio = np.frombuffer(win.pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            self._emit_completion(win, dropped=True)
            return
        rms = float(np.sqrt(np.mean(audio * audio)))
        if rms < self._cfg.min_rms:
            # Sub-speech window — skip. Whisper would hallucinate AND burn
            # 4-9s of CPU. webrtcvad's "voiced" threshold is much looser
            # than what Whisper wants. Still emit completion so the manager
            # advances its watermark past this window.
            logger.info(
                "Transcriber(%s) skipping window %d: rms=%.4f < min_rms=%.4f",
                self.label, win.window_id, rms, self._cfg.min_rms,
            )
            self._emit_completion(win, dropped=True)
            return
        if not np.isfinite(audio).all():
            logger.warning(
                "Transcriber(%s) skipping window %d: non-finite audio",
                self.label, win.window_id,
            )
            self._emit_completion(win, dropped=True)
            return
        logger.debug(
            "Transcriber(%s) win=%d dur_ms=%d rms=%.4f",
            self.label, win.window_id,
            win.end_accepted_ms - win.start_accepted_ms, rms,
        )
        initial_prompt = (
            self._rolling_prompt if self._cfg.fw.condition_on_previous_text else None
        )
        t0 = time.monotonic()
        # vad_filter=False — the preprocessor already stripped silence.
        segments_iter, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self._cfg.fw.beam_size,
            vad_filter=False,
            condition_on_previous_text=self._cfg.fw.condition_on_previous_text,
            initial_prompt=initial_prompt or None,
            no_speech_threshold=self._cfg.fw.no_speech_threshold,
            log_prob_threshold=self._cfg.fw.log_prob_threshold,
            word_timestamps=True,
        )

        decoded_texts: list[str] = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            decoded_texts.append(text)
            t_start_ms = win.start_accepted_ms + int(seg.start * 1000)
            t_end_ms = min(
                win.start_accepted_ms + int(seg.end * 1000),
                win.end_accepted_ms,
            )
            words: list[Word] = []
            for w in (getattr(seg, "words", None) or []):
                words.append(Word(
                    text=w.word,
                    start_accepted_ms=win.start_accepted_ms + int(w.start * 1000),
                    end_accepted_ms=min(
                        win.start_accepted_ms + int(w.end * 1000),
                        win.end_accepted_ms,
                    ),
                    probability=float(getattr(w, "probability", 0.0) or 0.0),
                ))
            segment = Segment(
                text=text,
                content_accepted_ms_start=t_start_ms,
                content_accepted_ms_end=t_end_ms,
                words=words,
                window_id=win.window_id,
                ends_on_marker_idx=win.ends_on_marker_idx,
            )
            self.segment_q.put(segment)
            self._bus.publish(Event(
                topic=_topic_for(self.label),
                emit_accepted_ms=self._clock.now_accepted_ms(),
                seq=next_seq(),
                payload={
                    "window_id": segment.window_id,
                    "text": segment.text,
                    "content_accepted_ms_start": segment.content_accepted_ms_start,
                    "content_accepted_ms_end": segment.content_accepted_ms_end,
                    "ends_on_marker_idx": segment.ends_on_marker_idx,
                    "words": [
                        {
                            "text": w.text,
                            "start_accepted_ms": w.start_accepted_ms,
                            "end_accepted_ms": w.end_accepted_ms,
                            "probability": w.probability,
                        } for w in segment.words
                    ],
                },
            ))

        self.windows_processed += 1
        elapsed = time.monotonic() - t0
        self.inference_seconds += elapsed
        logger.debug(
            "Transcriber(%s) win=%d done elapsed=%.2fs decoded=%d",
            self.label, win.window_id, elapsed, len(decoded_texts),
        )
        # Atomic watermark advance — all segments for this window are now
        # in segment_q ahead of the completion sentinel.
        self._emit_completion(win, dropped=False)

        if self._cfg.fw.condition_on_previous_text and decoded_texts:
            combined = (self._rolling_prompt + " " + " ".join(decoded_texts)).strip()
            if len(combined) > self._rolling_prompt_max_chars:
                combined = combined[-self._rolling_prompt_max_chars:]
            self._rolling_prompt = combined
