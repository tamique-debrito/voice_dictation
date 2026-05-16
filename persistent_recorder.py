"""Continuous PyAudio capture for v2.

Pushes raw int16 PCM frames into a queue, tagged with monotonic-since-start
timestamps so downstream consumers can correlate wall-clock without
guessing. Ported from voice_dictation/persistent_recorder.py with v2's
config dataclass instead of the legacy ``config.py`` module-level constants.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import pyaudio

from .runtime_config import AudioCaptureConfig


logger = logging.getLogger(__name__)


class PersistentRecorder:
    def __init__(
        self,
        out_queue: "queue.Queue[tuple[float, bytes]]",
        cfg: AudioCaptureConfig,
    ) -> None:
        self.out_queue = out_queue
        self._cfg = cfg
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.started_monotonic: Optional[float] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=self._cfg.channels,
            rate=self._cfg.sample_rate,
            input=True,
            frames_per_buffer=self._cfg.chunk_size,
        )
        self.started_monotonic = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="PersistentRecorder", daemon=True,
        )
        self._thread.start()
        logger.info(
            "PersistentRecorder started (rate=%d, chunk=%d)",
            self._cfg.sample_rate, self._cfg.chunk_size,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._stream.read(self._cfg.chunk_size, exception_on_overflow=False)
            except Exception:
                logger.exception("PyAudio read failed; exiting recorder loop")
                break
            ts = time.monotonic() - (self.started_monotonic or 0.0)
            try:
                self.out_queue.put_nowait((ts, data))
            except queue.Full:
                # Drop oldest to keep PyAudio's read loop unblocked.
                try:
                    self.out_queue.get_nowait()
                    self.out_queue.put_nowait((ts, data))
                except queue.Empty:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None
        logger.info("PersistentRecorder stopped")
