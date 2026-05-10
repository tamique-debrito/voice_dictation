"""Continuous PyAudio capture for the persistent streaming mode.

Pushes raw int16 PCM frames into a queue. Audio is never written to disk —
the queue feeds the VAD/transcriber and frames are discarded after use.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import pyaudio

from config import CHANNELS, CHUNK_SIZE, SAMPLE_RATE


class PersistentRecorder:
    """Always-on microphone capture.

    Frames are tagged with a monotonic timestamp at read time so downstream
    consumers don't have to guess wall-clock offsets when the transcriber
    falls behind.
    """

    def __init__(self, out_queue: "queue.Queue[tuple[float, bytes]]"):
        self.out_queue = out_queue
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
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        self.started_monotonic = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="PersistentRecorder", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
            except Exception as e:
                print(f"[PersistentRecorder] read error: {e}")
                break
            ts = time.monotonic() - (self.started_monotonic or 0.0)
            try:
                self.out_queue.put_nowait((ts, data))
            except queue.Full:
                # Backpressure: drop oldest by draining one slot. The
                # transcriber is the bottleneck; better to lose a tiny
                # window than to block PyAudio's read loop.
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
