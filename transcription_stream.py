"""TranscriptionStream: aggregator + transcriber, segments out via callback.

Previously this class owned a ``SessionWriter`` and fed segments into it.
After the TranscriptTimeline refactor, segments are emitted via an
``on_segment(stream_id, Segment)`` callback so the timeline can ingest
them into the unified canonical render.

Each instance owns:
  * The aggregator thread (PCM → AudioWindow at VAD silence boundaries).
  * The StreamingTranscriber worker thread.
  * The drain thread (Segment → callback).

No writer state lives here. Chunk file flushing is timeline-driven; the
fast aggregator notifies the timeline on silence boundaries via the
``on_silence_boundary`` callback.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from audio_fanout import MUTE_RESET
from config import SAMPLE_RATE
from silence_detector import SilenceDetector
from streaming_transcriber import AudioWindow, Segment, StreamingTranscriber


class TranscriptionStream:
    def __init__(
        self,
        *,
        label: str,
        audio_input_q: "queue.Queue",
        stop_event: threading.Event,
        model_name: str,
        device: str,
        compute_type: str,
        max_window_seconds: float,
        silence_ms: int,
        vad_aggressiveness: int,
        min_voiced_ms: int,
        min_voiced_frac: float,
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        initial_prompt: str = "",
        window_q_maxsize: int = 8,
        sample_rate: int = SAMPLE_RATE,
        chunk_flush_periodic_seconds: float = 5.0,
        debug_log: Optional[Callable[[str, dict], None]] = None,
        on_segment: Optional[Callable[[str, Segment], None]] = None,
        on_silence_boundary: Optional[Callable[[], None]] = None,
    ):
        self.label = label
        self._audio_input_q = audio_input_q
        self._app_stop = stop_event
        self._local_stop = threading.Event()
        self._max_window_seconds = max_window_seconds
        self._silence_ms = silence_ms
        self._vad_aggressiveness = vad_aggressiveness
        self._min_voiced_ms = min_voiced_ms
        self._min_voiced_frac = min_voiced_frac
        self._sample_rate = sample_rate
        self._chunk_flush_periodic_seconds = chunk_flush_periodic_seconds
        self._debug = debug_log or (lambda kind, data: None)
        self._on_segment = on_segment
        self._on_silence_boundary = on_silence_boundary

        self.window_q: queue.Queue = queue.Queue(maxsize=window_q_maxsize)
        self.segment_q: queue.Queue = queue.Queue()

        self.transcriber = StreamingTranscriber(
            in_queue=self.window_q,
            out_queue=self.segment_q,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            label=label,
        )

        self._force_flush = threading.Event()
        self._aggregator_thread: Optional[threading.Thread] = None
        self._drain_thread: Optional[threading.Thread] = None

        # Running totals of audio windows dropped at the aggregator stage.
        # Surfaced via the status payload + widget header so the user can
        # see at a glance if the stream is losing data.
        self.dropped_queue_full_count: int = 0
        self.dropped_silent_count: int = 0
        self.windows_accepted_count: int = 0

    # ------------------------------------------------------------------

    @property
    def model_label(self) -> str:
        return self.transcriber.model_label

    def request_force_flush(self) -> None:
        self._force_flush.set()

    def start(self) -> None:
        self.transcriber.start()
        self._aggregator_thread = threading.Thread(
            target=self._aggregator_loop,
            name=f"Aggregator-{self.label}",
            daemon=True,
        )
        self._aggregator_thread.start()
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name=f"Drain-{self.label}",
            daemon=True,
        )
        self._drain_thread.start()

    def stop(self) -> None:
        self.transcriber.stop()

    def teardown(self, timeout: float = 5.0) -> None:
        """Stop just this stream (for live HQ disable). Leaves siblings running."""
        import gc as _gc
        self._local_stop.set()
        try:
            self.transcriber.stop()
        except Exception:
            pass
        if self._aggregator_thread is not None:
            self._aggregator_thread.join(timeout=timeout)
            self._aggregator_thread = None
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=timeout)
            self._drain_thread = None
        try:
            self.transcriber._model = None
        except Exception:
            pass
        _gc.collect()

    def swap_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        beam_size: int,
        condition_on_previous_text: bool,
    ) -> None:
        import gc as _gc
        old = self.transcriber
        old.stop()
        try:
            old._model = None
        except Exception:
            pass
        _gc.collect()
        self.transcriber = StreamingTranscriber(
            in_queue=self.window_q,
            out_queue=self.segment_q,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            label=self.label,
        )
        self.transcriber.start()

    def drain_remaining_segments(self) -> int:
        """Pull pending segments from segment_q after stop() and pass them
        through ``on_segment`` so nothing in-flight is lost."""
        count = 0
        while True:
            try:
                seg: Segment = self.segment_q.get_nowait()
            except queue.Empty:
                break
            self._emit_segment(seg)
            count += 1
        return count

    # ------------------------------------------------------------------

    def _emit_segment(self, seg: Segment) -> None:
        self._debug("segment", {
            "stream": self.label,
            "text": seg.text,
            "start": round(seg.start_time, 3),
            "end": round(seg.end_time, 3),
            "words": [
                {"text": w.text, "s": round(w.start_time, 3),
                 "e": round(w.end_time, 3), "p": round(w.probability, 3)}
                for w in seg.words
            ],
        })
        if self._on_segment is not None:
            try:
                self._on_segment(self.label, seg)
            except Exception:
                pass

    def _aggregator_loop(self) -> None:
        vad = SilenceDetector(
            silence_ms=self._silence_ms,
            aggressiveness=self._vad_aggressiveness,
        )
        window_pcm = bytearray()
        window_start: Optional[float] = None
        window_end: float = 0.0
        last_periodic_tick = 0.0

        def flush_window(reason: str) -> None:
            nonlocal window_pcm, window_start, window_end
            if not window_pcm or window_start is None:
                window_pcm = bytearray()
                window_start = None
                window_end = 0.0
                vad.reset_counts()
                return
            voiced_ms = SilenceDetector.voiced_ms(vad.voiced_frames)
            voiced_frac = (
                vad.voiced_frames / vad.total_frames if vad.total_frames else 0.0
            )
            ws, we = window_start, window_end
            if voiced_ms < self._min_voiced_ms or voiced_frac < self._min_voiced_frac:
                self.dropped_silent_count += 1
                self._debug("audio_window", {
                    "stream": self.label,
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": "dropped_silent",
                    "trigger": reason,
                })
                window_pcm = bytearray()
                window_start = None
                window_end = 0.0
                vad.reset_counts()
                return
            try:
                self.window_q.put(
                    AudioWindow(pcm=bytes(window_pcm), start_time=ws, end_time=we),
                    timeout=2.0,
                )
                self.windows_accepted_count += 1
                self._debug("audio_window", {
                    "stream": self.label,
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": reason,
                })
            except queue.Full:
                self.dropped_queue_full_count += 1
                self._debug("audio_window", {
                    "stream": self.label,
                    "start": round(ws, 3), "end": round(we, 3),
                    "voiced_ms": int(voiced_ms),
                    "voiced_frac": round(voiced_frac, 3),
                    "reason": "dropped_queue_full",
                    "trigger": reason,
                })
            window_pcm = bytearray()
            window_start = None
            window_end = 0.0
            vad.reset_counts()

        while not self._app_stop.is_set() and not self._local_stop.is_set():
            try:
                item = self._audio_input_q.get(timeout=0.5)
            except queue.Empty:
                # Periodic non-silence-boundary tick for raw chunk flushes.
                if self._on_silence_boundary is not None:
                    now = time.monotonic()
                    if now - last_periodic_tick > self._chunk_flush_periodic_seconds:
                        last_periodic_tick = now
                        try:
                            self._on_silence_boundary()
                        except Exception:
                            pass
                continue

            if item is MUTE_RESET:
                if window_pcm:
                    window_pcm = bytearray()
                    window_start = None
                    window_end = 0.0
                    vad.reset_counts()
                continue

            ts, data = item
            if window_start is None:
                window_start = ts
            window_pcm.extend(data)
            window_end = ts + (len(data) / 2 / self._sample_rate)

            saw_boundary = False
            for evt in vad.feed(data):
                if evt == "boundary":
                    saw_boundary = True
                    break

            window_seconds = window_end - window_start
            force_flush_now = (
                self._force_flush.is_set() and self._audio_input_q.empty()
            )
            if force_flush_now:
                self._force_flush.clear()
            if force_flush_now or saw_boundary or window_seconds >= self._max_window_seconds:
                reason = (
                    "forced" if force_flush_now
                    else "silence" if saw_boundary
                    else "max_window"
                )
                flush_window(reason=reason)
                if (saw_boundary or force_flush_now) and self._on_silence_boundary is not None:
                    try:
                        self._on_silence_boundary()
                    except Exception:
                        pass

        flush_window(reason="shutdown")

    def _drain_loop(self) -> None:
        while not self._app_stop.is_set() and not self._local_stop.is_set():
            try:
                seg: Segment = self.segment_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._emit_segment(seg)
