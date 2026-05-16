"""AudioPreprocessor — single VAD, silence-strip, accepted-ms clock owner.

Operates on 20 ms webrtcvad sub-frames (the only legal frame sizes are
10/20/30 ms at 16 kHz; we pick 20 ms as a compromise between latency and
robustness).

Gate (the only thing that decides whether a sub-frame leaves the
preprocessor): a sliding window over the most recent N sub-frames. A
sub-frame is admitted iff ≥M of the last N are classified voiced by
webrtcvad. With the defaults (N=5, M=2) this gives:

  - Leading lost speech: up to M-1 = 1 sub-frame (20 ms).
  - Trailing kept tail:  up to N-M = 3 sub-frames (60 ms).
  - Mid-utterance brief silences are NOT stripped — they're admitted as
    part of the surrounding voiced run, preserving phonetic cues for
    Whisper.

Silence-boundary events (separate from the gate): a continuous run of
*dropped* sub-frames lasting ``silence_boundary_ms`` after at least one
admitted sub-frame produces one ``audio.silence`` event. Edge-triggered —
the next boundary only fires after another admitted sub-frame.

The accepted_ms cursor advances by 20 ms for each admitted sub-frame. It
never advances during silence — that is the whole point of the clock.

Real-ms (wall-clock since session start) is maintained alongside, but only
used internally so ``stamp_accepted_ms`` can resolve press timestamps.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional, Protocol

import webrtcvad

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .runtime_config import AudioCaptureConfig, PreprocessorConfig
from .types import AcceptedAudioSegment, Event, TOPIC_AUDIO_SILENCE, TOPIC_USER_ACTIONS


logger = logging.getLogger(__name__)


VAD_FRAME_MS = 20
BYTES_PER_SAMPLE = 2  # int16


class _SinkLike(Protocol):
    def put(self, segment: AcceptedAudioSegment, timeout: float = ...) -> None: ...


class AudioPreprocessor(AcceptedClock):
    def __init__(
        self,
        config: PreprocessorConfig,
        audio_cfg: AudioCaptureConfig,
        bus: EventBus,
        stop_event: threading.Event,
        *,
        raw_input_q_maxsize: int = 200,
    ) -> None:
        self._cfg = config
        self._audio = audio_cfg
        self._bus = bus
        self._stop = stop_event

        self._sample_rate = audio_cfg.sample_rate
        self._samples_per_frame = self._sample_rate * VAD_FRAME_MS // 1000
        self._bytes_per_frame = self._samples_per_frame * BYTES_PER_SAMPLE

        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._gate_history: deque[bool] = deque(
            [False] * config.voiced_gate_frames,
            maxlen=config.voiced_gate_frames,
        )
        self._gate_threshold = config.voiced_gate_min_voiced
        self._silence_boundary_frames = max(
            1, config.silence_boundary_ms // VAD_FRAME_MS
        )

        # Sub-frame leftover buffer (a PyAudio chunk is ~64 ms = ~3.2
        # sub-frames; the residual bytes wait for the next chunk).
        self._residual = bytearray()

        # Counters / clocks.
        self._accepted_ms = 0
        self._real_ms_processed = 0
        self._lock = threading.Lock()
        self._catch_up = threading.Condition(self._lock)

        # Silence-boundary edge state.
        self._armed = False                       # only fire boundary after ≥1 admitted sub-frame
        self._dropped_run_frames = 0              # consecutive dropped sub-frames since last admit
        self._silence_run_real_ms = 0             # wall-clock accumulator for the same run

        # Sinks consume PCM payloads via direct queues.
        self._sinks: list[_SinkLike] = []

        # Raw input queue + worker thread.
        self.raw_input_q: queue.Queue[tuple[float, bytes]] = queue.Queue(
            maxsize=raw_input_q_maxsize,
        )
        self._thread: Optional[threading.Thread] = None

        # Mute gate. While muted, incoming PCM is discarded (no VAD, no
        # admit, no segment emit, _accepted_ms frozen) but _real_ms_processed
        # still advances so stamp_accepted_ms doesn't block. The first
        # mute_toggle sets this True; the next clears it.
        self._muted = False
        self._mute_lock = threading.Lock()
        self._actions_sub = None

        logger.info(
            "AudioPreprocessor initialized "
            "(aggressiveness=%d, silence_boundary_ms=%d, gate=%d/%d frames)",
            config.aggressiveness, config.silence_boundary_ms,
            config.voiced_gate_min_voiced, config.voiced_gate_frames,
        )

    # ------------------------------------------------------------------
    # AcceptedClock + time-base API
    # ------------------------------------------------------------------

    def now_accepted_ms(self) -> int:
        with self._lock:
            return self._accepted_ms

    def stats(self) -> dict:
        with self._lock:
            accepted = self._accepted_ms
            real = self._real_ms_processed
        return {
            "accepted_ms": accepted,
            "real_ms_processed": real,
            "raw_input_q_size": self.raw_input_q.qsize(),
            "raw_input_q_maxsize": self.raw_input_q.maxsize,
            "armed": self._armed,
            "dropped_run_frames": self._dropped_run_frames,
        }

    def stamp_accepted_ms(self, real_ms_target: int, timeout: float = 1.0) -> int:
        """Wait until raw frames up to ``real_ms_target`` have been
        classified, then return the accepted_ms cursor at that moment.

        The mapping is approximate: accepted_ms only advances on admitted
        sub-frames, so a press during silence lands at the accepted_ms of
        the next admitted sub-frame. That is the right semantics — silence
        has zero duration in accepted-ms.
        """
        deadline = time.monotonic() + timeout
        with self._catch_up:
            while self._real_ms_processed < real_ms_target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "stamp_accepted_ms: timed out waiting for real_ms=%d "
                        "(processed=%d)", real_ms_target, self._real_ms_processed,
                    )
                    break
                self._catch_up.wait(timeout=remaining)
            return self._accepted_ms

    # ------------------------------------------------------------------
    # Sink wiring
    # ------------------------------------------------------------------

    def add_sink(self, sink: _SinkLike) -> None:
        self._sinks.append(sink)

    def _emit_segment(self, pcm: bytes, start_ms: int, end_ms: int) -> None:
        seg = AcceptedAudioSegment(
            pcm=pcm, start_accepted_ms=start_ms, end_accepted_ms=end_ms,
        )
        for sink in self._sinks:
            try:
                sink.put(seg, timeout=0.5)
            except Exception:
                logger.exception("sink %r refused segment", sink)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._actions_sub = self._bus.subscribe(
            TOPIC_USER_ACTIONS, self._on_action, name="preprocessor.user_actions",
        )
        self._thread = threading.Thread(
            target=self._loop, name="AudioPreprocessor", daemon=True,
        )
        self._thread.start()
        logger.info("AudioPreprocessor.start")

    def set_muted(self, muted: bool) -> None:
        with self._mute_lock:
            if self._muted == muted:
                return
            self._muted = muted
            if muted:
                # Drop straddling speech so it doesn't bleed across the
                # mute boundary, and re-seed the VAD gate as all-silence
                # so unmute starts from a clean state.
                self._residual.clear()
                self._gate_history.extend(
                    [False] * self._gate_history.maxlen
                )
                self._armed = False
                self._dropped_run_frames = 0
                self._silence_run_real_ms = 0
        logger.info("AudioPreprocessor.muted=%s", muted)

    def _on_action(self, ev: Event) -> None:
        if ev.payload.get("action") != "mute_toggle":
            return
        with self._mute_lock:
            target = not self._muted
        self.set_muted(target)

    def shutdown(self) -> None:
        if self._actions_sub is not None:
            try:
                self._actions_sub.shutdown()
            except Exception:
                logger.exception("preprocessor actions_sub.shutdown failed")
            self._actions_sub = None
        if self._thread is None:
            return
        # The worker loop checks _stop on every iteration. Push a None
        # sentinel so a blocked get() returns immediately.
        try:
            self.raw_input_q.put_nowait((0.0, b""))
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("AudioPreprocessor.shutdown")

    def _loop(self) -> None:
        # Drain queued audio even after stop is set; shutdown() pushes a
        # sentinel (empty pcm) so we wake up and exit cleanly once nothing
        # is left.
        while True:
            try:
                item = self.raw_input_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            ts_seconds, pcm = item
            if not pcm:
                if self._stop.is_set():
                    return
                continue
            self._feed(ts_seconds, pcm)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _feed(self, ts_seconds: float, pcm: bytes) -> None:
        """Process one PyAudio chunk: VAD per 20ms, gate, accumulate, emit."""
        # Real-ms advances by the wall-clock duration of this chunk's
        # audio (computed from byte length, not the ts argument — that way
        # short reads from PyAudio account correctly).
        chunk_real_ms = (len(pcm) // BYTES_PER_SAMPLE) * 1000 // self._sample_rate

        with self._mute_lock:
            muted = self._muted
        if muted:
            # Drop PCM entirely: no VAD, no admit, accepted_ms frozen, no
            # segments. Still advance real_ms_processed so stamp_accepted_ms
            # doesn't block on presses that occur during mute.
            with self._catch_up:
                self._real_ms_processed += chunk_real_ms
                self._catch_up.notify_all()
            return

        self._residual.extend(pcm)

        emit_buf = bytearray()
        emit_start_accepted_ms = -1
        # Multiple boundaries can fire within one input chunk if a long
        # silence follows speech that already crossed prior thresholds.
        # Collect and publish them all (one event per boundary, in order)
        # at the end of the chunk after segment emit.
        pending_silence_events: list[tuple[int, int]] = []

        while len(self._residual) >= self._bytes_per_frame:
            frame = bytes(self._residual[:self._bytes_per_frame])
            del self._residual[:self._bytes_per_frame]
            try:
                voiced = self._vad.is_speech(frame, self._sample_rate)
            except Exception:
                voiced = False

            self._gate_history.append(voiced)
            admit = sum(self._gate_history) >= self._gate_threshold

            if admit:
                # Mark start of an emit-batch.
                if emit_start_accepted_ms < 0:
                    with self._lock:
                        emit_start_accepted_ms = self._accepted_ms
                emit_buf.extend(frame)
                with self._lock:
                    self._accepted_ms += VAD_FRAME_MS
                # Reset silence-run tracking; arm boundary detection.
                self._armed = True
                self._dropped_run_frames = 0
                self._silence_run_real_ms = 0
            else:
                # Drop the sub-frame.
                if self._armed:
                    self._dropped_run_frames += 1
                    self._silence_run_real_ms += VAD_FRAME_MS
                    if self._dropped_run_frames == self._silence_boundary_frames:
                        # Fire boundary; disarm until next admit.
                        self._armed = False
                        pending_silence_events.append((
                            self._accepted_ms,            # boundary_at_accepted_ms
                            self._silence_run_real_ms,    # duration_real_ms (so far)
                        ))

        # Advance real-ms cursor and wake stampers.
        with self._catch_up:
            self._real_ms_processed += chunk_real_ms
            self._catch_up.notify_all()

        if emit_buf:
            self._emit_segment(
                bytes(emit_buf),
                emit_start_accepted_ms,
                emit_start_accepted_ms + (len(emit_buf) // self._bytes_per_frame) * VAD_FRAME_MS,
            )

        for boundary_at, dur in pending_silence_events:
            self._bus.publish(Event(
                topic=TOPIC_AUDIO_SILENCE,
                emit_accepted_ms=boundary_at,
                seq=next_seq(),
                payload={
                    "boundary_at_accepted_ms": boundary_at,
                    "duration_real_ms": dur,
                },
            ))
