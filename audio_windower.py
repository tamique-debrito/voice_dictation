"""AudioWindower — assembles ``AudioWindow`` objects from accepted-audio
segments plus a marker stream.

One instance per stream (fast / hq). Same class — only ``StreamConfig``
differs.

Inputs:
  - Direct put queue of ``AcceptedAudioSegment`` from AudioPreprocessor.
  - Bus subscription to silence-boundary + clipboard-end-press markers.

Cut rules (priority order, evaluated after each accepted segment is consumed):
  1. ``len(window) >= max_window_seconds`` → flush(reason="max_window").
  2. If ``cfg.flush_on_markers`` AND ``len(window) >= min_window_seconds``
     AND there is at least one pending marker → flush(reason from marker;
     ``ends_on_marker_idx`` from clipboard-end marker if present).
  3. Otherwise continue accumulating.

On flush:
  - Push ``AudioWindow`` to ``self.window_q`` with put-timeout. On Full,
    drop the window and publish ``windows.<label>.dropped_queue_full``.
  - Publish ``windows.<label>`` event with the window's metadata.
  - Reset the in-progress buffer.

Markers that arrive while the window is too short to flush are queued and
flushed on the next eligible segment-arrival. If multiple markers queue up
between flushes, only the most recent is used (the older ones are folded —
each fires its own bus event for the timeline, but the windower only acts
on the latest).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .runtime_config import AudioCaptureConfig, StreamConfig
from .types import (
    AcceptedAudioSegment, AudioWindow, Event, WindowCompletion,
    TOPIC_AUDIO_SILENCE, TOPIC_USER_PRESSES,
    TOPIC_WINDOWS_FAST, TOPIC_WINDOWS_HQ,
    TOPIC_PROGRESS_FAST, TOPIC_PROGRESS_HQ,
)


logger = logging.getLogger(__name__)


def _topic_for(label: str) -> str:
    if label == "fast":
        return TOPIC_WINDOWS_FAST
    if label == "hq":
        return TOPIC_WINDOWS_HQ
    return f"windows.{label}"


def _drop_topic_for(label: str) -> str:
    return f"{_topic_for(label)}.dropped_queue_full"


def _progress_topic_for(label: str) -> str:
    if label == "fast":
        return TOPIC_PROGRESS_FAST
    if label == "hq":
        return TOPIC_PROGRESS_HQ
    return f"progress.{label}"


@dataclass
class _PendingMarker:
    kind: str               # "silence" | "clipboard_end"
    marker_idx: Optional[int]
    arrived_at_accepted_ms: int


class AudioWindower:
    def __init__(
        self,
        config: StreamConfig,
        audio_cfg: AudioCaptureConfig,
        bus: EventBus,
        clock: AcceptedClock,
        stop_event: threading.Event,
    ) -> None:
        self._cfg = config
        self._audio = audio_cfg
        self._bus = bus
        self._clock = clock
        self._stop = stop_event
        self.label = config.label

        self._max_ms = int(config.max_window_seconds * 1000)
        self._min_ms = int(config.min_window_seconds * 1000)

        # Output queue to the Transcriber. Carries AudioWindow for real
        # work AND WindowCompletion tombstones when we drop a window due
        # to queue-full backpressure (so the downstream watermark advances).
        self.window_q: queue.Queue = queue.Queue(
            maxsize=config.window_q_maxsize,
        )

        # In-flight window state.
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._window_start_ms: Optional[int] = None
        self._window_end_ms: int = 0
        self._window_id_counter = 0
        # Markers waiting for a flush opportunity.
        self._pending: deque[_PendingMarker] = deque(maxlen=16)

        # Counters.
        self.windows_emitted = 0
        self.dropped_queue_full = 0

        # Bus subscriptions (set in start()).
        self._silence_sub = None
        self._press_sub = None

        logger.info(
            "AudioWindower(%s) initialized "
            "(max=%.1fs, min=%.1fs, flush_on_markers=%s, q_max=%d)",
            self.label, config.max_window_seconds, config.min_window_seconds,
            config.flush_on_markers, config.window_q_maxsize,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._silence_sub = self._bus.subscribe(
            TOPIC_AUDIO_SILENCE, self._on_silence_marker,
            name=f"windower.{self.label}.silence",
        )
        self._press_sub = self._bus.subscribe(
            TOPIC_USER_PRESSES, self._on_press_marker,
            name=f"windower.{self.label}.press",
        )
        logger.info("AudioWindower(%s).start", self.label)

    def shutdown(self) -> None:
        # Final flush so trailing audio reaches the transcriber.
        self._flush("shutdown")
        if self._silence_sub:
            self._silence_sub.shutdown()
        if self._press_sub:
            self._press_sub.shutdown()
        logger.info(
            "AudioWindower(%s).shutdown (emitted=%d, dropped_q_full=%d)",
            self.label, self.windows_emitted, self.dropped_queue_full,
        )

    def stats(self) -> dict:
        with self._lock:
            buf_ms = (
                (self._window_end_ms - self._window_start_ms)
                if self._window_start_ms is not None else 0
            )
            pending = len(self._pending)
        return {
            "label": self.label,
            "windows_emitted": self.windows_emitted,
            "dropped_queue_full": self.dropped_queue_full,
            "buffer_ms": buf_ms,
            "pending_markers": pending,
            "window_q_size": self.window_q.qsize(),
            "window_q_maxsize": self.window_q.maxsize,
        }

    # ------------------------------------------------------------------
    # Direct-queue input (called by AudioPreprocessor)
    # ------------------------------------------------------------------

    def put(self, segment: AcceptedAudioSegment, timeout: float = 0.5) -> None:
        with self._lock:
            if self._window_start_ms is None:
                self._window_start_ms = segment.start_accepted_ms
            self._buf.extend(segment.pcm)
            self._window_end_ms = segment.end_accepted_ms
            window_ms = self._window_end_ms - self._window_start_ms
        # Cut decision outside the lock (it acquires its own).
        self._maybe_flush(window_ms)

    def _maybe_flush(self, window_ms: int) -> None:
        # Rule 1: max_window.
        if window_ms >= self._max_ms:
            self._flush("max_window")
            return
        # Rule 2: marker flush.
        if not self._cfg.flush_on_markers:
            return
        with self._lock:
            if not self._pending:
                return
            # Clipboard-end always wins — the user has signaled "I'm done,"
            # and waiting hurts paste latency. It also bypasses min_ms;
            # the downstream watermark needs this window to land so the
            # manager can fire the paste, even if it's tiny.
            clip_end = next(
                (m for m in reversed(self._pending) if m.kind == "clipboard_end"),
                None,
            )
            if clip_end is None and window_ms < self._min_ms:
                # Plain silence marker with too-small buffer — keep accumulating.
                return
            marker = clip_end if clip_end is not None else self._pending[-1]
            self._pending.clear()
        if marker.kind == "clipboard_end":
            self._flush("marker_end_clipboard", marker.marker_idx)
        else:
            self._flush("marker", None)

    # ------------------------------------------------------------------
    # Marker stream (bus subscribers)
    # ------------------------------------------------------------------

    def _on_silence_marker(self, ev: Event) -> None:
        with self._lock:
            self._pending.append(_PendingMarker(
                kind="silence",
                marker_idx=None,
                arrived_at_accepted_ms=ev.emit_accepted_ms,
            ))
            window_ms = (
                self._window_end_ms - self._window_start_ms
                if self._window_start_ms is not None else 0
            )
        # _maybe_flush gates on min_ms internally for silence markers.
        if self._cfg.flush_on_markers:
            self._maybe_flush(window_ms)

    def _on_press_marker(self, ev: Event) -> None:
        # Only clipboard-end presses are flush triggers. The press payload
        # carries kind="release" and an "is_clipboard_end" flag set by the
        # UserActionManager when it composes the press.
        if not ev.payload.get("is_clipboard_end"):
            return
        idx = ev.payload.get("press_idx")
        with self._lock:
            self._pending.append(_PendingMarker(
                kind="clipboard_end",
                marker_idx=idx,
                arrived_at_accepted_ms=ev.emit_accepted_ms,
            ))
            window_ms = (
                self._window_end_ms - self._window_start_ms
                if self._window_start_ms is not None else 0
            )
        # Clipboard-end bypasses min_ms; flush whatever's in the buffer
        # (could be small or even empty). If empty, _flush is a no-op and
        # the manager will fire via the watermark from previously emitted
        # windows alone.
        if self._cfg.flush_on_markers:
            self._maybe_flush(window_ms)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def _flush(self, reason: str, ends_on_marker_idx: Optional[int] = None) -> None:
        with self._lock:
            if not self._buf or self._window_start_ms is None:
                return
            pcm = bytes(self._buf)
            start = self._window_start_ms
            end = self._window_end_ms
            self._buf = bytearray()
            self._window_start_ms = None
            self._window_end_ms = 0
            self._window_id_counter += 1
            wid = self._window_id_counter

        # voiced_ms == window length in accepted-ms because silence was
        # already stripped upstream. Kept as an explicit field on the event
        # for backward-compatibility with the v1 timeline shape.
        voiced_ms = end - start

        window = AudioWindow(
            pcm=pcm,
            start_accepted_ms=start,
            end_accepted_ms=end,
            voiced_ms=voiced_ms,
            flush_reason=reason,
            ends_on_marker_idx=ends_on_marker_idx,
            window_id=wid,
        )

        try:
            self.window_q.put(window, timeout=2.0)
            self.windows_emitted += 1
        except queue.Full:
            self.dropped_queue_full += 1
            self._bus.publish(Event(
                topic=_drop_topic_for(self.label),
                emit_accepted_ms=self._clock.now_accepted_ms(),
                seq=next_seq(),
                payload={
                    "window_id": wid,
                    "start_accepted_ms": start,
                    "end_accepted_ms": end,
                    "voiced_ms": voiced_ms,
                    "flush_reason": reason,
                    "ends_on_marker_idx": ends_on_marker_idx,
                },
            ))
            logger.warning(
                "AudioWindower(%s) dropped window (queue full): id=%d, ms=%d",
                self.label, wid, voiced_ms,
            )
            # Push a tombstone so the downstream watermark still advances
            # past this window's end. Block briefly — if even the tombstone
            # can't fit, the consumer is wedged anyway.
            try:
                self.window_q.put(WindowCompletion(
                    stream_label=self.label, window_id=wid,
                    end_accepted_ms=end, dropped=True,
                ), timeout=2.0)
            except queue.Full:
                logger.error(
                    "AudioWindower(%s) couldn't even push tombstone for id=%d",
                    self.label, wid,
                )
            return

        self._bus.publish(Event(
            topic=_topic_for(self.label),
            emit_accepted_ms=self._clock.now_accepted_ms(),
            seq=next_seq(),
            payload={
                "window_id": wid,
                "start_accepted_ms": start,
                "end_accepted_ms": end,
                "voiced_ms": voiced_ms,
                "flush_reason": reason,
                "ends_on_marker_idx": ends_on_marker_idx,
            },
        ))
