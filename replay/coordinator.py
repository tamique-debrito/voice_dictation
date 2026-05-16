"""ReplayCoordinator — replays persisted streams into a real
TranscriptStreamManager for deterministic re-runs.

Loads every JSONL stream from a session directory, merges them into one
heap ordered by ``(emit_accepted_ms, seq)``, then dispatches events in
that exact order. Non-segment events go on the bus. Segment events also
go on the bus AND into the manager's direct segment queues (because the
manager consumes those via ordered direct delivery, not the bus).

Two pacing modes:
  - ``rate_multiplier == 0`` (default, "as fast as possible"): drain the
    heap with no wall-clock delay. Best for unit-test-style determinism.
  - ``rate_multiplier > 0``: pace dispatch so accepted-ms advances at the
    given multiple of real time. ``1.0`` = real-time playback; ``5.0`` =
    5× faster. Useful for visual UI replay.

Live and replay share the exact same ``TranscriptStreamManager`` class
and the same EventBus topics; only the producer modules differ.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from ..event_bus import EventBus
from ..types import (
    Event, Segment, Word, WindowCompletion,
    TOPIC_PROGRESS_FAST, TOPIC_PROGRESS_HQ,
    TOPIC_SEGMENTS_FAST, TOPIC_SEGMENTS_HQ,
)


logger = logging.getLogger(__name__)


# Topic → filename mapping (must match app.PERSIST_TOPICS).
TOPIC_FILES: dict[str, str] = {
    "audio.silence": "silence_annotations.jsonl",
    "audio.chunk_committed": "audio_chunks.jsonl",
    "windows.fast": "fast_windows.jsonl",
    "windows.hq": "hq_windows.jsonl",
    "segments.fast": "fast_segments.jsonl",
    "segments.hq": "hq_segments.jsonl",
    "progress.fast": "fast_progress.jsonl",
    "progress.hq": "hq_progress.jsonl",
    "user.presses": "user_press_events.jsonl",
    "user.actions": "user_actions.jsonl",
    "paste.actions": "paste_actions.jsonl",
    "transcript.canonical": "transcript_canonical.jsonl",
}


@dataclass(order=True)
class _HeapEntry:
    sort_key: tuple   # (emit_accepted_ms, seq)
    payload: Event = None  # type: ignore[assignment]


def _load_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                logger.exception("malformed line in %s: %r", path, line)


def _record_to_event(record: dict) -> Optional[Event]:
    if "topic" not in record:
        return None
    try:
        topic = record.pop("topic")
        emit_ms = record.pop("emit_accepted_ms")
        seq = record.pop("seq")
    except KeyError:
        return None
    return Event(topic=topic, emit_accepted_ms=emit_ms, seq=seq, payload=record)


def _payload_to_segment(payload: dict) -> Segment:
    words = [
        Word(
            text=w.get("text", ""),
            start_accepted_ms=w.get("start_accepted_ms", 0),
            end_accepted_ms=w.get("end_accepted_ms", 0),
            probability=w.get("probability", 0.0),
        )
        for w in (payload.get("words") or [])
    ]
    return Segment(
        text=payload.get("text", ""),
        content_accepted_ms_start=payload["content_accepted_ms_start"],
        content_accepted_ms_end=payload["content_accepted_ms_end"],
        words=words,
        window_id=payload.get("window_id", 0),
        ends_on_marker_idx=payload.get("ends_on_marker_idx"),
    )


class ReplayCoordinator:
    def __init__(
        self,
        session_dir: str,
        bus: EventBus,
        fast_segment_q: queue.Queue[Segment],
        hq_segment_q: Optional[queue.Queue[Segment]],
        *,
        replay_topics: Optional[set[str]] = None,
        skip_paste_actions: bool = True,
    ) -> None:
        """
        replay_topics: limit replay to these topics. By default replays
            every topic except ``paste.actions`` (we don't replay output
            streams — those are what the manager produces under test).
        skip_paste_actions: convenience flag layered on top of replay_topics.
        """
        self._session_dir = session_dir
        self._bus = bus
        self._fast_q = fast_segment_q
        self._hq_q = hq_segment_q
        self._skip_paste = skip_paste_actions
        self._explicit = replay_topics

    def _topic_allowed(self, topic: str) -> bool:
        if self._explicit is not None:
            return topic in self._explicit
        if self._skip_paste and topic == "paste.actions":
            return False
        return True

    def _build_heap(self) -> list[_HeapEntry]:
        heap: list[_HeapEntry] = []
        for topic, filename in TOPIC_FILES.items():
            if not self._topic_allowed(topic):
                continue
            path = os.path.join(self._session_dir, filename)
            for record in _load_jsonl(path):
                # Ensure topic on record matches our table — strict.
                if record.get("topic") != topic:
                    continue
                ev = _record_to_event(dict(record))
                if ev is None:
                    continue
                heapq.heappush(heap, _HeapEntry(
                    sort_key=(ev.emit_accepted_ms, ev.seq), payload=ev,
                ))
        return heap

    def run(self, rate_multiplier: float = 0.0) -> int:
        """Replay events. Returns the number of dispatched events."""
        heap = self._build_heap()
        logger.info("ReplayCoordinator: %d events loaded from %s",
                    len(heap), self._session_dir)
        if not heap:
            return 0

        start_real = time.monotonic()
        start_accepted_ms = heap[0].sort_key[0]
        dispatched = 0
        while heap:
            entry = heapq.heappop(heap)
            ev = entry.payload
            if rate_multiplier > 0:
                # Wait until wall-clock catches up to where accepted-ms
                # should be at this multiplier.
                target_real = start_real + (
                    (ev.emit_accepted_ms - start_accepted_ms) / 1000.0
                ) / rate_multiplier
                delay = target_real - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self._dispatch(ev)
            dispatched += 1
        return dispatched

    def _dispatch(self, ev: Event) -> None:
        # Bus delivery for everything we replay.
        self._bus.publish(ev)
        # Segments + WindowCompletions also need direct-queue delivery to
        # the manager (its watermark advances on those).
        if ev.topic == TOPIC_SEGMENTS_FAST:
            self._fast_q.put(_payload_to_segment(ev.payload))
        elif ev.topic == TOPIC_SEGMENTS_HQ and self._hq_q is not None:
            self._hq_q.put(_payload_to_segment(ev.payload))
        elif ev.topic == TOPIC_PROGRESS_FAST:
            self._fast_q.put(WindowCompletion(
                stream_label="fast",
                window_id=ev.payload.get("window_id", 0),
                end_accepted_ms=ev.payload.get("transcribed_up_to_accepted_ms", 0),
                dropped=ev.payload.get("dropped", False),
            ))
        elif ev.topic == TOPIC_PROGRESS_HQ and self._hq_q is not None:
            self._hq_q.put(WindowCompletion(
                stream_label="hq",
                window_id=ev.payload.get("window_id", 0),
                end_accepted_ms=ev.payload.get("transcribed_up_to_accepted_ms", 0),
                dropped=ev.payload.get("dropped", False),
            ))
