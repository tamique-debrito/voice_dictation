"""In-process pub/sub for low-bandwidth metadata events.

Design notes:

* PCM bytes do NOT go on the bus — they flow over direct queues (see
  ``audio_preprocessor`` outputs and ``audio_windower.window_q``). The bus
  carries small JSON-shaped records: silence boundaries, window emits,
  segment emits, presses, actions, paste actions, periodic status.

* Each subscriber owns its own bounded deque + thread, so a slow
  subscriber (e.g. a stalled persister) cannot block the producer or
  other subscribers. Drop policy on full is ``drop_oldest`` — fine for
  UI history (the ring buffer is bounded anyway) and for persisters
  (which should not stall in practice).

* Topics are plain strings. ``subscribe("windows.*", ...)`` wildcards
  only one trailing segment (split on '.'). Status topics use this so
  ``status.*`` catches ``status.preprocessor``, ``status.fast``, etc.

* No external dependencies. Roughly 150 LOC.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Iterable, Optional

from .types import Event


logger = logging.getLogger(__name__)


class _Subscription:
    """One subscriber's private queue + worker thread."""

    __slots__ = ("topic_pattern", "callback", "queue", "thread", "stop", "lock",
                 "cond", "name", "dropped_count")

    def __init__(
        self,
        topic_pattern: str,
        callback: Callable[[Event], None],
        max_queued: int,
        name: str,
    ) -> None:
        self.topic_pattern = topic_pattern
        self.callback = callback
        self.queue: deque[Event] = deque(maxlen=max_queued)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.name = name
        self.dropped_count = 0
        self.thread = threading.Thread(
            target=self._run, name=f"bus-sub-{name}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def offer(self, ev: Event) -> None:
        with self.cond:
            if len(self.queue) == self.queue.maxlen:
                # drop_oldest: deque(maxlen=...) auto-evicts on append
                self.dropped_count += 1
            self.queue.append(ev)
            self.cond.notify()

    def _run(self) -> None:
        while not self.stop.is_set():
            with self.cond:
                while not self.queue and not self.stop.is_set():
                    self.cond.wait(timeout=0.5)
                if self.stop.is_set():
                    return
                ev = self.queue.popleft()
            try:
                self.callback(ev)
            except Exception:
                logger.exception("subscriber %s failed on %s", self.name, ev.topic)

    def shutdown(self) -> None:
        with self.cond:
            self.stop.set()
            self.cond.notify_all()


def _topic_matches(pattern: str, topic: str) -> bool:
    """Match ``pattern`` against ``topic``.

    Trailing ``.*`` wildcards exactly one segment. Exact match otherwise.
    ``*`` alone matches everything.
    """
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        if not topic.startswith(prefix + "."):
            return False
        rest = topic[len(prefix) + 1:]
        return "." not in rest
    return pattern == topic


class EventBus:
    """Single in-process bus. Thread-safe."""

    def __init__(self) -> None:
        self._subs: list[_Subscription] = []
        self._lock = threading.Lock()

    def subscribe(
        self,
        topic_pattern: str,
        callback: Callable[[Event], None],
        *,
        max_queued: int = 1024,
        name: Optional[str] = None,
    ) -> _Subscription:
        sub = _Subscription(
            topic_pattern=topic_pattern,
            callback=callback,
            max_queued=max_queued,
            name=name or topic_pattern,
        )
        with self._lock:
            self._subs.append(sub)
        sub.start()
        return sub

    def publish(self, event: Event) -> None:
        # Snapshot under lock; deliver outside to keep publish lock-free
        # for the actual handoff.
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if _topic_matches(sub.topic_pattern, event.topic):
                sub.offer(event)

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Wait until every subscriber's queue is empty AND its worker
        is idle (waiting on the condition variable, not in a callback).

        Returns True on idle, False on timeout. Used by the replay harness
        and tests to drain async subscriber dispatch before shutdown.
        """
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            with self._lock:
                subs = list(self._subs)
            all_empty = True
            for sub in subs:
                with sub.cond:
                    if sub.queue:
                        all_empty = False
                        break
            if all_empty:
                # Subscribers may have just popped an item and be running
                # their callback. A tiny sleep gives those callbacks time
                # to return before we declare idle.
                _time.sleep(0.02)
                with self._lock:
                    subs = list(self._subs)
                still_empty = all(
                    not sub.queue for sub in subs
                )
                if still_empty:
                    return True
            else:
                _time.sleep(0.01)
        return False

    def shutdown(self) -> None:
        with self._lock:
            subs = list(self._subs)
            self._subs.clear()
        for sub in subs:
            sub.shutdown()
        for sub in subs:
            sub.thread.join(timeout=1.0)

    def stats(self) -> dict[str, dict]:
        with self._lock:
            return {
                sub.name: {
                    "pattern": sub.topic_pattern,
                    "queue_depth": len(sub.queue),
                    "dropped": sub.dropped_count,
                }
                for sub in self._subs
            }


__all__ = ["EventBus", "Event"]
