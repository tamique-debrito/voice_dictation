"""AudioFanout: drains the shared raw_q and fans frames to N subscribers.

PersistentRecorder writes int16 PCM frames (ts, bytes) into a shared
``raw_q``. With a single transcription stream the aggregator drains
raw_q directly. With multiple streams (fast + hq), each stream needs to
see the same frames in real time. AudioFanout sits between the recorder
and the aggregators: one drainer thread reads raw_q and pushes a copy of
every non-muted frame into every subscriber's input queue.

Mute is owned here (not in the per-stream aggregator) so the contract is
"audio frames a stream sees are the audio frames any stream sees" —
mute can never desync streams. On mute *transition* (off → on) the
fanout pushes a single ``MUTE_RESET`` sentinel into each subscriber so
they can drop any partially-accumulated window state and start fresh
when audio resumes.

Backpressure: each subscriber queue is bounded. On Full, the fanout
drops the oldest queued item and pushes the new one. This stops a slow
subscriber (e.g. an HQ stream behind on a large model) from blocking
faster ones. The lost frames manifest as missing audio in that
subscriber's transcript region — for the HQ case, the merged view
falls back to fast-stream text for those regions.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional


# Sentinel pushed into each subscriber queue when mute transitions on.
# Subscribers should reset any in-flight windowing state when they see
# this and continue draining normally.
MUTE_RESET = ("__mute_reset__",)


class AudioFanout:
    def __init__(self, raw_q: "queue.Queue", stop_event: threading.Event):
        self._raw_q = raw_q
        self._stop = stop_event
        self._muted = threading.Event()
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, maxsize: int = 200) -> "queue.Queue":
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue") -> bool:
        """Stop fanning frames to ``q``. Returns True if it was registered."""
        with self._lock:
            try:
                self._subscribers.remove(q)
                return True
            except ValueError:
                return False

    # ------------------------------------------------------------------
    # Mute
    # ------------------------------------------------------------------

    def set_muted(self, val: bool) -> None:
        if val:
            self._muted.set()
        else:
            self._muted.clear()

    def is_muted(self) -> bool:
        return self._muted.is_set()

    def toggle_muted(self) -> bool:
        """Flip mute; return the new state."""
        if self._muted.is_set():
            self._muted.clear()
            return False
        self._muted.set()
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="AudioFanout", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        # Caller is responsible for setting stop_event; we just join.
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        last_muted = False
        while not self._stop.is_set():
            try:
                item = self._raw_q.get(timeout=0.5)
            except queue.Empty:
                # No frames available; let subscribers' own queue timeouts
                # drive their periodic work (chunk-flush ticks, etc).
                continue

            muted_now = self._muted.is_set()
            if muted_now and not last_muted:
                self._broadcast(MUTE_RESET)
            last_muted = muted_now

            if muted_now:
                # Frame is dropped — subscribers see a gap.
                continue

            self._broadcast(item)

    def _broadcast(self, item: Any) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for s in subs:
            try:
                s.put_nowait(item)
            except queue.Full:
                # Drop oldest, then push new. The dropped frame is lost to
                # this subscriber only.
                try:
                    s.get_nowait()
                except queue.Empty:
                    pass
                try:
                    s.put_nowait(item)
                except queue.Full:
                    pass
