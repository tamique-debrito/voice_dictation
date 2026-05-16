"""The accepted-ms clock and global sequence counter.

Producers stamp every event with ``(emit_accepted_ms, seq)`` so the replay
coordinator can dispatch in the exact order the live system saw them.

In live, the clock is fronted by the AudioPreprocessor (it owns the
voiced-time cursor). In replay, the ReplayCoordinator drives it directly.
This module defines the interface and the global seq counter so all
producers can share it without depending on the preprocessor concrete class.
"""

from __future__ import annotations

import itertools
import threading
from typing import Protocol


class AcceptedClock(Protocol):
    """Read-only view of the accepted-ms cursor."""

    def now_accepted_ms(self) -> int:
        ...


class _SeqCounter:
    """Process-wide monotonic counter, thread-safe.

    Singleton pattern is fine here: every producer in a process must share
    the same counter for ``seq`` to break replay ties deterministically.
    """

    def __init__(self) -> None:
        self._it = itertools.count(1)
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            return next(self._it)


_SEQ = _SeqCounter()


def next_seq() -> int:
    return _SEQ.next()
