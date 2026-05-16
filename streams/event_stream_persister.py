"""Per-topic JSONL writer that subscribes to the EventBus.

One persister per topic (or topic-pattern). Each event is serialized as one
JSON line with ``emit_accepted_ms``, ``seq``, and the payload flattened in.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from ..event_bus import EventBus
from ..types import Event


logger = logging.getLogger(__name__)


class EventStreamPersister:
    """Subscribe to a topic pattern; append every matching event as JSONL.

    Sessions are file-per-topic. The file is opened lazily on first event,
    flushed after every write (for cheap forensic durability), and closed
    on ``shutdown()``.
    """

    def __init__(
        self,
        bus: EventBus,
        topic_pattern: str,
        output_path: str,
        *,
        max_queued: int = 4096,
    ) -> None:
        self.topic_pattern = topic_pattern
        self.output_path = output_path
        self._f = None
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._sub = bus.subscribe(
            topic_pattern,
            self._on_event,
            max_queued=max_queued,
            name=f"persist:{os.path.basename(output_path)}",
        )

    def _on_event(self, ev: Event) -> None:
        record = {
            "topic": ev.topic,
            "emit_accepted_ms": ev.emit_accepted_ms,
            "seq": ev.seq,
            **ev.payload,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            if self._f is None:
                self._f = open(self.output_path, "a", buffering=1)
            self._f.write(line)

    def shutdown(self) -> None:
        self._sub.shutdown()
        with self._lock:
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:
                    logger.exception("closing %s failed", self.output_path)
                self._f = None
