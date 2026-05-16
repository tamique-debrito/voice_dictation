"""Replay harness — wires a real TranscriptStreamManager + bus to mocked
producers (the ReplayCoordinator) for deterministic re-runs.

Usage:
    python -m voice_dictation.replay.harness <session_dir>
       [--rate-multiplier 0.0]
       [--out-paste-actions PATH]

Loads every persisted stream except ``paste.actions`` (the stream we're
recomputing), runs them through a fresh TranscriptStreamManager, and
optionally writes the freshly-emitted ``paste.actions`` to PATH.

If the live session also has ``paste.actions.jsonl``, the harness diffs
the replay output against it and exits 0 only on byte match (after sort).
This is the determinism check called out in the plan's Verification section.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
from typing import Optional

from ..event_bus import EventBus
from ..paste_executor import NullPasteExecutor
from ..streams import EventStreamPersister
from ..transcript_stream_manager import TranscriptStreamManager
from ..types import Segment, TOPIC_PASTE_ACTIONS
from .coordinator import ReplayCoordinator, TOPIC_FILES


logger = logging.getLogger(__name__)


class _ReplayClock:
    """Bus is the source of truth in replay; we just remember the latest
    ``emit_accepted_ms`` seen so other modules' ``now_accepted_ms()`` calls
    return a coherent value during dispatch."""

    def __init__(self) -> None:
        self._ms = 0

    def now_accepted_ms(self) -> int:
        return self._ms

    def set(self, ms: int) -> None:
        self._ms = ms


def run_replay(
    session_dir: str,
    rate_multiplier: float = 0.0,
    out_paste_actions: Optional[str] = None,
) -> tuple[int, list[dict]]:
    """Run a replay. Returns (events_dispatched, replay_paste_actions)."""
    bus = EventBus()
    fast_q: queue.Queue[Segment] = queue.Queue()
    hq_q: queue.Queue[Segment] = queue.Queue()
    stop = threading.Event()

    # Track the accepted-ms cursor by listening on every bus event.
    clock = _ReplayClock()
    bus.subscribe(
        "*", lambda ev: clock.set(ev.emit_accepted_ms),
        name="replay.clock-track",
    )

    # The module under test.
    manager = TranscriptStreamManager(bus, clock, fast_q, hq_q, stop)
    manager.start()

    null_executor = NullPasteExecutor(bus)

    # Optional persister for the replayed paste_actions.
    persister = None
    if out_paste_actions is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_paste_actions)) or ".", exist_ok=True)
        persister = EventStreamPersister(
            bus, TOPIC_PASTE_ACTIONS, out_paste_actions,
        )

    coord = ReplayCoordinator(
        session_dir, bus, fast_q, hq_q,
        skip_paste_actions=True,
    )
    n = coord.run(rate_multiplier=rate_multiplier)
    logger.info("dispatched %d events", n)

    # Drain async subscriber dispatch (action subscriber etc.) before
    # shutdown, so any segment-arrival or action-arrival processing the
    # manager owes us has actually run.
    bus.wait_idle(timeout=3.0)

    # Allow the manager to drain pending pastes + the persister to write.
    stop.set()
    manager.shutdown()
    bus.wait_idle(timeout=2.0)
    if persister is not None:
        persister.shutdown()
    null_executor.shutdown()
    bus.shutdown()

    return n, list(null_executor.pastes)


def _normalize_pastes(records: list[dict]) -> list[dict]:
    """Strip non-deterministic fields and sort by paste_idx for diffing."""
    out = []
    for r in records:
        r = dict(r)
        # emit_accepted_ms and seq are coordinator-side timing fields that
        # could legitimately differ between live and replay (the live one
        # was stamped at paste time; replay re-stamps at replay time).
        r.pop("emit_accepted_ms", None)
        r.pop("seq", None)
        out.append(r)
    out.sort(key=lambda r: r.get("paste_idx", 0))
    return out


def _load_live_pastes(session_dir: str) -> list[dict]:
    path = os.path.join(session_dir, TOPIC_FILES["paste.actions"])
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r.pop("topic", None)
            r.pop("emit_accepted_ms", None)
            r.pop("seq", None)
            out.append(r)
    out.sort(key=lambda r: r.get("paste_idx", 0))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a v2 session")
    parser.add_argument("session_dir")
    parser.add_argument("--rate-multiplier", type=float, default=0.0)
    parser.add_argument("--out-paste-actions", default=None)
    parser.add_argument("--check-determinism", action="store_true",
                        help="diff replay paste_actions against live; "
                             "exit 1 on mismatch")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    n, replay = run_replay(
        args.session_dir,
        rate_multiplier=args.rate_multiplier,
        out_paste_actions=args.out_paste_actions,
    )
    if args.check_determinism:
        live = _load_live_pastes(args.session_dir)
        replay_norm = _normalize_pastes(replay)
        if live != replay_norm:
            logger.error("DETERMINISM MISMATCH")
            logger.error("live:   %s", json.dumps(live, indent=2))
            logger.error("replay: %s", json.dumps(replay_norm, indent=2))
            return 1
        logger.info("determinism check PASSED (%d pastes match)", len(live))
    return 0


if __name__ == "__main__":
    sys.exit(main())
