#!/usr/bin/env python3
"""Replay a recorded session in the persistent-monitoring widget.

A live session with ``debug_recording`` enabled (on by default) writes:

  <session_dir>/debug_events.jsonl    — every ring-buffer debug event
  <session_dir>/status_snapshots.jsonl — periodic full status payloads

This script reuses ``widget.py``'s HTTP server but feeds it pre-recorded
snapshots instead of live state. Controls:

  --speed 1.0   playback speed (2.0 = 2x, 0.5 = half)
  --port 0      widget port (0 = ephemeral; default 0)

The widget URL is printed to stdout. Open it in a browser; the timeline,
transcript tail, paste log, and drop chips all advance as they did live.

Replay reads ``debug_events.jsonl`` once into memory and injects events
into the served snapshot in time order; ``status_snapshots.jsonl`` supplies
the rest of the payload (transcript tail, paste log, capture state, drop
counters). If a snapshot is missing for a given replay-time, the most
recent snapshot whose timestamp <= replay-time is served.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Optional

# Reuse widget.StatusServer with a snapshot-providing closure.
import widget as widget_module


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


class ReplayController:
    """Tracks playback time and serves the right (snapshot + events) tuple."""

    def __init__(
        self,
        session_dir: str,
        events: list[dict],
        snapshots: list[dict],
        speed: float = 1.0,
    ):
        self.session_dir = session_dir
        self.events = events
        # Sort defensively.
        self.events.sort(key=lambda e: e.get("ts", 0.0))
        self.snapshots = snapshots
        self.speed = max(0.01, speed)
        # Replay time origin: snapshots are stamped with uptime_seconds
        # (relative to session start). Use the same axis here.
        self._t0 = time.monotonic()
        # Total replay duration so we can loop at the end.
        self._duration = 0.0
        if self.snapshots:
            self._duration = max(
                self.snapshots[-1].get("uptime_seconds", 0.0),
                self.events[-1].get("ts", 0.0) if self.events else 0.0,
            )

    def reset(self) -> None:
        self._t0 = time.monotonic()

    def replay_seconds(self) -> float:
        elapsed = (time.monotonic() - self._t0) * self.speed
        if self._duration > 0 and elapsed > self._duration + 5:
            # Loop after a brief pause at the end.
            self._t0 = time.monotonic()
            return 0.0
        return elapsed

    def current_status(self) -> dict:
        """Return the snapshot whose uptime <= replay_seconds, merged with
        the slice of debug events that have occurred by now."""
        now = self.replay_seconds()
        # Binary-search the snapshot list for the right one.
        snap: Optional[dict] = None
        for s in self.snapshots:
            if s.get("uptime_seconds", 0.0) <= now:
                snap = s
            else:
                break
        if snap is None:
            snap = self.snapshots[0] if self.snapshots else {
                "session_id": "(replay)", "model": "(replay)",
                "device": "?", "compute": "?",
                "started_at": "", "uptime_seconds": 0.0,
                "chunk_count": 0,
                "capture": {"mode": "passive", "r_active": False,
                            "aside_active": False, "muted": False},
                "open_marker": None, "open_marker_since": None,
                "paste_log": [], "transcript_tail": "",
                "debug_events": [],
            }
        out = dict(snap)
        # Slice debug events up to `now` and put them on the payload so the
        # widget's timeline + event log advance in lockstep with replay.
        relevant = [e for e in self.events if e.get("ts", 0.0) <= now]
        if len(relevant) > 500:
            relevant = relevant[-500:]
        out["debug_events"] = relevant
        out["now_seconds"] = round(now, 3)
        out["replay"] = {
            "mode": True,
            "speed": self.speed,
            "duration_seconds": round(self._duration, 3),
            "elapsed_seconds": round(now, 3),
        }
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded persistent-dictation session in the widget."
    )
    parser.add_argument(
        "session_dir",
        help="Path to a session directory containing debug_events.jsonl "
             "and status_snapshots.jsonl.",
    )
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0).")
    parser.add_argument("--port", type=int, default=0,
                        help="Widget port (0 = ephemeral).")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        print(f"error: {session_dir} is not a directory", file=sys.stderr)
        sys.exit(2)
    events_path = os.path.join(session_dir, "debug_events.jsonl")
    snapshots_path = os.path.join(session_dir, "status_snapshots.jsonl")
    if not os.path.exists(events_path) and not os.path.exists(snapshots_path):
        print(
            f"error: no debug_events.jsonl or status_snapshots.jsonl in "
            f"{session_dir} — was the session recorded with "
            f"persistent.debug_recording=true?",
            file=sys.stderr,
        )
        sys.exit(2)

    events = _load_jsonl(events_path)
    snapshots = _load_jsonl(snapshots_path)
    print(f"loaded {len(events)} debug events, {len(snapshots)} status snapshots")

    controller = ReplayController(
        session_dir, events, snapshots, speed=args.speed
    )

    # Read-only widget: config GET returns the snapshot's saved config view
    # (or an empty dict); config PUT is a no-op so users can't accidentally
    # mutate the replay.
    def status_provider() -> dict:
        return controller.current_status()

    def config_provider() -> dict:
        return {"replay": {"mode": True, "speed": controller.speed}}

    def config_setter(_data: dict) -> dict:
        return {"saved_to": "", "applied": [],
                "deferred": ["replay mode: config edits are no-ops"]}

    server = widget_module.StatusServer(
        status_provider,
        config_provider=config_provider,
        config_setter=config_setter,
    )
    host, port = server.start(port=args.port)
    print(f"replay widget: http://{host}:{port}")
    print(f"  session: {session_dir}")
    print(f"  duration: ~{controller._duration:.1f}s @ {controller.speed}x")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
