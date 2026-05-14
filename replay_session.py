#!/usr/bin/env python3
"""Replay a recorded persistent-dictation session in the widget.

Two recording formats are supported:

**Modern (per-stream files).** The session dir contains some of
``stream_fast.jsonl``, ``stream_hq.jsonl``, ``stream_user.jsonl``,
``stream_observability.jsonl``. Replay merges them by ``(ts, seq)`` and
feeds each event through a fresh ``SessionState``; the widget's
``/status`` reads ``state.snapshot()`` at the current replay clock.
This is the load-bearing replay path going forward.

**Legacy (snapshot-sampled).** The session dir has only
``debug_events.jsonl`` + ``status_snapshots.jsonl`` (no per-stream
files). Replay picks the most recent snapshot with ``uptime_seconds
<= now`` for transcript_tail / paste_log / capture state, and overlays
debug events up to ``now``. Lower temporal resolution than the modern
path, but keeps old sessions replayable without a manual migration.

Both paths serve audio playback (if ``audio/manifest.jsonl`` exists)
through the same per-chunk wav route.

CLI:
    --speed 1.0   playback speed (2.0 = 2x, 0.5 = half)
    --port 0      widget port (0 = ephemeral; default 0)
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional

# Reuse widget.StatusServer with a snapshot-providing closure.
import widget as widget_module


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _iter_jsonl(path: str) -> Iterator[dict]:
    """Lazy generator over a JSONL — used to merge-sort large streams
    without loading everything into memory."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_audio_payload(session_dir: str) -> tuple[list[dict], Optional[str]]:
    """Returns (audio_chunks_for_status_payload, audio_dir_or_None)."""
    audio_dir = os.path.join(session_dir, "audio")
    if not os.path.isdir(audio_dir):
        return [], None
    manifest = _load_jsonl(os.path.join(audio_dir, "manifest.jsonl"))
    if not manifest:
        return [], None
    manifest.sort(key=lambda r: r.get("t_start", 0.0))
    chunks = [
        {
            "idx": int(r["idx"]),
            "t_start": float(r.get("t_start", 0.0)),
            "t_end": float(r.get("t_end", 0.0)),
            "duration_s": float(r.get("duration_s", 0.0)),
            "segments": r.get("audio_segments") or [],
        }
        for r in manifest
    ]
    return chunks, audio_dir


# ---------------------------------------------------------------------------
# Modern path: event-driven SessionState replay
# ---------------------------------------------------------------------------

# All per-stream file basenames the modern path looks for. Missing files
# are simply skipped — a session with no HQ stream just won't have
# ``stream_hq.jsonl``.
_STREAM_FILES = (
    "stream_fast.jsonl",
    "stream_hq.jsonl",
    "stream_user.jsonl",
    "stream_observability.jsonl",
)


def _has_per_stream_files(session_dir: str) -> bool:
    """At least one per-stream file exists — modern format."""
    return any(
        os.path.exists(os.path.join(session_dir, n))
        for n in _STREAM_FILES
    )


def _merged_events(session_dir: str) -> Iterator[dict]:
    """Merge all four per-stream files in (ts, seq) order. Uses
    heapq.merge so memory stays O(streams), not O(events)."""
    iters = []
    for name in _STREAM_FILES:
        path = os.path.join(session_dir, name)
        if os.path.exists(path):
            iters.append(_iter_jsonl(path))
    if not iters:
        return iter(())
    return heapq.merge(
        *iters,
        key=lambda e: (float(e.get("ts", 0.0)), int(e.get("seq", 0))),
    )


def _build_replay_clipboard_mgr(timeline):
    """Construct a ClipboardWindowManager identical to live mode's setup
    but with no debug-event emissions and no force-flush side effects.
    Lets ``SessionState._apply_user_action`` drive clipboard window
    state during replay using the same code path as live."""
    from clipboard_window import ClipboardWindowManager, ClipboardWindowType
    from config import ASIDE_MARKER_TYPE, RECORDING_MARKER_TYPE

    clipboard_types = [
        ClipboardWindowType(
            key="r", label="r", marker_type=RECORDING_MARKER_TYPE,
            can_park_others=False, can_be_parked=True,
            schedule_end_marker=True,
        ),
        ClipboardWindowType(
            key="a", label="aside", marker_type=ASIDE_MARKER_TYPE,
            can_park_others=True, can_be_parked=False,
            schedule_end_marker=False,
        ),
    ]
    return ClipboardWindowManager(
        types=clipboard_types,
        timeline=timeline,
        debug_log=lambda kind, data: None,
        request_force_flush=lambda: None,
        cursor_time_at=lambda ts: ts,  # identity — caller passes session time directly
    )


class _ReplayTimelineHost:
    """Wraps a TranscriptTimeline configured for replay-only use.

    Lives under a tempdir so any accidental chunk writes don't pollute
    the real session dir (in practice ``tick``/``force_flush`` are
    never called during replay, so this is belt-and-braces)."""

    def __init__(self, meta: dict):
        from transcript_timeline import MarkerType, TranscriptTimeline

        marker_types = [
            MarkerType(
                key=m.get("key", ""),
                type=m.get("type", ""),
                description=m.get("description", ""),
                flag=bool(m.get("flag", False)),
            )
            for m in meta.get("marker_types", [])
        ]
        self._scratch_dir = tempfile.mkdtemp(prefix="replay_")
        self.timeline = TranscriptTimeline(
            marker_types=marker_types,
            transcripts_dir=self._scratch_dir,
            session_id=meta.get("session_id", "replay"),
        )
        for s in meta.get("streams", []):
            try:
                self.timeline.register_stream(
                    s.get("stream_id", "fast"),
                    priority=int(s.get("priority", 0)),
                    model_label=s.get("model", "?"),
                    coverage_start_time=float(s.get("coverage_start_time", 0.0)),
                )
            except Exception:
                pass


class ModernReplayController:
    """Drives a fresh SessionState forward through the recorded event
    stream as the replay clock advances. status_provider returns
    state.snapshot(now_seconds=clock)."""

    def __init__(self, session_dir: str, speed: float = 1.0):
        from session_state import SessionState

        self.session_dir = session_dir
        self.speed = max(0.01, speed)
        # _t0 set at the END of __init__ so loading session metadata, the
        # event stream, and audio manifest doesn't eat into the user's
        # playback budget. Otherwise a short session whose init took
        # longer than ``duration + 5`` real seconds would trigger the
        # end-of-stream loop reset immediately on first poll.

        meta_path = os.path.join(session_dir, "session_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        self._host = _ReplayTimelineHost(meta)

        started_str = meta.get("started_at", "")
        try:
            started_at = datetime.fromisoformat(
                started_str.replace("Z", "+00:00")
            )
        except ValueError:
            started_at = datetime.now(tz=timezone.utc)
        device = meta.get("device", "?")
        compute = meta.get("compute", "?")

        self._state = SessionState(
            timeline=self._host.timeline,
            started_at=started_at,
            device=device,
            compute=compute,
        )
        self._state.set_hq_active(
            any(s.get("stream_id") == "hq" for s in meta.get("streams", []))
        )
        # Replay needs its own clipboard manager so user_action events can
        # drive clipboard window state forward through the timeline. No
        # paste callback, no debug_log emissions — replay shouldn't
        # perform OS paste effects or duplicate events into the ring.
        self._state.attach_clipboard_manager(_build_replay_clipboard_mgr(self._host.timeline))

        # Pre-load all events into memory. For long sessions this could
        # be huge — but realistically a 10-minute session has a few
        # thousand events × <500 bytes each = MBs, not GBs. If it ever
        # becomes a problem the merge iterator can stream on demand.
        self._events = list(_merged_events(session_dir))
        self._events.sort(
            key=lambda e: (float(e.get("ts", 0.0)), int(e.get("seq", 0)))
        )
        self._next_idx = 0
        self._lock = threading.Lock()

        # Duration = the largest ts in any stream, with a small tail so
        # the UI can rest at the end before looping.
        self._duration = (
            float(self._events[-1].get("ts", 0.0)) if self._events else 0.0
        )

        # Audio (optional).
        self.audio_chunks, self.audio_dir = _load_audio_payload(session_dir)

        # All init done — start the replay clock now.
        self._t0 = time.monotonic()

    @property
    def audio_dir_for_server(self) -> Optional[str]:
        return self.audio_dir if self.audio_chunks else None

    def replay_seconds(self) -> float:
        elapsed = (time.monotonic() - self._t0) * self.speed
        if self._duration > 0 and elapsed > self._duration + 5:
            # Loop after a brief pause at the end. Reset state too so we
            # rebuild from scratch.
            with self._lock:
                self._reset_locked()
            return 0.0
        return elapsed

    def _reset_locked(self) -> None:
        from session_state import SessionState
        # Rebuild state and host so the second pass is identical to the
        # first. Cheaper than trying to "rewind" the timeline.
        meta_path = os.path.join(self.session_dir, "session_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        self._host = _ReplayTimelineHost(meta)
        self._state = SessionState(
            timeline=self._host.timeline,
            started_at=self._state._started_at,
            device=self._state._device,
            compute=self._state._compute,
        )
        self._state.set_hq_active(
            any(s.get("stream_id") == "hq" for s in meta.get("streams", []))
        )
        self._state.attach_clipboard_manager(
            _build_replay_clipboard_mgr(self._host.timeline)
        )
        self._next_idx = 0
        self._t0 = time.monotonic()

    def current_status(self) -> dict:
        now = self.replay_seconds()
        # Feed any events whose ts has now been reached. State is forward-
        # only — we never un-apply. The reset path above handles loops by
        # rebuilding from scratch.
        with self._lock:
            while self._next_idx < len(self._events):
                evt = self._events[self._next_idx]
                if float(evt.get("ts", 0.0)) > now:
                    break
                self._state.ingest(evt)
                self._next_idx += 1
            payload = self._state.snapshot(now_seconds=now)

        payload["replay"] = {
            "mode": True,
            "speed": self.speed,
            "duration_seconds": round(self._duration, 3),
            "elapsed_seconds": round(now, 3),
        }
        if self.audio_chunks:
            payload["audio"] = {
                "available": True,
                "chunks": self.audio_chunks,
            }
        return payload


# ---------------------------------------------------------------------------
# Legacy path: snapshot-driven replay
# ---------------------------------------------------------------------------


class LegacyReplayController:
    """Snapshot-driven replay for old sessions (debug_events.jsonl +
    status_snapshots.jsonl, no per-stream files)."""

    def __init__(
        self,
        session_dir: str,
        events: list[dict],
        snapshots: list[dict],
        speed: float = 1.0,
        audio_chunks: Optional[list[dict]] = None,
    ):
        self.session_dir = session_dir
        self.events = sorted(events, key=lambda e: e.get("ts", 0.0))
        self.snapshots = snapshots
        self.speed = max(0.01, speed)
        self._t0 = time.monotonic()
        self._duration = 0.0
        if self.snapshots:
            self._duration = max(
                self.snapshots[-1].get("uptime_seconds", 0.0),
                self.events[-1].get("ts", 0.0) if self.events else 0.0,
            )
        self.audio_chunks = audio_chunks or []

    def replay_seconds(self) -> float:
        elapsed = (time.monotonic() - self._t0) * self.speed
        if self._duration > 0 and elapsed > self._duration + 5:
            self._t0 = time.monotonic()
            return 0.0
        return elapsed

    def current_status(self) -> dict:
        now = self.replay_seconds()
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
        if self.audio_chunks:
            out["audio"] = {"available": True, "chunks": self.audio_chunks}
        return out


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded persistent-dictation session in the widget."
    )
    parser.add_argument(
        "session_dir",
        help="Path to a session directory.",
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

    audio_chunks, audio_dir = _load_audio_payload(session_dir)
    if audio_chunks:
        print(f"loaded {len(audio_chunks)} audio chunks from {audio_dir}")

    if _has_per_stream_files(session_dir):
        controller: object = ModernReplayController(session_dir, speed=args.speed)
        # Audio chunks are owned by the controller in the modern path.
        audio_dir_for_server = controller.audio_dir_for_server  # type: ignore[attr-defined]
        print(f"replay format: per-stream (modern)")
        print(f"  duration: ~{controller._duration:.1f}s @ {controller.speed}x")  # type: ignore[attr-defined]
    else:
        events_path = os.path.join(session_dir, "debug_events.jsonl")
        snapshots_path = os.path.join(session_dir, "status_snapshots.jsonl")
        if not os.path.exists(events_path) and not os.path.exists(snapshots_path):
            print(
                f"error: no replay data in {session_dir} — expected per-stream "
                f"files, debug_events.jsonl, or status_snapshots.jsonl. Was the "
                f"session recorded with persistent.debug_recording=true?",
                file=sys.stderr,
            )
            sys.exit(2)
        events = _load_jsonl(events_path)
        snapshots = _load_jsonl(snapshots_path)
        print(f"replay format: legacy (snapshot-sampled)")
        print(f"  {len(events)} events, {len(snapshots)} snapshots")
        controller = LegacyReplayController(
            session_dir, events, snapshots, speed=args.speed,
            audio_chunks=audio_chunks,
        )
        audio_dir_for_server = audio_dir if audio_chunks else None
        print(f"  duration: ~{controller._duration:.1f}s @ {controller.speed}x")  # type: ignore[attr-defined]

    def status_provider() -> dict:
        return controller.current_status()  # type: ignore[attr-defined]

    def config_provider() -> dict:
        return {"replay": {"mode": True, "speed": controller.speed}}  # type: ignore[attr-defined]

    def config_setter(_data: dict) -> dict:
        return {"saved_to": "", "applied": [],
                "deferred": ["replay mode: config edits are no-ops"]}

    server = widget_module.StatusServer(
        status_provider,
        config_provider=config_provider,
        config_setter=config_setter,
        audio_dir=audio_dir_for_server,
    )
    host, port = server.start(port=args.port)
    print(f"replay widget: http://{host}:{port}")
    print(f"  session: {session_dir}")
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
