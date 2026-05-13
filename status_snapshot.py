"""Builds the JSON payload served by the widget's ``/status`` endpoint.

After the TranscriptTimeline refactor, all transcript state (tail text,
chunk counts, HQ leading edge, marker state, source-tagged spans for
two-color rendering) comes from one ``TranscriptTimeline`` instance.

The payload schema is preserved (same keys the widget HTML/JS reads) so
the client doesn't need changes.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def build_status(
    *,
    timeline,
    hq_active: bool,
    device: str,
    compute: str,
    started_at: datetime,
    audio_fanout,
    clipboard_mgr,
    paste_log: list,
    marker_open_since,
    recorder,
    debug,
    stream_stats: dict = None,
) -> dict:
    r_active = clipboard_mgr.is_active("r")
    aside_active = clipboard_mgr.is_active("a")

    if r_active and aside_active:
        mode = "r+aside"
    elif r_active:
        mode = "r"
    elif aside_active:
        mode = "aside"
    else:
        mode = "passive"

    open_marker = timeline.open_marker_type()
    if open_marker is None:
        marker_open_since = None

    payload = {
        "session_id": timeline.session_id,
        "session_dir": timeline.session_dir,
        "model": timeline.model_label("fast"),
        "device": device,
        "compute": compute,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "uptime_seconds": (
            datetime.now(tz=timezone.utc) - started_at
        ).total_seconds(),
        "chunk_count": timeline.chunk_count(),  # canonical chunks
        "capture": {
            "mode": mode,
            "r_active": r_active,
            "aside_active": aside_active,
            "muted": audio_fanout.is_muted(),
        },
        "open_marker": open_marker,
        "open_marker_since": marker_open_since,
        "paste_log": list(paste_log),
        "transcript_tail": timeline.tail(2000),
        "now_seconds": round(
            time.monotonic() - (recorder.started_monotonic or time.monotonic()),
            3,
        ),
        "debug_events": debug.snapshot(limit=500),
    }
    if hq_active and timeline.stream_exists("hq"):
        payload["hq"] = {
            "model": timeline.model_label("hq"),
            "chunk_count": timeline.chunk_count("hq"),
            "leading_edge_seconds": round(timeline.hq_high_watermark(), 3),
        }
        spans = timeline.tail_annotated(2000)
        payload["transcript_tail_merged"] = spans
    if stream_stats is not None:
        payload["stream_stats"] = stream_stats
    return payload
