#!/usr/bin/env python3
"""Annotation-mode launcher for voice-dictation sessions.

Sibling of ``replay_session.py``. Where replay walks the timeline in
wall-clock order through SessionState, annotate is **random access** —
the whole session is visible up front, no advancing clock. It serves a
dedicated annotation HTML page (``/annotate``) that lets you scrub each
chunk's audio, see segments in a ±15s window around the scrub position,
edit segment text, and reject whole chunks. All edits land in
``<session_dir>/annotations.jsonl`` (append-only, last write wins per
``(chunk_idx, segment_idx)``); the per-stream JSONL and chunk text files
are never mutated.

Usage:
    python annotate_session.py <session_dir> [--port 0]

Refuses if ``<session_dir>/audio/`` is missing (annotation needs WAVs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import widget as widget_module
from replay_session import _load_jsonl, _load_audio_payload


# ---------------------------------------------------------------------------
# Segment loading
# ---------------------------------------------------------------------------

def _load_segments(session_dir: str) -> tuple[list[dict], list[dict]]:
    """Returns (hq_segments, fast_segments) flattened from per-stream JSONL.

    Each row is ``{stream, t_start, t_end, text, words}``."""
    def parse(path: str, stream: str) -> list[dict]:
        out: list[dict] = []
        for row in _load_jsonl(path):
            if row.get("kind") != "segment":
                continue
            data = row.get("data") or {}
            if data.get("stream") != stream:
                continue
            out.append({
                "stream": stream,
                "t_start": float(data.get("start", 0.0)),
                "t_end": float(data.get("end", 0.0)),
                "text": (data.get("text") or "").strip(),
                "words": data.get("words") or [],
            })
        out.sort(key=lambda r: r["t_start"])
        return out

    hq = parse(os.path.join(session_dir, "stream_hq.jsonl"), "hq")
    fast = parse(os.path.join(session_dir, "stream_fast.jsonl"), "fast")
    return hq, fast


def _assign_segments_to_chunks(
    chunks: list[dict],
    segments: list[dict],
) -> dict[int, list[dict]]:
    """Bucket segments under chunks by ``t_start``: a segment belongs to
    the chunk whose ``[t_start, t_end)`` contains its own ``t_start``.
    Segments before the first chunk attach to chunk 0; segments after
    the last chunk attach to the last chunk. Within each chunk we sort
    by start time and assign per-chunk ``segment_idx``."""
    if not chunks:
        return {}
    buckets: dict[int, list[dict]] = {c["idx"]: [] for c in chunks}
    sorted_chunks = sorted(chunks, key=lambda c: c["t_start"])
    for seg in segments:
        target = sorted_chunks[0]["idx"]
        for c in sorted_chunks:
            if seg["t_start"] >= c["t_start"]:
                target = c["idx"]
            else:
                break
        buckets[target].append(seg)
    out: dict[int, list[dict]] = {}
    for idx, segs in buckets.items():
        segs.sort(key=lambda s: s["t_start"])
        for i, s in enumerate(segs):
            s["segment_idx"] = i
        out[idx] = segs
    return out


def _build_chunk_payload(
    chunks: list[dict],
    hq_by_chunk: dict[int, list[dict]],
    fast_by_chunk: dict[int, list[dict]],
) -> list[dict]:
    """Produce the per-chunk payload sent to the annotation page.

    We use HQ segments as the primary annotation unit when available
    (better baseline accuracy, segments line up with longer phrases).
    Fast segments are attached as a secondary list so the UI can offer
    Accept-Fast by finding the fast segment(s) whose time range overlaps
    the HQ segment. Falls back to fast if HQ has no coverage."""
    out: list[dict] = []
    for c in sorted(chunks, key=lambda x: x["idx"]):
        idx = c["idx"]
        hq_segs = hq_by_chunk.get(idx, [])
        fast_segs = fast_by_chunk.get(idx, [])
        primary = hq_segs if hq_segs else fast_segs
        primary_stream = "hq" if hq_segs else "fast"
        out.append({
            "idx": idx,
            "t_start": c["t_start"],
            "t_end": c["t_end"],
            "duration_s": c["duration_s"],
            "audio_segments": c.get("segments") or [],
            "primary_stream": primary_stream,
            "segments": [
                {
                    "segment_idx": s["segment_idx"],
                    "stream": s["stream"],
                    "t_start": round(s["t_start"], 3),
                    "t_end": round(s["t_end"], 3),
                    "text": s["text"],
                }
                for s in primary
            ],
            "alt_segments": [
                {
                    "stream": s["stream"],
                    "t_start": round(s["t_start"], 3),
                    "t_end": round(s["t_end"], 3),
                    "text": s["text"],
                }
                for s in (fast_segs if hq_segs else [])
            ],
        })
    return out


# ---------------------------------------------------------------------------
# Annotations state (read + write)
# ---------------------------------------------------------------------------

class AnnotationStore:
    """Append-only log of annotation events in ``<session_dir>/annotations.jsonl``.

    Latest row wins per ``(chunk_idx, segment_idx)``. Chunk-level rejection
    is stored with ``segment_idx=null``; the latest rejection-row state
    (rejected vs. un-rejected) wins per chunk. We never mutate or rewrite
    existing rows — the full edit history stays on disk for auditability."""

    def __init__(self, session_dir: str):
        self._path = os.path.join(session_dir, "annotations.jsonl")
        self._lock = threading.Lock()
        # Derived state, refreshed on every write so callers see the latest.
        self._segments: dict[tuple[int, int], dict] = {}
        self._chunks: dict[int, dict] = {}
        self._rebuild_state_locked()

    # --- state derivation -----------------------------------------------

    def _rebuild_state_locked(self) -> None:
        self._segments.clear()
        self._chunks.clear()
        if not os.path.exists(self._path):
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._ingest_locked(row)

    def _ingest_locked(self, row: dict) -> None:
        cidx = row.get("chunk_idx")
        sidx = row.get("segment_idx")
        if cidx is None:
            return
        if sidx is None:
            self._chunks[int(cidx)] = row
        else:
            self._segments[(int(cidx), int(sidx))] = row

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "segments": {
                    f"{c}_{s}": v for (c, s), v in self._segments.items()
                },
                "chunks": {
                    str(c): v for c, v in self._chunks.items()
                },
            }

    # --- writes ---------------------------------------------------------

    def _append(self, row: dict) -> None:
        row.setdefault("ts", datetime.now(timezone.utc).isoformat())
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
        self._ingest_locked(row)

    def put_segment(self, chunk_idx: int, segment_idx: int, body: dict) -> dict:
        text = body.get("text")
        status = body.get("status") or "edited"
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if status not in ("edited", "accepted_fast", "accepted_hq"):
            raise ValueError(f"unknown status: {status}")
        row = {
            "chunk_idx": int(chunk_idx),
            "segment_idx": int(segment_idx),
            "text": text,
            "status": status,
        }
        if body.get("notes"):
            row["notes"] = body["notes"]
        with self._lock:
            self._append(row)
        return row

    def put_chunk(self, chunk_idx: int, body: dict) -> dict:
        status = body.get("status") or "rejected"
        if status not in ("rejected", "unrejected"):
            raise ValueError(f"unknown chunk-status: {status}")
        row = {
            "chunk_idx": int(chunk_idx),
            "segment_idx": None,
            "status": status,
        }
        if body.get("notes"):
            row["notes"] = body["notes"]
        with self._lock:
            self._append(row)
        return row


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def _build_payload(session_dir: str) -> tuple[dict, str]:
    """Load everything once at startup. Returns (manifest_payload, audio_dir)."""
    chunks_for_status, audio_dir = _load_audio_payload(session_dir)
    if audio_dir is None or not chunks_for_status:
        raise SystemExit(
            f"no audio manifest at {session_dir}/audio/manifest.jsonl; "
            f"annotation needs per-chunk WAVs (re-record with --save-audio)"
        )
    hq, fast = _load_segments(session_dir)
    hq_by_chunk = _assign_segments_to_chunks(chunks_for_status, hq)
    fast_by_chunk = _assign_segments_to_chunks(chunks_for_status, fast)
    chunk_payload = _build_chunk_payload(chunks_for_status, hq_by_chunk, fast_by_chunk)
    return {"chunks": chunk_payload}, audio_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Annotation-mode dashboard for a recorded session.",
    )
    parser.add_argument("session_dir", help="Path to the session transcript directory.")
    parser.add_argument("--port", type=int, default=0, help="Widget port (0 = ephemeral).")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        print(f"not a directory: {session_dir}", file=sys.stderr)
        return 2

    payload, audio_dir = _build_payload(session_dir)
    store = AnnotationStore(session_dir)

    def annotation_provider() -> dict:
        return {**payload, "state": store.snapshot()}

    def annotation_put_segment(c: int, s: int, body: dict) -> dict:
        return store.put_segment(c, s, body)

    def annotation_put_chunk(c: int, body: dict) -> dict:
        return store.put_chunk(c, body)

    def status_provider() -> dict:
        return {
            "mode": "annotate",
            "session_dir": session_dir,
            "chunks_total": len(payload["chunks"]),
        }

    server = widget_module.StatusServer(
        snapshot_provider=status_provider,
        audio_dir=audio_dir,
        annotation_provider=annotation_provider,
        annotation_put_segment=annotation_put_segment,
        annotation_put_chunk=annotation_put_chunk,
    )
    host, port = server.start(host=args.host, port=args.port)
    url = f"http://{host}:{port}/annotate"
    print(f"annotation server: {url}")
    print(f"session: {session_dir}")
    print(f"chunks:  {len(payload['chunks'])}")
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(60.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
