"""Append-only annotations.jsonl reader/writer.

Schema (one row per save)::

  {
    "ts": <unix ms>,
    "chunk_idx": int,
    "segment_id": str | None,    # None = chunk-level action (reject)
    "text": str | None,
    "status": "edited" | "accepted_fast" | "accepted_hq" | "rejected",
    "source_text_fast": str | None,
    "source_text_hq": str | None,
    "notes": str | None,
  }

Last write per ``(chunk_idx, segment_id)`` wins on read. A chunk-level
reject is ``segment_id=None, status="rejected"``; saving any segment-level
row in that chunk does NOT clear the chunk reject — the user must
explicitly un-reject (write another chunk-level row with a non-rejected
status, or the special status ``"unrejected"`` which we treat as a clear).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


VALID_SEGMENT_STATUSES = {"edited", "accepted_fast", "accepted_hq"}
VALID_CHUNK_STATUSES = {"rejected", "unrejected"}


@dataclass
class Annotation:
    ts: int
    chunk_idx: int
    segment_id: Optional[str]   # None = chunk-level row
    text: Optional[str]
    status: str
    source_text_fast: Optional[str] = None
    source_text_hq: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AnnotationStore:
    """Reads + appends to ``<session_dir>/annotations.jsonl``."""

    def __init__(self, session_dir: str | Path) -> None:
        self._path = Path(session_dir) / "annotations.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------

    def append_segment(
        self,
        *,
        chunk_idx: int,
        segment_id: str,
        text: str,
        status: str,
        source_text_fast: Optional[str] = None,
        source_text_hq: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Annotation:
        if status not in VALID_SEGMENT_STATUSES:
            raise ValueError(
                f"invalid segment status: {status!r} "
                f"(expected one of {sorted(VALID_SEGMENT_STATUSES)})"
            )
        ann = Annotation(
            ts=int(time.time() * 1000),
            chunk_idx=int(chunk_idx),
            segment_id=str(segment_id),
            text=str(text),
            status=status,
            source_text_fast=source_text_fast,
            source_text_hq=source_text_hq,
            notes=notes,
        )
        self._append(ann)
        return ann

    def append_chunk(
        self,
        *,
        chunk_idx: int,
        status: str,
        notes: Optional[str] = None,
    ) -> Annotation:
        if status not in VALID_CHUNK_STATUSES:
            raise ValueError(
                f"invalid chunk status: {status!r} "
                f"(expected one of {sorted(VALID_CHUNK_STATUSES)})"
            )
        ann = Annotation(
            ts=int(time.time() * 1000),
            chunk_idx=int(chunk_idx),
            segment_id=None,
            text=None,
            status=status,
            notes=notes,
        )
        self._append(ann)
        return ann

    def _append(self, ann: Annotation) -> None:
        line = json.dumps(ann.to_dict(), ensure_ascii=False) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)

    # ------------------------------------------------------------------

    def resolved_state(self) -> dict:
        """Return ``{segments: {(chunk_idx, segment_id): row},
                     rejected_chunks: set[int]}`` with last-write-wins.

        Keys in the returned ``segments`` dict are JSON-friendly tuples
        but represented as ``f"{chunk_idx}|{segment_id}"`` strings so the
        whole thing can be JSON-encoded directly for the UI.
        """
        segments: dict[str, dict] = {}
        rejected: set[int] = set()
        if not self._path.exists():
            return {"segments": segments, "rejected_chunks": sorted(rejected)}
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("segment_id") is None:
                    cidx = int(r["chunk_idx"])
                    if r.get("status") == "rejected":
                        rejected.add(cidx)
                    elif r.get("status") == "unrejected":
                        rejected.discard(cidx)
                else:
                    key = f"{r['chunk_idx']}|{r['segment_id']}"
                    segments[key] = r
        return {
            "segments": segments,
            "rejected_chunks": sorted(rejected),
        }
