"""Load a session directory and build the per-chunk annotation view.

Inputs read from ``<session_dir>/``:
  - ``audio/manifest.json`` — list of WAV chunks with accepted_ms ranges
  - ``fast_segments.jsonl`` / ``hq_segments.jsonl`` — segments with words

Output: a list of ``ChunkView`` objects, one per WAV chunk, each carrying
its audio metadata and the segments that overlap its accepted_ms range,
with boundary-straddling segments split on word midpoints.

Segment id scheme
-----------------
``<stream>:<seq>`` for an original segment. After boundary splitting,
each part gets a ``:p<n>`` suffix where n is the chunk-order part
(``p0`` is the earliest chunk piece, ``p1`` the next, etc.). The id is
stable across re-loads as long as the source JSONL doesn't change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class _Word:
    text: str
    start_accepted_ms: int
    end_accepted_ms: int
    probability: float = 0.0


@dataclass
class SegmentView:
    """One segment as it appears in a chunk view.

    ``boundary`` is non-None when this is a piece of a segment that was
    split because the segment straddled one or more chunk boundaries.
    Values:
      - ``"trailing"``: the piece is the *end* of a segment that began
        in this chunk and continues into a later chunk.
      - ``"leading"``: the piece is the *start* of a segment whose body
        continues from an earlier chunk.
      - ``"middle"``: rare — segment spans more than one full chunk.
    Even when a segment is fully inside the chunk's range we still set
    ``boundary = None``; the UI uses that to skip the review highlight.
    """

    segment_id: str             # e.g. "fast:1234" or "fast:1234:p1"
    stream: str                 # "fast" | "hq"
    text: str
    start_accepted_ms: int
    end_accepted_ms: int
    boundary: Optional[str]     # None | "leading" | "trailing" | "middle"
    words: list[_Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ChunkView:
    chunk_idx: int
    file: str                   # relative to audio/, e.g. "chunk_0000.wav"
    start_accepted_ms: int
    end_accepted_ms: int
    duration_ms: int
    segments: list[SegmentView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_idx": self.chunk_idx,
            "file": self.file,
            "start_accepted_ms": self.start_accepted_ms,
            "end_accepted_ms": self.end_accepted_ms,
            "duration_ms": self.duration_ms,
            "segments": [s.to_dict() for s in self.segments],
        }


# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed rows; they shouldn't exist but be lenient.
                continue
    return rows


def _segments_from_rows(rows: list[dict], stream: str) -> list[dict]:
    """Normalize a segments.<stream>.jsonl into dicts with a stable
    segment_id and word list."""
    out: list[dict] = []
    for r in rows:
        seq = r.get("seq")
        if seq is None:
            continue
        words = [
            _Word(
                text=w.get("text", ""),
                start_accepted_ms=int(w.get("start_accepted_ms", 0)),
                end_accepted_ms=int(w.get("end_accepted_ms", 0)),
                probability=float(w.get("probability", 0.0) or 0.0),
            )
            for w in (r.get("words") or [])
        ]
        out.append({
            "segment_id": f"{stream}:{seq}",
            "stream": stream,
            "text": r.get("text", ""),
            "start_accepted_ms": int(r.get("content_accepted_ms_start", 0)),
            "end_accepted_ms": int(r.get("content_accepted_ms_end", 0)),
            "words": words,
        })
    return out


def _word_midpoint(w: _Word) -> int:
    return (w.start_accepted_ms + w.end_accepted_ms) // 2


def _assign_pieces(
    seg: dict, chunks: list[ChunkView],
) -> list[tuple[int, SegmentView]]:
    """Partition a segment into pieces, one per chunk it touches.

    Returns ``[(chunk_idx, SegmentView), ...]``.
    If the segment is fully inside one chunk, returns a single piece
    with ``boundary=None``. Otherwise, words are partitioned by midpoint
    into the chunks they belong to, and each piece is flagged
    ``"leading"`` / ``"trailing"`` / ``"middle"`` depending on its
    position in the segment.
    """
    if not chunks:
        return []
    words: list[_Word] = seg["words"]

    # Which chunk a given ms belongs to, or None if outside every chunk.
    # Segments emitted after the audio persister stopped (e.g. on shutdown
    # before the next chunk roll) sometimes overshoot the last chunk's
    # end_accepted_ms — those have no audio backing and must be dropped.
    def find_chunk(ms: int) -> Optional[int]:
        for c in chunks:
            if c.start_accepted_ms <= ms < c.end_accepted_ms:
                return c.chunk_idx
        return None

    # If no words (rare — e.g. silent segment), use start/end midpoints.
    if not words:
        midpoint = (seg["start_accepted_ms"] + seg["end_accepted_ms"]) // 2
        idx = find_chunk(midpoint)
        if idx is None:
            return []
        return [(
            idx,
            SegmentView(
                segment_id=seg["segment_id"],
                stream=seg["stream"],
                text=seg["text"],
                start_accepted_ms=seg["start_accepted_ms"],
                end_accepted_ms=seg["end_accepted_ms"],
                boundary=None,
                words=[],
            ),
        )]

    # Group consecutive words by chunk index. Words outside every chunk
    # are dropped (no audio to back them).
    groups: list[tuple[int, list[_Word]]] = []
    n_dropped = 0
    for w in words:
        idx = find_chunk(_word_midpoint(w))
        if idx is None:
            n_dropped += 1
            continue
        if groups and groups[-1][0] == idx:
            groups[-1][1].append(w)
        else:
            groups.append((idx, [w]))

    if not groups:
        return []

    if len(groups) == 1:
        idx, group_words = groups[0]
        if n_dropped == 0:
            # Fully inside one chunk; preserve original text & range.
            return [(
                idx,
                SegmentView(
                    segment_id=seg["segment_id"],
                    stream=seg["stream"],
                    text=seg["text"],
                    start_accepted_ms=seg["start_accepted_ms"],
                    end_accepted_ms=seg["end_accepted_ms"],
                    boundary=None,
                    words=group_words,
                ),
            )]
        # Some words were dropped — derive text & range from survivors.
        # No chunk-internal boundary, but flag for review since some
        # content was lost.
        text = "".join(w.text for w in group_words).strip()
        return [(
            idx,
            SegmentView(
                segment_id=seg["segment_id"],
                stream=seg["stream"],
                text=text,
                start_accepted_ms=group_words[0].start_accepted_ms,
                end_accepted_ms=group_words[-1].end_accepted_ms,
                boundary="trailing",   # tail of the segment was outside
                words=group_words,
            ),
        )]

    # Multi-chunk: split into pieces.
    pieces: list[tuple[int, SegmentView]] = []
    n = len(groups)
    for i, (idx, group_words) in enumerate(groups):
        if i == 0:
            boundary = "trailing"   # last words of segment in this chunk
        elif i == n - 1:
            boundary = "leading"    # first words of segment in this chunk
        else:
            boundary = "middle"
        piece_id = f"{seg['segment_id']}:p{i}"
        text = "".join(w.text for w in group_words).strip()
        if not text:
            continue
        pieces.append((
            idx,
            SegmentView(
                segment_id=piece_id,
                stream=seg["stream"],
                text=text,
                start_accepted_ms=group_words[0].start_accepted_ms,
                end_accepted_ms=group_words[-1].end_accepted_ms,
                boundary=boundary,
                words=group_words,
            ),
        ))
    return pieces


# ---------------------------------------------------------------------------


def load_chunks(session_dir: str | Path) -> list[ChunkView]:
    """Build the per-chunk annotation view for a session.

    Raises ``FileNotFoundError`` if the manifest is missing or empty.
    """
    root = Path(session_dir)
    manifest_path = root / "audio" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_rows = manifest.get("chunks") or []
    if not chunk_rows:
        raise FileNotFoundError(f"manifest has no chunks: {manifest_path}")

    chunks: list[ChunkView] = []
    for i, r in enumerate(chunk_rows):
        chunks.append(ChunkView(
            chunk_idx=i,
            file=r["file"],
            start_accepted_ms=int(r["start_accepted_ms"]),
            end_accepted_ms=int(r["end_accepted_ms"]),
            duration_ms=int(r.get(
                "duration_ms",
                int(r["end_accepted_ms"]) - int(r["start_accepted_ms"]),
            )),
        ))

    # Load segments from both streams and merge HQ-preferred:
    # take all HQ segments; drop fast segments whose [start,end] overlaps
    # any HQ range. Mirrors TranscriptStreamManager.get_canonical_tail —
    # the user wants the annotation source to match the canonical merge.
    fast_segs = _segments_from_rows(
        _read_jsonl(root / "fast_segments.jsonl"), "fast",
    )
    hq_segs = _segments_from_rows(
        _read_jsonl(root / "hq_segments.jsonl"), "hq",
    )
    hq_ranges = [(s["start_accepted_ms"], s["end_accepted_ms"]) for s in hq_segs]

    def _overlaps_hq(s: dict) -> bool:
        for a, b in hq_ranges:
            if s["end_accepted_ms"] > a and s["start_accepted_ms"] < b:
                return True
        return False

    kept_fast = [s for s in fast_segs if not _overlaps_hq(s)]
    all_segs = sorted(
        kept_fast + hq_segs,
        key=lambda s: (s["start_accepted_ms"], 0 if s["stream"] == "fast" else 1),
    )

    # Map segments to chunks.
    chunks_by_idx = {c.chunk_idx: c for c in chunks}
    for seg in all_segs:
        for idx, piece in _assign_pieces(seg, chunks):
            c = chunks_by_idx.get(idx)
            if c is None:
                continue
            c.segments.append(piece)

    # Sort each chunk's segments by start time.
    for c in chunks:
        c.segments.sort(key=lambda s: (s.start_accepted_ms,
                                       0 if s.stream == "fast" else 1))

    return chunks
