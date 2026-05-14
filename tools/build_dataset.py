#!/usr/bin/env python3
"""Build a Whisper fine-tuning dataset from annotated session directories.

Reads each session's audio manifest, per-stream segment events, and
``annotations.jsonl`` overlay, and emits one training row per included
chunk: ``{audio: <abs_wav_path>, text: <ground_truth_full_text>}``.

Output: ``<out_dir>/train.jsonl`` and ``<out_dir>/validation.jsonl``,
loadable with::

    from datasets import load_dataset
    ds = load_dataset("json", data_files={
        "train": f"{out_dir}/train.jsonl",
        "validation": f"{out_dir}/validation.jsonl",
    })

Train/val split is **by session** (a whole session goes to one split),
not by chunk — splitting within a session would leak the user's voice
+ vocabulary across train and eval, inflating apparent WER gains.

Inclusion modes (``--mode``):

- ``overlay`` (default) — include every non-rejected chunk. For each
  segment, use the annotation text if one exists, else fall back to the
  original segment text from ``stream_hq.jsonl`` (or ``stream_fast.jsonl``
  if HQ has no coverage). Maximizes data volume. Risk: un-reviewed
  segments train the model on Whisper's own (possibly wrong) outputs,
  reinforcing existing errors. Good for high-volume fine-tunes where
  most of the data is already correct.
- ``strict`` — include a chunk only if **every** segment has an
  annotation row (status ``edited``/``accepted_fast``/``accepted_hq``).
  Rejected chunks dropped, un-reviewed chunks dropped. Every label is
  human-verified. Best for quality but requires reviewing every segment.
- ``hybrid`` — include a chunk if **at least one** segment is annotated
  (signal: "user reviewed this chunk"); within included chunks,
  un-annotated segments fall back to original text. Compromise between
  the two.

Length filter (``--min-s`` / ``--max-s``) drops chunks outside the
Whisper-recommended 1–30s range by default.

Usage:
    python tools/build_dataset.py <session_dir> [<session_dir> ...] \\
        --out <out_dir> [--mode overlay|strict|hybrid] \\
        [--val-fraction 0.1] [--seed 42] [--min-s 1.0] [--max-s 30.0]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
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


def _load_segments(session_dir: str, stream: str) -> list[dict]:
    rows: list[dict] = []
    path = os.path.join(session_dir, f"stream_{stream}.jsonl")
    for row in _read_jsonl(path):
        if row.get("kind") != "segment":
            continue
        data = row.get("data") or {}
        if data.get("stream") != stream:
            continue
        rows.append({
            "t_start": float(data.get("start", 0.0)),
            "t_end": float(data.get("end", 0.0)),
            "text": (data.get("text") or "").strip(),
        })
    rows.sort(key=lambda r: r["t_start"])
    return rows


def _load_annotation_state(session_dir: str) -> tuple[dict, dict]:
    """Returns (segments_state, chunks_state). Latest row wins per key."""
    path = os.path.join(session_dir, "annotations.jsonl")
    segs: dict[tuple[int, int], dict] = {}
    chunks: dict[int, dict] = {}
    for row in _read_jsonl(path):
        cidx = row.get("chunk_idx")
        sidx = row.get("segment_idx")
        if cidx is None:
            continue
        if sidx is None:
            chunks[int(cidx)] = row
        else:
            segs[(int(cidx), int(sidx))] = row
    return segs, chunks


# ---------------------------------------------------------------------------
# Per-chunk reconstruction
# ---------------------------------------------------------------------------

def _assign_segments_to_chunks(chunks: list[dict], segs: list[dict]) -> dict[int, list[dict]]:
    """Bucket segments under chunks by ``t_start`` (matches
    annotate_session.py's bucketing, so segment_idx values line up)."""
    if not chunks:
        return {}
    sorted_chunks = sorted(chunks, key=lambda c: c["t_start"])
    buckets: dict[int, list[dict]] = {c["idx"]: [] for c in sorted_chunks}
    for seg in segs:
        target = sorted_chunks[0]["idx"]
        for c in sorted_chunks:
            if seg["t_start"] >= c["t_start"]:
                target = c["idx"]
            else:
                break
        buckets[target].append(seg)
    for idx, lst in buckets.items():
        lst.sort(key=lambda s: s["t_start"])
        for i, s in enumerate(lst):
            s["segment_idx"] = i
    return buckets


def _build_row_for_chunk(
    chunk: dict,
    primary_segs: list[dict],
    seg_annotations: dict[tuple[int, int], dict],
    mode: str,
) -> Optional[str]:
    """Build the ground-truth text for a chunk, or return None if the
    chunk should be excluded under ``mode``."""
    cidx = chunk["idx"]
    if not primary_segs:
        # No segments under this chunk — nothing to train on.
        return None

    annotated_count = sum(
        1 for s in primary_segs
        if (cidx, s["segment_idx"]) in seg_annotations
    )

    if mode == "strict" and annotated_count < len(primary_segs):
        return None
    if mode == "hybrid" and annotated_count == 0:
        return None

    parts: list[str] = []
    for s in primary_segs:
        ann = seg_annotations.get((cidx, s["segment_idx"]))
        text = (ann["text"] if ann else s["text"]).strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    return " ".join(parts)


def _session_rows(session_dir: str, mode: str, min_s: float, max_s: float) -> list[dict]:
    """Returns one ``{audio, text, session, chunk_idx}`` row per included chunk."""
    manifest = _read_jsonl(os.path.join(session_dir, "audio", "manifest.jsonl"))
    if not manifest:
        return []
    chunks = [
        {
            "idx": int(r["idx"]),
            "t_start": float(r.get("t_start", 0.0)),
            "t_end": float(r.get("t_end", 0.0)),
            "duration_s": float(r.get("duration_s", 0.0)),
            "audio_path": os.path.abspath(
                os.path.join(session_dir, "audio", f"chunk_{int(r['idx']):03d}.wav")
            ),
        }
        for r in manifest
    ]
    hq = _load_segments(session_dir, "hq")
    fast = _load_segments(session_dir, "fast")
    seg_ann, chunk_ann = _load_annotation_state(session_dir)

    # Match annotate_session.py: HQ is primary when available, else fast.
    primary_stream = hq if hq else fast
    by_chunk = _assign_segments_to_chunks(chunks, primary_stream)

    rows: list[dict] = []
    session_id = os.path.basename(os.path.normpath(session_dir))
    for c in sorted(chunks, key=lambda x: x["idx"]):
        cidx = c["idx"]
        if chunk_ann.get(cidx, {}).get("status") == "rejected":
            continue
        if not os.path.isfile(c["audio_path"]):
            continue
        if c["duration_s"] < min_s or c["duration_s"] > max_s:
            continue
        text = _build_row_for_chunk(c, by_chunk.get(cidx, []), seg_ann, mode)
        if text is None:
            continue
        rows.append({
            "audio": c["audio_path"],
            "text": text,
            "session": session_id,
            "chunk_idx": cidx,
            "duration_s": c["duration_s"],
        })
    return rows


# ---------------------------------------------------------------------------
# Split + write
# ---------------------------------------------------------------------------

def _split_by_session(
    rows: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    sessions = sorted({r["session"] for r in rows})
    if not sessions:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val = max(1, int(round(len(sessions) * val_fraction))) if len(sessions) > 1 else 0
    val_sessions = set(sessions[:n_val])
    train, val = [], []
    for r in rows:
        (val if r["session"] in val_sessions else train).append(r)
    return train, val


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Inclusion modes:\n"
            "  overlay (default) — every non-rejected chunk; un-annotated\n"
            "                       segments fall back to original Whisper text.\n"
            "  strict             — only chunks where every segment has an\n"
            "                       annotation; every label is human-verified.\n"
            "  hybrid             — only chunks with at least one annotation;\n"
            "                       un-annotated segments fall back to original.\n"
        ),
    )
    p.add_argument("session_dirs", nargs="+", help="One or more session directories.")
    p.add_argument("--out", required=True, help="Output directory for train.jsonl + validation.jsonl.")
    p.add_argument(
        "--mode", choices=("overlay", "strict", "hybrid"), default="overlay",
        help="Chunk inclusion policy (default: overlay).",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of sessions to send to validation (default: 0.1).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the session split (default: 42).")
    p.add_argument("--min-s", type=float, default=1.0, help="Drop chunks shorter than this (default: 1.0s).")
    p.add_argument("--max-s", type=float, default=30.0, help="Drop chunks longer than this (default: 30.0s).")
    args = p.parse_args(argv)

    all_rows: list[dict] = []
    for sd in args.session_dirs:
        sd_abs = os.path.abspath(sd)
        if not os.path.isdir(sd_abs):
            print(f"skip (not a dir): {sd}", file=sys.stderr)
            continue
        rows = _session_rows(sd_abs, args.mode, args.min_s, args.max_s)
        print(f"{os.path.basename(sd_abs)}: {len(rows)} chunks")
        all_rows.extend(rows)

    if not all_rows:
        print("no rows produced — nothing to write", file=sys.stderr)
        return 1

    train, val = _split_by_session(all_rows, args.val_fraction, args.seed)
    out_dir = os.path.abspath(args.out)
    _write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    _write_jsonl(os.path.join(out_dir, "validation.jsonl"), val)

    print(f"\nwrote {out_dir}/train.jsonl       ({len(train)} rows)")
    print(f"wrote {out_dir}/validation.jsonl  ({len(val)} rows)")
    print(f"mode={args.mode}  sessions={len({r['session'] for r in all_rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
