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

Discovery:
    Inputs can be individual session directories OR parent directories.
    A parent dir (e.g. ``transcripts/``) is expanded to its immediate
    children that look like session dirs — i.e. contain
    ``stream_fast.jsonl``, ``stream_hq.jsonl``, or ``audio/manifest.jsonl``.
    Older sessions with only ``debug_events.jsonl`` (pre-refactor format)
    are skipped — this builder requires the per-stream JSONL files.

Usage:
    # Inventory: see what's annotated and what would be usable.
    python tools/build_dataset.py transcripts/ --list [--mode overlay|strict|hybrid]

    # Build: writes train.jsonl + validation.jsonl into <out_dir>.
    python tools/build_dataset.py transcripts/ --out <out_dir> \\
        [--mode overlay|strict|hybrid] \\
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


# ---------------------------------------------------------------------------
# Discovery + capability probe
# ---------------------------------------------------------------------------

def _looks_like_session_dir(path: str) -> bool:
    """A session dir is something containing per-stream JSONL or an audio
    manifest. Older sessions may have only ``debug_events.jsonl`` —
    those are legacy and not usable by this dataset builder."""
    if not os.path.isdir(path):
        return False
    for name in ("stream_fast.jsonl", "stream_hq.jsonl"):
        if os.path.isfile(os.path.join(path, name)):
            return True
    if os.path.isfile(os.path.join(path, "audio", "manifest.jsonl")):
        return True
    return False


def _discover_sessions(paths: list[str]) -> list[str]:
    """Expand each input path:

    - If it's already a session dir, take it as-is.
    - If it's a directory whose immediate children include any session
      dirs, take all those children (one level of recursion).
    - Otherwise drop it with a warning.
    """
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        p_abs = os.path.abspath(p)
        if p_abs in seen:
            continue
        if _looks_like_session_dir(p_abs):
            out.append(p_abs)
            seen.add(p_abs)
            continue
        if os.path.isdir(p_abs):
            for entry in sorted(os.listdir(p_abs)):
                sub = os.path.join(p_abs, entry)
                if sub in seen:
                    continue
                if _looks_like_session_dir(sub):
                    out.append(sub)
                    seen.add(sub)
            continue
        print(f"warn: not a session or parent dir, skipping: {p}", file=sys.stderr)
    return out


def _probe_session(session_dir: str) -> dict:
    """Cheap inventory of one session — what it has, how annotated it is,
    why it would be skipped (if so). Used by ``--list``."""
    info: dict = {
        "session": os.path.basename(os.path.normpath(session_dir)),
        "path": session_dir,
        "has_audio_manifest": False,
        "wav_count": 0,
        "has_fast_segments": False,
        "has_hq_segments": False,
        "primary_stream": None,
        "primary_segments": 0,
        "annotations_exist": False,
        "annotated_segments": 0,
        "rejected_chunks": 0,
        "chunks_in_manifest": 0,
        "fully_annotated_chunks": 0,
        "partially_annotated_chunks": 0,
        "untouched_chunks": 0,
        "reason": None,
    }

    audio_dir = os.path.join(session_dir, "audio")
    manifest_path = os.path.join(audio_dir, "manifest.jsonl")
    manifest = _read_jsonl(manifest_path)
    if manifest:
        info["has_audio_manifest"] = True
        info["chunks_in_manifest"] = len(manifest)
        info["wav_count"] = sum(
            1 for r in manifest
            if os.path.isfile(os.path.join(audio_dir, f"chunk_{int(r['idx']):03d}.wav"))
        )
        durations = [float(r.get("duration_s", 0.0)) for r in manifest]
        if durations:
            info["shortest_chunk_s"] = min(durations)
            info["longest_chunk_s"] = max(durations)

    hq = _load_segments(session_dir, "hq")
    fast = _load_segments(session_dir, "fast")
    info["has_hq_segments"] = bool(hq)
    info["has_fast_segments"] = bool(fast)
    primary = hq if hq else fast
    info["primary_stream"] = "hq" if hq else ("fast" if fast else None)
    info["primary_segments"] = len(primary)

    ann_path = os.path.join(session_dir, "annotations.jsonl")
    info["annotations_exist"] = os.path.isfile(ann_path)
    seg_ann, chunk_ann = _load_annotation_state(session_dir)
    info["annotated_segments"] = len(seg_ann)
    info["rejected_chunks"] = sum(
        1 for v in chunk_ann.values() if v.get("status") == "rejected"
    )

    # Per-chunk annotation coverage.
    if info["has_audio_manifest"] and primary:
        chunks_list = [
            {"idx": int(r["idx"]), "t_start": float(r.get("t_start", 0.0)),
             "t_end": float(r.get("t_end", 0.0))}
            for r in manifest
        ]
        by_chunk = _assign_segments_to_chunks(chunks_list, primary)
        for c in chunks_list:
            cidx = c["idx"]
            segs = by_chunk.get(cidx, [])
            if not segs:
                info["untouched_chunks"] += 1
                continue
            n_ann = sum(1 for s in segs if (cidx, s["segment_idx"]) in seg_ann)
            if n_ann == 0:
                info["untouched_chunks"] += 1
            elif n_ann < len(segs):
                info["partially_annotated_chunks"] += 1
            else:
                info["fully_annotated_chunks"] += 1

    if not info["has_audio_manifest"]:
        info["reason"] = "no audio/manifest.jsonl (record with --save-audio)"
    elif info["wav_count"] == 0:
        info["reason"] = "manifest exists but no chunk_NNN.wav files"
    elif not primary:
        info["reason"] = "no segment events in stream_fast.jsonl / stream_hq.jsonl"
    return info


def _format_inventory(infos: list[dict], mode: str, min_s: float, max_s: float, per_segment: bool) -> str:
    """One row per session, columns describing capability. ``rows`` is
    the exact count this session would emit at build time (accounts for
    duration filter and mode-specific annotation predicate)."""
    if not infos:
        return "no sessions found.\n"
    headers = ("session", "audio", "segs", "ann", "full", "part", "untch", "rej", "rows", "note")
    out_rows: list[tuple] = []
    usable = 0
    total_rows = 0
    for info in infos:
        # The exact-count call. Cheaper than it looks: just re-parses
        # already-loaded JSONL paths to count, no audio decoding.
        produced = _session_rows(info["path"], mode, min_s, max_s, per_segment=per_segment)
        n = len(produced)
        total_rows += n
        if info["reason"]:
            note = info["reason"]
        elif n == 0:
            # Distinguish "duration filter ate everything" from "nothing annotated".
            longest = info.get("longest_chunk_s", 0.0)
            shortest = info.get("shortest_chunk_s", 0.0)
            unit = "segments" if per_segment else "chunks"
            if (
                not per_segment
                and info["chunks_in_manifest"] > 0
                and (longest > max_s or shortest < min_s)
            ):
                note = f"all chunks outside duration filter [{min_s}, {max_s}]s (range {shortest:.1f}–{longest:.1f}s)"
            else:
                note = f"no qualifying {unit} under mode={mode}"
        else:
            note = ""
            usable += 1
        out_rows.append((
            info["session"],
            f"{info['wav_count']}/{info['chunks_in_manifest']}",
            f"{info['primary_segments']}{'(hq)' if info['primary_stream']=='hq' else '(fast)' if info['primary_stream']=='fast' else ''}",
            str(info["annotated_segments"]),
            str(info["fully_annotated_chunks"]),
            str(info["partially_annotated_chunks"]),
            str(info["untouched_chunks"]),
            str(info["rejected_chunks"]),
            str(n),
            note,
        ))
    widths = [max(len(h), max(len(r[i]) for r in out_rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    for r in out_rows:
        lines.append(fmt.format(*r))
    lines.append("")
    lines.append(
        f"mode={mode}  unit={'segment' if per_segment else 'chunk'}  "
        f"duration filter=[{min_s}, {max_s}]s  "
        f"sessions={len(infos)}  usable={usable}  total_rows={total_rows}"
    )
    return "\n".join(lines) + "\n"


def _session_rows(
    session_dir: str,
    mode: str,
    min_s: float,
    max_s: float,
    per_segment: bool = False,
) -> list[dict]:
    """Returns one row per included unit (chunk in default mode, segment
    in ``per_segment`` mode). Each row carries enough metadata for the
    fine-tune script to slice audio at load time."""
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

    if per_segment:
        for c in sorted(chunks, key=lambda x: x["idx"]):
            cidx = c["idx"]
            if chunk_ann.get(cidx, {}).get("status") == "rejected":
                continue
            if not os.path.isfile(c["audio_path"]):
                continue
            segs = by_chunk.get(cidx, [])
            if not segs:
                continue
            # In hybrid mode, the qualifying signal is "this chunk has
            # at least one annotated segment" — skip the whole chunk
            # otherwise. Strict applies per-segment below.
            if mode == "hybrid":
                if not any((cidx, s["segment_idx"]) in seg_ann for s in segs):
                    continue
            for s in segs:
                sidx = s["segment_idx"]
                ann = seg_ann.get((cidx, sidx))
                if mode == "strict" and ann is None:
                    continue
                text = (ann["text"] if ann else s["text"]).strip()
                if not text:
                    continue
                # Segment audio range is session-absolute; convert to
                # WAV-relative by subtracting chunk t_start. Clamp to
                # [0, chunk_duration] to be safe against off-by-epsilon
                # at boundaries.
                start = max(0.0, s["t_start"] - c["t_start"])
                end = min(c["duration_s"], s["t_end"] - c["t_start"])
                if end <= start:
                    continue
                dur = end - start
                if dur < min_s or dur > max_s:
                    continue
                rows.append({
                    "audio": c["audio_path"],
                    "audio_start_s": round(start, 3),
                    "audio_end_s": round(end, 3),
                    "text": text,
                    "session": session_id,
                    "chunk_idx": cidx,
                    "segment_idx": sidx,
                    "duration_s": round(dur, 3),
                    "annotated": ann is not None,
                })
        return rows

    # Per-chunk path (original behavior).
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
    p.add_argument(
        "session_dirs", nargs="+",
        help=(
            "One or more session directories. May also be a parent directory "
            "(e.g. `transcripts/`) — its immediate children are scanned for "
            "session-shaped subdirectories."
        ),
    )
    p.add_argument(
        "--list", action="store_true", dest="list_only",
        help=(
            "Dry-run: print an inventory of each discovered session "
            "(audio, segments, annotation coverage, would-it-be-usable for "
            "the chosen --mode) and exit without writing any files."
        ),
    )
    p.add_argument(
        "--out", default=None,
        help="Output directory for train.jsonl + validation.jsonl (required unless --list).",
    )
    p.add_argument(
        "--mode", choices=("overlay", "strict", "hybrid"), default="overlay",
        help="Inclusion policy (default: overlay).",
    )
    p.add_argument(
        "--per-segment", action="store_true",
        help=(
            "Emit one row per HQ segment (or fast fallback) instead of "
            "one row per chunk. The row carries `audio_start_s` + "
            "`audio_end_s` so finetune_whisper.py slices the WAV in "
            "memory at load time. This is the right mode when chunks "
            "are longer than Whisper's 30s encoder window (every chunk "
            "longer than 30s would otherwise be truncated)."
        ),
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of sessions to send to validation (default: 0.1).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the session split (default: 42).")
    p.add_argument("--min-s", type=float, default=1.0, help="Drop chunks shorter than this (default: 1.0s).")
    p.add_argument("--max-s", type=float, default=30.0, help="Drop chunks longer than this (default: 30.0s).")
    args = p.parse_args(argv)

    if not args.list_only and not args.out:
        print("--out is required (unless --list)", file=sys.stderr)
        return 2

    sessions = _discover_sessions(args.session_dirs)
    if not sessions:
        print("no session directories discovered", file=sys.stderr)
        return 1

    if args.list_only:
        infos = [_probe_session(sd) for sd in sessions]
        print(_format_inventory(infos, args.mode, args.min_s, args.max_s, args.per_segment))
        return 0

    unit = "segments" if args.per_segment else "chunks"
    all_rows: list[dict] = []
    for sd in sessions:
        rows = _session_rows(sd, args.mode, args.min_s, args.max_s, per_segment=args.per_segment)
        # Annotate the reason when a session contributes 0 rows so the
        # operator can tell "not yet annotated" from "no audio".
        if not rows:
            probe = _probe_session(sd)
            reason = probe["reason"] or f"no qualifying {unit} under mode={args.mode}"
            print(f"{probe['session']}: 0 {unit}  [{reason}]")
            continue
        all_rows.extend(rows)
        print(f"{os.path.basename(sd)}: {len(rows)} {unit}")

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
