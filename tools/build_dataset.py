"""Build a HuggingFace-style fine-tune dataset from one or more sessions.

Usage::

    python -m voice_dictation.tools.build_dataset \
        --out path/to/out_dir \
        voice_dictation/sessions/20260515_080929 \
        voice_dictation/sessions/...

Emits per-segment audio clips + a manifest. Annotations override the
original Whisper text per segment; rejected chunks are skipped.

Output layout::

    out_dir/
      manifest.jsonl        # all clips
      train.jsonl           # train split (by session)
      val.jsonl             # val split (by session)
      audio/<session>/seg_NNNN.wav

Clips outside ``[min_seconds, max_seconds]`` are dropped (Whisper trains
on ≤ 30 s clips; very short clips contribute little).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The annotator package only depends on the stdlib (no torch / soundfile
# / faster_whisper imports), so importing it here is cheap.
from voice_dictation.annotator.loader import ChunkView, SegmentView, load_chunks
from voice_dictation.annotator.storage import AnnotationStore


logger = logging.getLogger(__name__)


# Default whisper-friendly clip-duration band.
DEFAULT_MIN_SECONDS = 1.0
DEFAULT_MAX_SECONDS = 30.0


@dataclass
class Clip:
    session: str
    chunk_idx: int
    segment_id: str
    text: str
    status: str           # "edited" | "accepted_fast" | "accepted_hq" | "original"
    boundary: Optional[str]
    audio_rel: str        # relative to out_dir
    duration_ms: int
    start_accepted_ms: int
    end_accepted_ms: int


# ---------------------------------------------------------------------------


def _slice_wav(
    src_wav: Path,
    dst_wav: Path,
    *,
    chunk_start_ms: int,
    clip_start_ms: int,
    clip_end_ms: int,
) -> int:
    """Write a sliced WAV from ``src_wav`` covering
    ``[clip_start_ms, clip_end_ms]`` (relative to accepted_ms, where
    ``chunk_start_ms`` is the chunk's accepted_ms start). Returns the
    number of samples written."""
    with wave.open(str(src_wav), "rb") as r:
        n_channels = r.getnchannels()
        sampwidth = r.getsampwidth()
        framerate = r.getframerate()
        n_frames = r.getnframes()

        # Convert ms-since-chunk-start to frame index.
        start_frame = int((clip_start_ms - chunk_start_ms) * framerate / 1000)
        end_frame = int((clip_end_ms - chunk_start_ms) * framerate / 1000)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        if end_frame <= start_frame:
            return 0

        r.setpos(start_frame)
        frames = r.readframes(end_frame - start_frame)

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_wav), "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(frames)
    return end_frame - start_frame


def _resolve_segment_text(
    seg: SegmentView, ann_state: dict, chunk_idx: int,
) -> tuple[str, str]:
    """Return (text, status) for the segment.

    status: "edited" | "accepted_fast" | "accepted_hq" | "original"
    """
    key = f"{chunk_idx}|{seg.segment_id}"
    ann = ann_state["segments"].get(key)
    if ann is not None:
        return (ann.get("text") or "").strip(), ann["status"]
    return (seg.text or "").strip(), "original"


def _emit_clips_for_session(
    session_dir: Path,
    out_dir: Path,
    *,
    include_unannotated: bool,
    min_seconds: float,
    max_seconds: float,
    include_boundary: bool,
) -> list[Clip]:
    session_id = session_dir.name
    chunks = load_chunks(session_dir)
    store = AnnotationStore(session_dir)
    ann_state = store.resolved_state()
    rejected = set(ann_state["rejected_chunks"])

    out_clips: list[Clip] = []
    seg_counter = 0
    for chunk in chunks:
        if chunk.chunk_idx in rejected:
            continue
        src_wav = session_dir / "audio" / chunk.file
        if not src_wav.exists():
            logger.warning("missing audio %s — skipping chunk", src_wav)
            continue
        for seg in chunk.segments:
            if seg.boundary is not None and not include_boundary:
                continue
            text, status = _resolve_segment_text(seg, ann_state, chunk.chunk_idx)
            if not text:
                continue
            if not include_unannotated and status == "original":
                continue
            duration_ms = seg.end_accepted_ms - seg.start_accepted_ms
            if duration_ms < int(min_seconds * 1000):
                continue
            if duration_ms > int(max_seconds * 1000):
                continue

            seg_counter += 1
            rel = f"audio/{session_id}/seg_{seg_counter:04d}.wav"
            dst = out_dir / rel
            n = _slice_wav(
                src_wav, dst,
                chunk_start_ms=chunk.start_accepted_ms,
                clip_start_ms=seg.start_accepted_ms,
                clip_end_ms=seg.end_accepted_ms,
            )
            if n == 0:
                continue
            out_clips.append(Clip(
                session=session_id,
                chunk_idx=chunk.chunk_idx,
                segment_id=seg.segment_id,
                text=text,
                status=status,
                boundary=seg.boundary,
                audio_rel=rel,
                duration_ms=duration_ms,
                start_accepted_ms=seg.start_accepted_ms,
                end_accepted_ms=seg.end_accepted_ms,
            ))
    return out_clips


def _split_by_session(
    clips: list[Clip], val_fraction: float, seed: int,
) -> tuple[list[Clip], list[Clip]]:
    sessions = sorted({c.session for c in clips})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val = max(1, int(round(len(sessions) * val_fraction))) if val_fraction > 0 else 0
    val_sessions = set(sessions[:n_val])
    train, val = [], []
    for c in clips:
        (val if c.session in val_sessions else train).append(c)
    return train, val


def _write_manifest(path: Path, clips: list[Clip]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in clips:
            f.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Slice annotated sessions into a Whisper-fine-tune "
                    "dataset (audio + manifest).",
    )
    parser.add_argument("sessions", nargs="+",
                        help="one or more sessions/<ts>/ directories")
    parser.add_argument("--out", required=True,
                        help="output directory (will be created)")
    parser.add_argument("--include-unannotated", action="store_true",
                        help="emit segments without any annotation (label = "
                             "Whisper's original output). Default off — only "
                             "annotated / accepted segments are emitted.")
    parser.add_argument("--include-boundary", action="store_true",
                        help="emit segments flagged as boundary (default off "
                             "— boundary segments are partial transcriptions "
                             "and the time-trim may be inaccurate).")
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="fraction of SESSIONS held out for val (not clips)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_clips: list[Clip] = []
    n_sessions_ok = 0
    for s in args.sessions:
        session_dir = Path(s).resolve()
        if not session_dir.is_dir():
            print(f"warn: not a dir: {session_dir}", file=sys.stderr)
            continue
        try:
            clips = _emit_clips_for_session(
                session_dir, out_dir,
                include_unannotated=args.include_unannotated,
                min_seconds=args.min_seconds,
                max_seconds=args.max_seconds,
                include_boundary=args.include_boundary,
            )
        except FileNotFoundError as e:
            print(f"warn: {e}", file=sys.stderr)
            continue
        logger.info("%s → %d clips", session_dir.name, len(clips))
        all_clips.extend(clips)
        n_sessions_ok += 1

    if not all_clips:
        print("no clips emitted; nothing to write", file=sys.stderr)
        return 1

    _write_manifest(out_dir / "manifest.jsonl", all_clips)
    train, val = _split_by_session(all_clips, args.val_fraction, args.seed)
    _write_manifest(out_dir / "train.jsonl", train)
    _write_manifest(out_dir / "val.jsonl", val)

    total_dur_s = sum(c.duration_ms for c in all_clips) / 1000
    logger.info(
        "wrote %d clips from %d sessions (%.1fs audio) → train=%d val=%d at %s",
        len(all_clips), n_sessions_ok, total_dur_s, len(train), len(val), out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
