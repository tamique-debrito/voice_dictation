"""Export plain-text transcripts from finalized voice_dictation sessions.

Reads one or more session directories the same way the annotator UI does
(HQ-preferred merge, boundary splits, last-write-wins annotations) and
emits one ``.txt`` per session. Useful for pasting transcripts into
external analysis tools or feeding into a local LLM.

Each output file is prefaced with a header containing the session name,
its recorded date/time (parsed from the directory name), and the source
audio path when the session was regenerated from existing audio.

Usage::

    # All regenerated sessions:
    python -m voice_dictation.tools.export_transcripts \\
        --root voice_dictation/regenerated_sessions \\
        --out  voice_dictation/transcripts_text

    # Specific sessions:
    python -m voice_dictation.tools.export_transcripts \\
        voice_dictation/sessions/20260515_224523 \\
        --out voice_dictation/transcripts_text

    # Date-filtered + combined into one file:
    python -m voice_dictation.tools.export_transcripts \\
        --root voice_dictation/sessions \\
        --since 2026-05-13 --until 2026-05-15 \\
        --combined --out voice_dictation/transcripts_text

Source selection:
  - When ``annotations.jsonl`` exists, edited / accepted text wins over
    the raw Whisper output. Rejected chunks are skipped.
  - When no annotation exists for a segment, the segment's own text
    (HQ-preferred) is used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..annotator.loader import ChunkView, load_chunks
from ..annotator.storage import AnnotationStore


# Matches the leading ``YYYYMMDD_HHMMSS`` stamp in both live session names
# (``20260515_224523``) and regenerated names
# (``regen_20260515_193718_p64371_20260514_160743_audio``).
_STAMP_RE = re.compile(r"(?:^|_)(\d{8})_(\d{6})")


@dataclass
class SessionMeta:
    path: Path
    name: str
    recorded_at: Optional[datetime]
    is_regenerated: bool
    source_audio: Optional[str]
    source_label: Optional[str]


def parse_session_stamp(name: str) -> Optional[datetime]:
    """Parse the leading YYYYMMDD_HHMMSS in a session dir name. Returns
    a naive (clock-only) datetime in UTC."""
    m = _STAMP_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(
            m.group(1) + m.group(2), "%Y%m%d%H%M%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def session_meta(session_dir: Path) -> SessionMeta:
    name = session_dir.name
    recorded_at = parse_session_stamp(name)
    regen_marker = session_dir / "regenerated.json"
    is_regen = regen_marker.exists()
    source_audio = source_label = None
    if is_regen:
        try:
            r = json.loads(regen_marker.read_text(encoding="utf-8"))
            source_audio = r.get("source_audio")
            source_label = r.get("source_label")
        except Exception:
            pass
    return SessionMeta(
        path=session_dir, name=name, recorded_at=recorded_at,
        is_regenerated=is_regen, source_audio=source_audio,
        source_label=source_label,
    )


def discover_sessions(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    return [
        child for child in sorted(root.iterdir())
        if child.is_dir() and (child / "audio" / "manifest.json").exists()
    ]


def filter_by_date(
    metas: list[SessionMeta],
    since: Optional[datetime],
    until: Optional[datetime],
) -> list[SessionMeta]:
    out: list[SessionMeta] = []
    for m in metas:
        if m.recorded_at is None:
            continue   # unstamped sessions get filtered out when dates are set
        if since and m.recorded_at < since:
            continue
        if until and m.recorded_at > until:
            continue
        out.append(m)
    return out


def render_transcript(
    chunks: list[ChunkView], resolved: dict, *, paragraph_gap_ms: int = 1500,
) -> str:
    """Concatenate segments into plain text. Rejected chunks are skipped.
    Inserts a blank line between segments when there's a large silence
    gap, so the result reads as paragraphs instead of a wall of text.

    Empty-edited segments (status="edited", text="") are dropped — those
    are hallucinations the annotator agent wiped out.
    """
    rejected: set[int] = set(resolved.get("rejected_chunks", []))
    seg_anns: dict[str, dict] = resolved.get("segments", {})

    out_lines: list[str] = []
    prev_end_ms: Optional[int] = None
    for c in chunks:
        if c.chunk_idx in rejected:
            prev_end_ms = c.end_accepted_ms
            continue
        for s in c.segments:
            key = f"{c.chunk_idx}|{s.segment_id}"
            ann = seg_anns.get(key)
            if ann is not None:
                text = ann.get("text") or ""
            else:
                text = s.text or ""
            text = text.strip()
            if not text:
                continue
            if (prev_end_ms is not None
                    and (s.start_accepted_ms - prev_end_ms) >= paragraph_gap_ms
                    and out_lines and out_lines[-1] != ""):
                out_lines.append("")
            out_lines.append(text)
            prev_end_ms = s.end_accepted_ms
    return "\n".join(out_lines).rstrip() + "\n"


def render_header(
    meta: SessionMeta, exported_at: datetime, *, full: bool = False,
) -> str:
    """Build the file header.

    Default (``full=False``) is minimal — just the recorded time. The
    rich metadata confuses LLMs into thinking it's content (e.g. the
    word "regenerated" in the header gets picked up as a topic). Use
    ``--full-header`` to opt back in for human-review exports.
    """
    if not full:
        if meta.recorded_at:
            return f"recorded: {meta.recorded_at.isoformat()}\n\n"
        return "recorded: (unknown)\n\n"
    lines = [f"# {meta.name}"]
    if meta.recorded_at:
        lines.append(f"recorded: {meta.recorded_at.isoformat()}")
    else:
        lines.append("recorded: (unknown)")
    if meta.is_regenerated:
        if meta.source_audio:
            lines.append(f"regenerated from: {meta.source_audio}")
        elif meta.source_label:
            lines.append(f"regenerated from: {meta.source_label}")
        else:
            lines.append("regenerated (source unknown)")
    lines.append(f"exported_at: {exported_at.isoformat()}")
    lines.append("")
    return "\n".join(lines) + "\n"


def export_session(
    meta: SessionMeta, out_dir: Path, exported_at: datetime,
    *, full_header: bool = False,
) -> tuple[Path, int]:
    """Write ``<out_dir>/<session_name>.txt`` and return (path, line_count)."""
    chunks = load_chunks(meta.path)
    resolved = AnnotationStore(meta.path).resolved_state()
    body = render_transcript(chunks, resolved)
    text = render_header(meta, exported_at, full=full_header) + body
    out_path = out_dir / f"{meta.name}.txt"
    out_path.write_text(text, encoding="utf-8")
    line_count = sum(1 for line in body.splitlines() if line.strip())
    return out_path, line_count


def export_combined(
    metas: list[SessionMeta], out_path: Path, exported_at: datetime,
    *, full_header: bool = False,
) -> int:
    """Write a single file with all sessions concatenated, separated by
    a header per session. Returns the total non-blank line count."""
    parts: list[str] = []
    if full_header:
        parts.append(
            f"# Combined transcript export\n"
            f"sessions: {len(metas)}\n"
            f"exported_at: {exported_at.isoformat()}\n\n"
        )
    total_lines = 0
    for meta in metas:
        chunks = load_chunks(meta.path)
        resolved = AnnotationStore(meta.path).resolved_state()
        body = render_transcript(chunks, resolved)
        parts.append("=" * 78 + "\n")
        parts.append(render_header(meta, exported_at, full=full_header))
        parts.append(body)
        parts.append("\n")
        total_lines += sum(1 for line in body.splitlines() if line.strip())
    out_path.write_text("".join(parts), encoding="utf-8")
    return total_lines


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"--since/--until must be YYYY-MM-DD: {e}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sessions", nargs="*", type=Path,
                   help="explicit session directories (alternative to --root)")
    p.add_argument("--root", type=Path, default=None,
                   help="discover sessions under this directory")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory (will be created)")
    p.add_argument("--since", default=None,
                   help="only include sessions recorded on/after YYYY-MM-DD")
    p.add_argument("--until", default=None,
                   help="only include sessions recorded on/before YYYY-MM-DD")
    p.add_argument("--combined", action="store_true",
                   help="write a single combined.txt instead of one file per session")
    p.add_argument("--combined-name", default="combined.txt",
                   help="filename for --combined output (default combined.txt)")
    p.add_argument("--full-header", action="store_true",
                   help="include rich metadata header (name, source, exported_at, "
                        "etc). Default header is minimal: just `recorded:` — that "
                        "avoids confusing downstream LLMs which otherwise pick up "
                        "metadata as content.")
    args = p.parse_args(argv)

    candidate_dirs: list[Path] = []
    if args.root:
        candidate_dirs.extend(discover_sessions(args.root))
    for sd in args.sessions:
        if not sd.is_dir():
            print(f"error: not a directory: {sd}", file=sys.stderr)
            return 2
        candidate_dirs.append(sd)
    if not candidate_dirs:
        print("error: pass session dirs or --root", file=sys.stderr)
        return 2

    # De-dup while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for d in candidate_dirs:
        key = str(d.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)

    metas = [session_meta(d) for d in unique]
    from datetime import timedelta
    since = _parse_date(args.since)
    until = _parse_date(args.until)
    if until is not None:
        # --until YYYY-MM-DD is inclusive — interpret as end-of-day.
        until = until + timedelta(days=1) - timedelta(seconds=1)
    metas = filter_by_date(metas, since, until)
    metas.sort(key=lambda m: (m.recorded_at or datetime.min.replace(tzinfo=timezone.utc),
                              m.name))
    if not metas:
        print("error: no sessions matched filters", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    exported_at = datetime.now(tz=timezone.utc)
    if args.combined:
        out_path = args.out / args.combined_name
        total = export_combined(
            metas, out_path, exported_at, full_header=args.full_header,
        )
        print(f"wrote {out_path} ({len(metas)} sessions, {total} lines)")
    else:
        results = []
        for m in metas:
            path, n = export_session(
                m, args.out, exported_at, full_header=args.full_header,
            )
            results.append((path, n))
            print(f"wrote {path} ({n} lines)")
        print(f"total: {len(results)} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
