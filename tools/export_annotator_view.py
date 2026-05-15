"""Export a compact per-session annotator view for batch auto-annotation.

Reads the same data the annotator UI loads (via ``annotator.loader``) and
writes ``<session_dir>/annotator_view.json`` listing chunks + segments
using the post-boundary-split segment IDs that the UI looks up.

Auto-annotation agents read this file (not the raw ``*_segments.jsonl``)
so their annotation rows in ``annotations.jsonl`` apply correctly after
boundary splitting.

Usage::

    python -m voice_dictation.tools.export_annotator_view <session_dir> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..annotator.loader import load_chunks


def export_one(session_dir: Path) -> Path:
    chunks = load_chunks(session_dir)
    view = {
        "session_dir": str(session_dir),
        "chunks": [c.to_dict() for c in chunks],
    }
    out = session_dir / "annotator_view.json"
    out.write_text(json.dumps(view, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session_dirs", nargs="+", type=Path)
    args = p.parse_args(argv)
    for sd in args.session_dirs:
        if not sd.is_dir():
            print(f"error: not a dir: {sd}", file=sys.stderr)
            return 2
        out = export_one(sd)
        n_chunks = json.loads(out.read_text())["chunks"]
        n_segs = sum(len(c["segments"]) for c in n_chunks)
        print(f"{out}: chunks={len(n_chunks)} segments={n_segs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
