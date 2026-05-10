#!/usr/bin/env python3
"""Extract all annotations of a single marker type from a session.

Marker tokens may span chunk boundaries (the writer flushes chunks at
silence + token-target without regard to open markers). This tool walks
chunk_*.txt files in order, holding any in-progress span across files
until its closing token is found. A warning is emitted when a single span
covers more than two chunk files, since that's an unusually long
annotation that may indicate a missing end-toggle.

Usage::

    python extract_markers.py --session 20260510_143022 --type note
    python extract_markers.py --session-dir /abs/path/to/transcripts/<sid> --type note
    python extract_markers.py --session 20260510_143022 --type note \
                              --config local_config.json --output notes.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Optional

from config import DEFAULT_MARKERS, TRANSCRIPTS_DIR, marker_end, marker_start
from hotkeys import load_config


def _resolve_session_dir(session: Optional[str], session_dir: Optional[str]) -> str:
    if session_dir:
        return session_dir
    if not session:
        sys.exit("error: must provide --session or --session-dir")
    return os.path.join(TRANSCRIPTS_DIR, session)


def _markers_from_config(config_path: Optional[str]) -> list[dict]:
    if config_path is None:
        cfg = load_config(None)
    else:
        cfg = load_config(config_path)
    persistent = cfg.get("persistent") or {}
    markers = persistent.get("markers") or DEFAULT_MARKERS
    return list(markers)


def _list_chunks(session_dir: str) -> list[str]:
    if not os.path.isdir(session_dir):
        sys.exit(f"error: not a session directory: {session_dir}")
    chunks = sorted(
        f for f in os.listdir(session_dir)
        if f.startswith("chunk_") and f.endswith(".txt")
    )
    if not chunks:
        sys.exit(f"error: no chunk_*.txt files in {session_dir}")
    return chunks


def extract(session_dir: str, marker_type: str) -> dict:
    """Walk chunks in order; collect every closed `marker_type` span."""
    chunks = _list_chunks(session_dir)
    start_token = marker_start(marker_type)
    end_token = marker_end(marker_type)
    start_re = re.compile(re.escape(start_token))
    end_re = re.compile(re.escape(end_token))

    instances: list[dict] = []
    warnings: list[str] = []

    in_span = False
    span_text_parts: list[str] = []
    span_start_chunk: Optional[str] = None
    span_chunks: list[str] = []

    for chunk_name in chunks:
        path = os.path.join(session_dir, chunk_name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        pos = 0
        while pos < len(text):
            if in_span:
                m = end_re.search(text, pos)
                if m is None:
                    # Span continues past this chunk.
                    if not span_chunks or span_chunks[-1] != chunk_name:
                        span_chunks.append(chunk_name)
                    span_text_parts.append(text[pos:])
                    pos = len(text)
                else:
                    if not span_chunks or span_chunks[-1] != chunk_name:
                        span_chunks.append(chunk_name)
                    span_text_parts.append(text[pos:m.start()])
                    span_text = re.sub(
                        r"\s+", " ", "".join(span_text_parts)
                    ).strip()
                    instance = {
                        "index": len(instances),
                        "start_chunk": span_start_chunk,
                        "end_chunk": chunk_name,
                        "chunks_spanned": len(span_chunks),
                        "text": span_text,
                    }
                    instances.append(instance)
                    if instance["chunks_spanned"] > 2:
                        warnings.append(
                            f"span #{instance['index']} ({marker_type}) covers "
                            f"{instance['chunks_spanned']} chunks "
                            f"({span_start_chunk} → {chunk_name}) — "
                            "possible missing end-toggle?"
                        )
                    pos = m.end()
                    in_span = False
                    span_text_parts = []
                    span_start_chunk = None
                    span_chunks = []
            else:
                m = start_re.search(text, pos)
                if m is None:
                    pos = len(text)
                else:
                    in_span = True
                    span_start_chunk = chunk_name
                    span_chunks = [chunk_name]
                    span_text_parts = []
                    pos = m.end()

    unclosed: list[dict] = []
    if in_span:
        unclosed.append({
            "start_chunk": span_start_chunk,
            "chunks_spanned_so_far": len(span_chunks),
            "text_so_far": re.sub(r"\s+", " ", "".join(span_text_parts)).strip(),
        })
        warnings.append(
            f"unclosed `{marker_type}` span starting in {span_start_chunk}; "
            "treated as incomplete"
        )

    return {
        "session_dir": os.path.abspath(session_dir),
        "session_id": os.path.basename(os.path.normpath(session_dir)),
        "marker_type": marker_type,
        "instance_count": len(instances),
        "instances": instances,
        "unclosed": unclosed,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract all instances of one marker type from a session."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--session", help="Session id (folder under transcripts/)")
    src.add_argument("--session-dir", help="Absolute path to a session dir")
    parser.add_argument("--type", "-t", required=True,
                        help="Marker type identifier (e.g. note, assistant_annotation)")
    parser.add_argument("--config", "-c",
                        help="Path to JSON config (default: local_config.json or DEFAULT_MARKERS)")
    parser.add_argument(
        "--output", "-o",
        help="Output JSON path. Default: <session_dir>/<type>_extracts.json. "
             "Use '-' for stdout.",
    )
    args = parser.parse_args()

    markers = _markers_from_config(args.config)
    known_types = {m["type"] for m in markers}
    if args.type not in known_types:
        sys.exit(
            f"error: type '{args.type}' is not in the config. "
            f"Known types: {sorted(known_types)}"
        )

    session_dir = _resolve_session_dir(args.session, args.session_dir)
    result = extract(session_dir, args.type)

    for w in result["warnings"]:
        print(f"warning: {w}", file=sys.stderr)

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(text)
    else:
        out_path = args.output or os.path.join(
            session_dir, f"{args.type}_extracts.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        print(
            f"wrote {result['instance_count']} {args.type} extract(s) → {out_path}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
