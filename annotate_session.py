"""CLI launcher for offline session annotation.

Usage::

    python -m voice_dictation.annotate_session <session_dir> [--port 0]

Refuses if ``<session_dir>/audio/manifest.json`` is missing or has no
chunks. See ``FINETUNE_PLAN.md`` (Phase B) for the design.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import webbrowser
from pathlib import Path

from .annotator.server import AnnotatorServer


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="annotate_session",
        description="Annotate a finalized voice_dictation session "
                    "for fine-tuning.",
    )
    parser.add_argument("session_dir", help="path to sessions/<ts>/")
    parser.add_argument("--port", type=int, default=0,
                        help="HTTP port (0 = random, default)")
    parser.add_argument("--no-open", action="store_true",
                        help="don't open the browser")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    session = Path(args.session_dir).resolve()
    if not session.is_dir():
        print(f"error: not a directory: {session}", file=sys.stderr)
        return 2

    try:
        server = AnnotatorServer(session, port=args.port)
        server.start()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{server.actual_port}/annotate"
    print(f"annotator: {url}")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
