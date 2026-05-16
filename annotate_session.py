"""CLI launcher for offline session annotation.

Usage::

    python -m voice_dictation.annotate_session <session_dir> [--port 0]
    python -m voice_dictation.annotate_session --root <folder> [--port 0]

Single-session mode: pass a directory containing ``audio/manifest.json``
and the usual JSONL streams.

Multi-session mode: pass ``--root <folder>`` to load every immediate
subdirectory that has an ``audio/manifest.json``. The UI shows a session
picker dropdown; ``[`` / ``]`` switch between them.
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
        description="Annotate one or more finalized voice_dictation sessions.",
    )
    parser.add_argument("session_dir", nargs="?",
                        help="path to a single session/<ts>/ (single-session mode)")
    parser.add_argument("--root", default=None,
                        help="folder containing multiple session dirs "
                             "(multi-session mode with picker)")
    parser.add_argument("--port", type=int, default=0,
                        help="HTTP port (0 = random, default)")
    parser.add_argument("--no-open", action="store_true",
                        help="don't open the browser")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if (args.session_dir is None) == (args.root is None):
        parser.error("provide exactly one of <session_dir> or --root <folder>")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    try:
        if args.root:
            server = AnnotatorServer.from_root(args.root, port=args.port)
        else:
            session = Path(args.session_dir).resolve()
            if not session.is_dir():
                print(f"error: not a directory: {session}", file=sys.stderr)
                return 2
            server = AnnotatorServer(session, port=args.port)
        server.start()
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{server.actual_port}/"
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
