"""ChunkWriter: append-only text chunks to disk under a session directory.

Three instances run in parallel under ``TranscriptTimeline``:

* ``chunk_NNN.txt`` — canonical (best-priority merged text + markers) flushed
  at fast-stream silence boundaries.
* ``chunk_fast_NNN.txt`` — raw fast-stream text (no marker injection).
* ``chunk_hq_NNN.txt`` — raw HQ-stream text.

Each writer owns its own buffer and chunk index; flushes are independent.
"""

from __future__ import annotations

import os
import threading
from typing import Optional


def _approx_word_count(text: str) -> int:
    return len(text.split())


class ChunkWriter:
    def __init__(
        self,
        session_dir: str,
        filename_fmt: str,        # e.g. "chunk_{:03d}.txt" or "chunk_fast_{:03d}.txt"
        token_target: int,
    ):
        self._dir = session_dir
        self._fmt = filename_fmt
        self._token_target = token_target
        self._lock = threading.Lock()
        self._buffer = ""
        self._chunk_index = 0
        os.makedirs(self._dir, exist_ok=True)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return self._chunk_index

    def append(self, text: str) -> None:
        """Append text to the pending chunk. Newlines/spacing preserved by caller."""
        if not text:
            return
        with self._lock:
            if self._buffer and not self._buffer.endswith((" ", "\n")) \
                    and not text.startswith((" ", "\n")):
                self._buffer += " "
            self._buffer += text

    def replace(self, text: str) -> None:
        """Replace the pending chunk buffer wholesale. Used by the canonical
        writer when the merged view rewinds because HQ filled a gap that had
        been previously emitted as fast-only."""
        with self._lock:
            self._buffer = text

    def maybe_flush(self, at_silence_boundary: bool) -> Optional[str]:
        if not at_silence_boundary:
            return None
        with self._lock:
            if _approx_word_count(self._buffer) < int(self._token_target * 0.75):
                return None
            return self._flush_locked()

    def force_flush(self) -> Optional[str]:
        with self._lock:
            if not self._buffer.strip():
                return None
            return self._flush_locked()

    def _flush_locked(self) -> str:
        text = self._buffer.strip()
        path = os.path.join(self._dir, self._fmt.format(self._chunk_index))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        self._chunk_index += 1
        self._buffer = ""
        return path
