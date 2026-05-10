"""Owns the on-disk layout for a persistent dictation session.

Single-threaded by contract: every method that mutates the buffer must be
called from the aggregator thread (the persistent app uses a lock for the
hotkey thread). The writer is the sole source of truth for the buffer
text — segments append, markers insert tokens, clipboard windows snapshot
cursors, and chunk flushes happen here.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config import (
    CHUNK_TOKEN_TARGET,
    MARKER_TOKEN_RE,
    TRANSCRIPTS_DIR,
    marker_end,
    marker_start,
)


_MARKER_RE = re.compile(MARKER_TOKEN_RE)


@dataclass
class MarkerType:
    key: str
    type: str
    description: str


def _approx_word_count(text: str) -> int:
    """Word-count approximation; we treat ~0.75 tokens/word so 7500 words ≈ 10k tokens."""
    return len(text.split())


def _strip_annotation_spans(text: str) -> str:
    """Remove ``<<<MARKER:type:start>>> ... <<<MARKER:type:end>>>`` ranges in full.

    Markers and the speech between them are deleted. An unclosed start marker
    in the input strips from the marker to the end of ``text``. Adjacent
    surrounding pieces are joined with a single space so word boundaries are
    preserved.
    """
    pieces: list[str] = []
    i = 0
    while i < len(text):
        m = _MARKER_RE.search(text, i)
        if not m:
            pieces.append(text[i:])
            break
        pieces.append(text[i:m.start()])
        if m.group(2) == "end":
            # Stray end without a matching start — drop just the token.
            i = m.end()
            continue
        type_name = m.group(1)
        end_re = re.compile(r"<<<MARKER:" + re.escape(type_name) + r":end>>>")
        em = end_re.search(text, m.end())
        if not em:
            # Unclosed start — drop everything from the start token onward.
            break
        i = em.end()
    # Join pieces with whitespace collapsed to single spaces around the seams.
    return re.sub(r"\s+", " ", " ".join(p.strip() for p in pieces if p.strip())).strip()


class SessionWriter:
    """Manages the in-memory transcript buffer and chunk file flushes.

    Threading model: ``feed_segment``, ``insert_marker``, ``open_window``,
    ``close_window``, ``flush`` and ``finalize`` all take the internal lock.
    Reading methods (``buffer_view``, ``cursor``) also lock; the user of this
    class shouldn't hold the lock across lengthy work.
    """

    def __init__(
        self,
        marker_types: list[MarkerType],
        model_label: str,
        transcripts_dir: str = TRANSCRIPTS_DIR,
        token_target: int = CHUNK_TOKEN_TARGET,
    ):
        self.marker_types = {m.type: m for m in marker_types}
        self._key_to_type = {m.key: m.type for m in marker_types}
        self.model_label = model_label
        self.token_target = token_target

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(transcripts_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._buffer = ""
        self._open_marker_type: Optional[str] = None
        self._chunk_index = 0
        # Position in ``self._buffer`` at which the current chunk started.
        # On flush, [_chunk_start_in_buffer..len(buffer)] is what we write.
        self._chunk_start_in_buffer = 0

        self._started_at = datetime.now(tz=timezone.utc)
        self._ended_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def feed_segment(self, text: str, end_time: float) -> None:
        """Append a transcribed segment to the buffer.

        ``end_time`` is the session-relative end of the segment. It's used by
        clipboard-window flush to know when the transcriber has caught up to
        a given hotkey-press timestamp.
        """
        if not text:
            return
        with self._lock:
            if self._buffer and not self._buffer.endswith((" ", "\n")):
                self._buffer += " "
            self._buffer += text
            self._latest_segment_end = end_time

    def latest_segment_end(self) -> float:
        with self._lock:
            return getattr(self, "_latest_segment_end", 0.0)

    def insert_marker(self, key: str) -> Optional[list[tuple[str, str]]]:
        """Toggle marker for the given hotkey.

        Returns a list of (action, type) events emitted, or None if the key
        isn't bound to a marker type. The list will contain one entry for a
        simple open or close, or two entries (close-then-open) when a
        cross-type switch auto-closes the previously open type.
        """
        type_name = self._key_to_type.get(key)
        if type_name is None:
            return None
        events: list[tuple[str, str]] = []
        with self._lock:
            if self._open_marker_type == type_name:
                self._append_token(marker_end(type_name))
                self._open_marker_type = None
                events.append(("close", type_name))
                return events
            if self._open_marker_type is not None:
                # No-overlap rule: close the currently-open type first.
                closed = self._open_marker_type
                self._append_token(marker_end(closed))
                self._open_marker_type = None
                events.append(("close", closed))
            self._append_token(marker_start(type_name))
            self._open_marker_type = type_name
            events.append(("open", type_name))
            return events

    def _append_token(self, token: str) -> None:
        if self._buffer and not self._buffer.endswith((" ", "\n")):
            self._buffer += " "
        self._buffer += token

    def open_marker_type(self) -> Optional[str]:
        with self._lock:
            return self._open_marker_type

    # ------------------------------------------------------------------
    # Cursors / clipboard windows
    # ------------------------------------------------------------------

    def cursor(self) -> int:
        """Current end-of-buffer cursor position (character index)."""
        with self._lock:
            return len(self._buffer)

    def slice_buffer(self, start: int, end: Optional[int] = None) -> str:
        with self._lock:
            if end is None:
                end = len(self._buffer)
            return self._buffer[start:end]

    def slice_for_paste(self, start: int, end: Optional[int] = None) -> str:
        """Slice with full annotation spans (markers + content) removed.

        Used by clipboard-window pastes.
        """
        raw = self.slice_buffer(start, end)
        return _strip_annotation_spans(raw)

    # ------------------------------------------------------------------
    # Chunk flushing
    # ------------------------------------------------------------------

    def maybe_flush_chunk(self, at_silence_boundary: bool) -> Optional[str]:
        """Flush the pending chunk if the size threshold is hit at a silence.

        Returns the path of the written chunk, or None if no flush happened.
        """
        if not at_silence_boundary:
            return None
        with self._lock:
            pending = self._buffer[self._chunk_start_in_buffer:]
            if _approx_word_count(pending) < int(self.token_target * 0.75):
                return None
            return self._flush_locked(force=False)

    def force_flush(self) -> Optional[str]:
        """Flush whatever is pending regardless of size. Used on shutdown."""
        with self._lock:
            pending = self._buffer[self._chunk_start_in_buffer:]
            if not pending.strip():
                return None
            return self._flush_locked(force=True)

    def _flush_locked(self, force: bool) -> str:
        chunk_text = self._buffer[self._chunk_start_in_buffer:].strip()
        path = os.path.join(self.session_dir, f"chunk_{self._chunk_index:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk_text)
            f.write("\n")
        self._chunk_index += 1
        self._chunk_start_in_buffer = len(self._buffer)
        return path

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def finalize(self) -> str:
        """Write session_meta.json and return its path. Caller should
        already have called ``force_flush`` to flush the final chunk."""
        self._ended_at = datetime.now(tz=timezone.utc)
        meta = {
            "session_id": self.session_id,
            "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
            "ended_at": self._ended_at.isoformat().replace("+00:00", "Z"),
            "model": {
                "name": self.model_label,
                "compute": os.getenv("FW_COMPUTE", "int8"),
            },
            "chunk_count": self._chunk_index,
            "marker_types": [
                {"key": m.key, "type": m.type, "description": m.description}
                for m in self.marker_types.values()
            ],
        }
        path = os.path.join(self.session_dir, "session_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path
