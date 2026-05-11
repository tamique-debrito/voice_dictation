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


def _strip_markers(text: str, strip_span_types: set[str]) -> str:
    """Remove marker tokens from ``text`` according to ``strip_span_types``.

    * For types listed in ``strip_span_types``: remove the entire span — start
      token, end token, and everything between (and the matching unclosed
      tail if there is no end token).
    * For all other marker types: remove only the token text; the surrounding
      content is preserved. This is how we treat the built-in recording /
      aside markers — they bracket content the user wants pasted, so the
      content stays but the literal token strings are scrubbed.

    Adjacent surviving pieces are joined with a single space so word
    boundaries are preserved.
    """
    pieces: list[str] = []
    i = 0
    in_strip_type: Optional[str] = None
    while i < len(text):
        m = _MARKER_RE.search(text, i)
        if m is None:
            if in_strip_type is None:
                pieces.append(text[i:])
            # else: unclosed strip span — drop trailing content.
            break
        if in_strip_type is None:
            pieces.append(text[i:m.start()])
        type_name = m.group(1)
        kind = m.group(2)
        if in_strip_type is not None:
            # Inside a fully-stripped span: only exit on matching end token;
            # otherwise swallow everything (including unrelated markers).
            if kind == "end" and type_name == in_strip_type:
                in_strip_type = None
            i = m.end()
            continue
        # Not inside a strip span.
        if kind == "start" and type_name in strip_span_types:
            in_strip_type = type_name
        # Other cases (token-only types, stray ends, etc.) just drop the token.
        i = m.end()
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

    def append_marker(self, type_name: str, kind: str) -> None:
        """Public append for built-in markers (recording, aside) that
        don't go through the no-overlap state machine in ``insert_marker``.
        ``kind`` is ``"start"`` or ``"end"``."""
        token = marker_start(type_name) if kind == "start" else marker_end(type_name)
        with self._lock:
            self._append_token(token)

    def remove_last_marker(self, type_name: str, kind: str) -> bool:
        """Find and remove the last occurrence of a specific marker token
        from the buffer. Used to scrub an unfinished recording/aside marker
        when the user cancels. Returns True if a token was found and removed.
        """
        token = marker_start(type_name) if kind == "start" else marker_end(type_name)
        with self._lock:
            idx = self._buffer.rfind(token)
            if idx == -1:
                return False
            before = self._buffer[:idx]
            after = self._buffer[idx + len(token):]
            # Collapse the doubled space that would otherwise be left behind.
            if before.endswith(" ") and after.startswith(" "):
                after = after.lstrip(" ")
            self._buffer = before + after
            return True

    def open_marker_type(self) -> Optional[str]:
        with self._lock:
            return self._open_marker_type

    def tail(self, n: int) -> str:
        """Return the last ``n`` characters of the buffer (full buffer if shorter)."""
        with self._lock:
            return self._buffer[-n:] if n > 0 else ""

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return self._chunk_index

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

    def slice_for_paste(
        self,
        start: int,
        end: Optional[int] = None,
        strip_span_types: Optional[set[str]] = None,
    ) -> str:
        """Slice with annotation spans removed and built-in marker tokens scrubbed.

        ``strip_span_types`` is the set of marker types whose full spans
        (markers + content) should be removed — typically the user-configured
        annotation types. Markers of other types (e.g. the built-in
        ``recording`` / ``aside`` ones) have only their token strings removed;
        the content they bracket is preserved.

        If ``strip_span_types`` is None, the writer's own configured marker
        types are used.
        """
        raw = self.slice_buffer(start, end)
        if strip_span_types is None:
            strip_span_types = set(self.marker_types.keys())
        return _strip_markers(raw, strip_span_types)

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
