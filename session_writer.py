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
from typing import Callable, Optional

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
        # Markers waiting for transcription to catch up to a given audio
        # timestamp before being inserted. Used for recording-end so the
        # token lands AFTER any in-flight segments whose audio precedes the
        # press, instead of at the cursor as of press time. See
        # ``schedule_marker``.
        self._pending_markers: list[tuple[float, str]] = []
        # Callbacks waiting for transcription to reach a given audio time.
        # Each entry is (audio_time, callback). The callback is invoked with
        # the current cursor (len(buffer)) once a segment with end_time >=
        # audio_time has been appended. Used by clipboard-window start to
        # wait until segments through the press moment have been folded into
        # the buffer before snapshotting the start cursor.
        self._pending_cursor_callbacks: list[tuple[float, Callable[[int], None]]] = []
        # Per-word index: parallel list of (buffer_offset, word_end_time)
        # for every word appended to the buffer. Used by
        # ``slice_for_paste_by_time`` to trim head/tail words whose audio
        # straddles a clipboard-window press.
        self._word_index: list[tuple[int, float, float]] = []  # (offset, w_start, w_end)

        self._started_at = datetime.now(tz=timezone.utc)
        self._ended_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def feed_segment(
        self,
        text: str,
        end_time: float,
        words: Optional[list] = None,
    ) -> None:
        """Append a transcribed segment to the buffer.

        ``end_time`` is the session-relative end of the segment. It's used by
        clipboard-window flush to know when the transcriber has caught up to
        a given hotkey-press timestamp.

        ``words``, when provided, is a list of objects with ``text``,
        ``start_time``, ``end_time``. The writer indexes their buffer
        offsets so word-level slicing can trim straddling segments.

        After appending the segment, any pending markers whose audio
        threshold has now been overshot are flushed so they land AFTER the
        segment that spans the press moment, and any pending cursor-capture
        callbacks whose audio time has been reached are fired with the
        post-append cursor.
        """
        if not text:
            return
        fired_callbacks: list[tuple[Callable[[int], None], int]] = []
        with self._lock:
            if self._buffer and not self._buffer.endswith((" ", "\n")):
                self._buffer += " "
            seg_offset = len(self._buffer)
            self._buffer += text
            self._latest_segment_end = end_time
            # Index the words in the appended segment text. We re-locate each
            # word by linear scan through the segment text so the offsets
            # tolerate Whisper's leading-space convention on word.text.
            if words:
                cursor = seg_offset
                seg_len = len(text)
                for w in words:
                    wt = (w.text if hasattr(w, "text") else w.get("text", "")).strip()
                    if not wt:
                        continue
                    rel = text.find(wt, cursor - seg_offset)
                    if rel < 0:
                        # Word text didn't match (punctuation merge etc.) —
                        # fall back to advancing cursor by word length.
                        rel = cursor - seg_offset
                    abs_off = seg_offset + rel
                    self._word_index.append((
                        abs_off,
                        float(w.start_time if hasattr(w, "start_time") else w["start_time"]),
                        float(w.end_time if hasattr(w, "end_time") else w["end_time"]),
                    ))
                    cursor = abs_off + len(wt)
            if self._pending_markers:
                still_pending: list[tuple[float, str]] = []
                for audio_time, token in self._pending_markers:
                    if audio_time <= end_time:
                        self._append_token(token)
                    else:
                        still_pending.append((audio_time, token))
                self._pending_markers = still_pending
            if self._pending_cursor_callbacks:
                still_cb: list[tuple[float, Callable[[int], None]]] = []
                for audio_time, cb in self._pending_cursor_callbacks:
                    if audio_time <= end_time:
                        fired_callbacks.append((cb, len(self._buffer)))
                    else:
                        still_cb.append((audio_time, cb))
                self._pending_cursor_callbacks = still_cb
        for cb, cursor in fired_callbacks:
            try:
                cb(cursor)
            except Exception:
                pass

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

    def schedule_marker(self, type_name: str, kind: str, audio_time: float) -> None:
        """Queue a marker to be inserted once transcription reaches ``audio_time``.

        Used for ``recording:end`` so the token lands AFTER any in-flight
        segments whose audio precedes the press. The marker is appended by
        ``feed_segment`` the first time a segment arrives whose ``end_time``
        meets or exceeds ``audio_time``. If the session ends with markers
        still pending, ``force_flush`` drains them.
        """
        token = marker_start(type_name) if kind == "start" else marker_end(type_name)
        with self._lock:
            self._pending_markers.append((audio_time, token))

    def schedule_cursor_capture(
        self,
        audio_time: float,
        callback: Callable[[int], None],
    ) -> None:
        """Invoke ``callback(cursor)`` once a segment with end_time >=
        audio_time has been folded into the buffer.

        If the latest seen segment already covers ``audio_time``, the
        callback fires immediately with the current cursor (still under the
        lock to avoid racing concurrent feed_segment calls).
        """
        fire_now: Optional[int] = None
        with self._lock:
            latest = getattr(self, "_latest_segment_end", 0.0)
            if latest >= audio_time:
                fire_now = len(self._buffer)
            else:
                self._pending_cursor_callbacks.append((audio_time, callback))
        if fire_now is not None:
            try:
                callback(fire_now)
            except Exception:
                pass

    def flush_pending_markers(self) -> int:
        """Drain all scheduled markers immediately, appending them at the
        current cursor. Returns the number of markers drained.

        Used by ``_finish_paste`` after the drain-wait completes so that
        an end-press on a recording followed by silence still materializes
        its end token, instead of leaving it pending until the next
        recording's first segment crosses the old press time.
        """
        with self._lock:
            n = len(self._pending_markers)
            if not n:
                return 0
            for _audio_time, token in self._pending_markers:
                self._append_token(token)
            self._pending_markers = []
            return n

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

    def slice_for_paste_by_time(
        self,
        cursor_start: int,
        cursor_end: Optional[int],
        start_press_time: Optional[float],
        end_press_time: Optional[float],
        strip_span_types: Optional[set[str]] = None,
    ) -> str:
        """Slice with word-level trimming at the press-time boundaries.

        Begins from the cursor-based span and then advances ``cursor_start``
        forward past any words whose audio ends before ``start_press_time``,
        and pulls ``cursor_end`` backward past any words whose audio starts
        after ``end_press_time``. Falls back to the cursor-based slice if
        the press times are None or no word index is available in the span.
        """
        with self._lock:
            buf_len = len(self._buffer)
            end = cursor_end if cursor_end is not None else buf_len
            adj_start = cursor_start
            adj_end = end
            # Forward-trim leading words whose audio entirely precedes start.
            if start_press_time is not None:
                for off, w_start, w_end in self._word_index:
                    if off < cursor_start or off >= end:
                        continue
                    if w_end <= start_press_time:
                        # Word ended before press; skip past it.
                        adj_start = max(adj_start, off + 0)
                        # We want to land BEFORE the first word that
                        # straddles or follows the press, so step the
                        # cursor to the next word's offset on the next
                        # iteration. Track that here:
                    else:
                        adj_start = off
                        break
            # Backward-trim trailing words whose audio entirely follows end.
            if end_press_time is not None:
                for off, w_start, w_end in reversed(self._word_index):
                    if off < cursor_start or off >= end:
                        continue
                    if w_start >= end_press_time:
                        adj_end = off
                    else:
                        break
            if adj_end < adj_start:
                adj_end = adj_start
            raw = self._buffer[adj_start:adj_end]
        if strip_span_types is None:
            strip_span_types = set(self.marker_types.keys())
        return _strip_markers(raw, strip_span_types)

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
            # Drain any markers still waiting for transcription to catch up
            # so they don't get lost on shutdown.
            if self._pending_markers:
                for _audio_time, token in self._pending_markers:
                    self._append_token(token)
                self._pending_markers = []
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
