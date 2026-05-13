"""Canonical transcript timeline shared across all transcription streams.

Replaces the per-stream ``SessionWriter`` + ``MarkerBus`` + ``merged_view``
triangle. One ``TranscriptTimeline`` instance owns:

* Per-stream segment buffers + word_index (registered streams: fast, hq).
  Each stream's segments cover a contiguous audio range; the timeline picks
  the highest-priority covered region per audio time at render time.
* A single markers list anchored to audio_time. Stored separately from
  segment text; injected into rendered output on demand.
* Three chunk writers: canonical (best-priority + markers), fast raw, hq raw.

Threading: all public methods take ``self._lock``.

Rendering model:

* The cached ``_render_text`` is the *clean* canonical text — segment text
  only, merged across priorities, with NO marker tokens inline. Word-time
  offsets in ``_render_offsets`` index this clean text.
* Display methods (``tail``, ``full_text``) inject markers at the right
  offsets on demand and return the marker-augmented string.
* ``slice_for_paste_by_time`` slices the clean text by audio time, then
  injects only the markers whose audio_time falls inside the slice, then
  strips per ``strip_span_types``.

This keeps the time→offset lookup stable across marker mutations and
avoids stale-offset bugs that arose when marker tokens were threaded into
the same buffer that the word_index was indexed against.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from chunk_writer import ChunkWriter
from config import (
    CHUNK_TOKEN_TARGET,
    FW_COMPUTE,
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


@dataclass
class Marker:
    audio_time: float
    type_name: str
    kind: str               # "start" | "end"
    seq: int                # insertion-order tie-break


@dataclass
class _StreamState:
    stream_id: str
    priority: int
    model_label: str
    coverage_start_time: float = 0.0
    buffer: str = ""
    word_index: list[tuple[int, float, float]] = field(default_factory=list)  # (offset, w_start, w_end)
    high_watermark: float = 0.0
    raw_chunk_writer: Optional[ChunkWriter] = None


def _strip_markers(text: str, strip_span_types: set[str]) -> str:
    """Remove marker tokens.

    * Types in ``strip_span_types``: remove the entire span (start..end).
    * Other types: remove only the token strings.
    """
    pieces: list[str] = []
    i = 0
    in_strip_type: Optional[str] = None
    while i < len(text):
        m = _MARKER_RE.search(text, i)
        if m is None:
            if in_strip_type is None:
                pieces.append(text[i:])
            break
        if in_strip_type is None:
            pieces.append(text[i:m.start()])
        type_name = m.group(1)
        kind = m.group(2)
        if in_strip_type is not None:
            if kind == "end" and type_name == in_strip_type:
                in_strip_type = None
            i = m.end()
            continue
        if kind == "start" and type_name in strip_span_types:
            in_strip_type = type_name
        i = m.end()
    return re.sub(r"\s+", " ", " ".join(p.strip() for p in pieces if p.strip())).strip()


class TranscriptTimeline:
    """Single source of truth for a session's transcript state."""

    def __init__(
        self,
        marker_types: list[MarkerType],
        transcripts_dir: str = TRANSCRIPTS_DIR,
        token_target: int = CHUNK_TOKEN_TARGET,
        session_id: Optional[str] = None,
    ):
        self.marker_types = {m.type: m for m in marker_types}
        self._key_to_type = {m.key: m.type for m in marker_types}
        self._token_target = token_target

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(transcripts_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self._started_at = datetime.now(tz=timezone.utc)
        self._ended_at: Optional[datetime] = None

        self._lock = threading.RLock()

        self._streams: dict[str, _StreamState] = {}

        self._markers: list[Marker] = []
        self._marker_seq = 0

        self._open_user_marker_type: Optional[str] = None

        # Pending cursor captures: fire once fast_high_watermark >= audio_time.
        self._pending_cursor_callbacks: list[tuple[float, Callable[[int], None]]] = []

        self._canonical_writer = ChunkWriter(
            self.session_dir, "chunk_{:03d}.txt", token_target
        )
        self._canonical_emitted_chars = 0

        # Cached clean render (no marker tokens).
        self._render_cache_dirty = True
        self._render_text: str = ""
        self._render_spans: list[dict] = []   # [{"source": "fast"|"hq", "text": ...}]
        self._render_span_meta: list[dict] = []  # parallel: {"t_lo","t_hi","local_offsets"}
        self._render_offsets: list[tuple[int, float, float]] = []  # (offset, ws, we)

    # ------------------------------------------------------------------
    # Stream registration
    # ------------------------------------------------------------------

    def register_stream(
        self,
        stream_id: str,
        priority: int,
        model_label: str,
        coverage_start_time: float = 0.0,
    ) -> None:
        with self._lock:
            if stream_id in self._streams:
                self._streams[stream_id].model_label = model_label
                return
            fmt = f"chunk_{stream_id}_{{:03d}}.txt"
            raw_writer = ChunkWriter(self.session_dir, fmt, self._token_target)
            self._streams[stream_id] = _StreamState(
                stream_id=stream_id,
                priority=priority,
                model_label=model_label,
                coverage_start_time=coverage_start_time,
                raw_chunk_writer=raw_writer,
            )
            self._render_cache_dirty = True

    def unregister_stream(self, stream_id: str) -> None:
        with self._lock:
            state = self._streams.pop(stream_id, None)
            if state and state.raw_chunk_writer is not None:
                state.raw_chunk_writer.force_flush()
            self._render_cache_dirty = True

    def stream_exists(self, stream_id: str) -> bool:
        with self._lock:
            return stream_id in self._streams

    def update_model_label(self, stream_id: str, model_label: str) -> None:
        with self._lock:
            s = self._streams.get(stream_id)
            if s is not None:
                s.model_label = model_label

    def model_label(self, stream_id: str) -> str:
        with self._lock:
            s = self._streams.get(stream_id)
            return s.model_label if s else ""

    def chunk_count(self, stream_id: Optional[str] = None) -> int:
        """``stream_id=None`` returns canonical chunk count; otherwise the
        raw chunk count for the named stream."""
        with self._lock:
            if stream_id is None:
                return self._canonical_writer.chunk_count
            s = self._streams.get(stream_id)
            return s.raw_chunk_writer.chunk_count if s and s.raw_chunk_writer else 0

    # ------------------------------------------------------------------
    # Segment ingest
    # ------------------------------------------------------------------

    def ingest_segment(
        self,
        stream_id: str,
        text: str,
        start_time: float,
        end_time: float,
        words: Optional[list] = None,
    ) -> None:
        if not text:
            return
        fired: list[tuple[Callable[[int], None], int]] = []
        with self._lock:
            state = self._streams.get(stream_id)
            if state is None:
                return
            if state.buffer and not state.buffer.endswith((" ", "\n")):
                state.buffer += " "
            seg_offset = len(state.buffer)
            state.buffer += text
            state.high_watermark = max(state.high_watermark, end_time)
            if words:
                cursor = seg_offset
                for w in words:
                    wt = (w.text if hasattr(w, "text") else w.get("text", "")).strip()
                    if not wt:
                        continue
                    rel = text.find(wt, cursor - seg_offset)
                    if rel < 0:
                        rel = cursor - seg_offset
                    abs_off = seg_offset + rel
                    state.word_index.append((
                        abs_off,
                        float(w.start_time if hasattr(w, "start_time") else w["start_time"]),
                        float(w.end_time if hasattr(w, "end_time") else w["end_time"]),
                    ))
                    cursor = abs_off + len(wt)
            if state.raw_chunk_writer is not None:
                state.raw_chunk_writer.append(text)
            self._render_cache_dirty = True

            # Fire pending cursor captures whose audio_time is now covered
            # by the fast watermark (canonical render is anchored to fast).
            fast_wm = self._fast_high_watermark_locked()
            if self._pending_cursor_callbacks:
                still: list[tuple[float, Callable[[int], None]]] = []
                for at, cb in self._pending_cursor_callbacks:
                    if at <= fast_wm:
                        off = self._find_offset_for_time_locked(at, mode="before")
                        fired.append((cb, off))
                    else:
                        still.append((at, cb))
                self._pending_cursor_callbacks = still
        for cb, off in fired:
            try:
                cb(off)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._marker_seq += 1
        return self._marker_seq

    def append_marker(self, type_name: str, kind: str) -> None:
        """Place a marker at the current fast high-watermark."""
        with self._lock:
            at = self._fast_high_watermark_locked()
            self._markers.append(
                Marker(audio_time=at, type_name=type_name, kind=kind, seq=self._next_seq())
            )

    def schedule_marker(self, type_name: str, kind: str, audio_time: float) -> None:
        with self._lock:
            self._markers.append(
                Marker(audio_time=audio_time, type_name=type_name, kind=kind,
                       seq=self._next_seq())
            )

    def flush_pending_markers(self) -> int:
        """Re-anchor markers whose audio_time exceeds the current fast
        watermark to the watermark. Returns count re-anchored."""
        with self._lock:
            wm = self._fast_high_watermark_locked()
            n = 0
            for m in self._markers:
                if m.audio_time > wm:
                    m.audio_time = wm
                    n += 1
            return n

    def remove_last_marker(self, type_name: str, kind: str) -> bool:
        with self._lock:
            for i in range(len(self._markers) - 1, -1, -1):
                m = self._markers[i]
                if m.type_name == type_name and m.kind == kind:
                    self._markers.pop(i)
                    return True
            return False

    def insert_marker(self, key: str) -> Optional[list[tuple[str, str]]]:
        type_name = self._key_to_type.get(key)
        if type_name is None:
            return None
        events: list[tuple[str, str]] = []
        with self._lock:
            if self._open_user_marker_type == type_name:
                self.append_marker(type_name, "end")
                self._open_user_marker_type = None
                events.append(("close", type_name))
                return events
            if self._open_user_marker_type is not None:
                closed = self._open_user_marker_type
                self.append_marker(closed, "end")
                self._open_user_marker_type = None
                events.append(("close", closed))
            self.append_marker(type_name, "start")
            self._open_user_marker_type = type_name
            events.append(("open", type_name))
            return events

    def open_marker_type(self) -> Optional[str]:
        with self._lock:
            return self._open_user_marker_type

    # ------------------------------------------------------------------
    # Cursor captures
    # ------------------------------------------------------------------

    def schedule_cursor_capture(
        self,
        audio_time: float,
        callback: Callable[[int], None],
    ) -> None:
        fire_now: Optional[int] = None
        with self._lock:
            wm = self._fast_high_watermark_locked()
            if wm >= audio_time:
                fire_now = self._find_offset_for_time_locked(audio_time, mode="before")
            else:
                self._pending_cursor_callbacks.append((audio_time, callback))
        if fire_now is not None:
            try:
                callback(fire_now)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # High-watermarks
    # ------------------------------------------------------------------

    def _fast_high_watermark_locked(self) -> float:
        s = self._streams.get("fast")
        return s.high_watermark if s else 0.0

    def _hq_high_watermark_locked(self) -> float:
        s = self._streams.get("hq")
        return s.high_watermark if s else 0.0

    def fast_high_watermark(self) -> float:
        with self._lock:
            return self._fast_high_watermark_locked()

    def hq_high_watermark(self) -> float:
        with self._lock:
            return self._hq_high_watermark_locked()

    def latest_segment_end(self) -> float:
        return self.fast_high_watermark()

    # ------------------------------------------------------------------
    # Canonical render (clean: no markers inline)
    # ------------------------------------------------------------------

    def _build_render_locked(self) -> None:
        fast = self._streams.get("fast")
        hq = self._streams.get("hq")

        # (source, slice_text, [(off_in_slice, ws, we), ...])
        parts: list[tuple[str, str, list[tuple[int, float, float]]]] = []

        if fast is None or not fast.buffer:
            if hq is not None and hq.buffer:
                parts.append(("hq", hq.buffer, list(hq.word_index)))
        elif hq is None or hq.high_watermark <= 0:
            parts.append(("fast", fast.buffer, list(fast.word_index)))
        else:
            # HQ wins for the entire range [coverage_start, hq_high_watermark].
            # Beyond hq_high_watermark, fall back to fast (HQ hasn't caught
            # up there yet). HQ is expected to produce equal-or-better
            # coverage than fast for the audio it sees; if it doesn't, the
            # right fix is making HQ not drop audio (bigger queue, no
            # voiced-gating mismatch) — not masking the loss by mixing
            # streams here. See `dropped_queue_full`/`dropped_silent`
            # counters in the status payload.
            hq_start = hq.coverage_start_time
            hq_edge = hq.high_watermark
            pre_end = (
                self._find_offset_in_stream(fast, hq_start, "before")
                if hq_start > 0 else 0
            )
            hq_lo = (
                self._find_offset_in_stream(hq, hq_start, "before")
                if hq_start > 0 else 0
            )
            hq_hi = self._find_offset_in_stream(hq, hq_edge, "before")
            post_lo = self._find_offset_in_stream(fast, hq_edge, "after")

            if pre_end > 0:
                slice_text = fast.buffer[:pre_end]
                words = [(o, ws, we) for (o, ws, we) in fast.word_index if o < pre_end]
                parts.append(("fast", slice_text, words))
            if hq_hi > hq_lo:
                slice_text = hq.buffer[hq_lo:hq_hi]
                words = [
                    (o - hq_lo, ws, we) for (o, ws, we) in hq.word_index
                    if hq_lo <= o < hq_hi
                ]
                parts.append(("hq", slice_text, words))
            if post_lo < len(fast.buffer):
                slice_text = fast.buffer[post_lo:]
                words = [
                    (o - post_lo, ws, we) for (o, ws, we) in fast.word_index
                    if o >= post_lo
                ]
                parts.append(("fast", slice_text, words))

        rendered = ""
        spans: list[dict] = []
        offsets: list[tuple[int, float, float]] = []
        # Parallel metadata for spans (used by tail_annotated / full_annotated
        # to inject markers per-span at render time). Each entry:
        #   {"t_lo", "t_hi", "local_offsets": [(rel_offset, ws, we), ...]}
        # where rel_offset is local to the span's stripped text.
        span_meta: list[dict] = []

        for source, slice_text, words in parts:
            stripped = slice_text.strip()
            if not stripped:
                continue
            left_strip = len(slice_text) - len(slice_text.lstrip())
            sep = " " if rendered else ""
            span_start = len(rendered) + len(sep)
            rendered += sep + stripped
            spans.append({"source": source, "text": stripped})
            local_off: list[tuple[int, float, float]] = []
            t_lo = float("inf")
            t_hi = float("-inf")
            for off_in_slice, ws, we in words:
                rel = off_in_slice - left_strip
                if 0 <= rel <= len(stripped):
                    offsets.append((span_start + rel, ws, we))
                    local_off.append((rel, ws, we))
                    if ws < t_lo:
                        t_lo = ws
                    if we > t_hi:
                        t_hi = we
            span_meta.append({
                "t_lo": t_lo if t_lo != float("inf") else 0.0,
                "t_hi": t_hi if t_hi != float("-inf") else 0.0,
                "local_offsets": local_off,
            })

        offsets.sort(key=lambda x: x[0])
        self._render_text = rendered
        self._render_spans = spans
        self._render_span_meta = span_meta
        self._render_offsets = offsets
        self._render_cache_dirty = False

    def _ensure_render(self) -> None:
        if self._render_cache_dirty:
            self._build_render_locked()

    @staticmethod
    def _find_offset_in_stream(state: _StreamState, t: float, mode: str) -> int:
        if mode == "before":
            last_idx = -1
            for i, (_off, _ws, we) in enumerate(state.word_index):
                if we <= t:
                    last_idx = i
                else:
                    break
            if last_idx < 0:
                return 0
            if last_idx + 1 < len(state.word_index):
                return state.word_index[last_idx + 1][0]
            return len(state.buffer)
        for off, ws, _we in state.word_index:
            if ws >= t:
                return off
        return len(state.buffer)

    @staticmethod
    def _offset_for_time_in(
        text: str,
        offsets: list[tuple[int, float, float]],
        t: float,
        mode: str = "before",
    ) -> int:
        if not offsets:
            return len(text) if mode == "before" else 0
        if mode == "before":
            last_idx = -1
            for i, (_off, _ws, we) in enumerate(offsets):
                if we <= t:
                    last_idx = i
                else:
                    break
            if last_idx < 0:
                return 0
            if last_idx + 1 < len(offsets):
                return offsets[last_idx + 1][0]
            return len(text)
        for off, ws, _we in offsets:
            if ws >= t:
                return off
        return len(text)

    def _find_offset_for_time_locked(self, t: float, mode: str) -> int:
        self._ensure_render()
        return self._offset_for_time_in(self._render_text, self._render_offsets, t, mode)

    # ------------------------------------------------------------------
    # Marker injection (on-demand at read time)
    # ------------------------------------------------------------------

    def _eligible_markers_locked(self) -> list[Marker]:
        """Markers whose audio_time is at-or-before the fast watermark — i.e.
        markers that can be placed at a real offset in the canonical render.
        """
        wm = self._fast_high_watermark_locked()
        return sorted(
            [m for m in self._markers if m.audio_time <= wm],
            key=lambda m: (m.audio_time, m.seq),
        )

    def _pending_markers_locked(self) -> list[Marker]:
        """Markers whose audio_time hasn't been reached by transcription yet.

        These can't be placed at a meaningful offset in the canonical render
        (the audio they anchor to hasn't been turned into words), but they
        SHOULD still be visible to the user immediately — otherwise pressing
        ``r`` and then waiting for the next segment looks like nothing
        happened. ``tail``/``full_text``/``*_annotated`` surface these at
        the trailing edge of the rendered output. Once subsequent segments
        cross their audio_time, they move from this list to the eligible
        list and get placed at the correct word-level offset.
        """
        wm = self._fast_high_watermark_locked()
        return sorted(
            [m for m in self._markers if m.audio_time > wm],
            key=lambda m: (m.audio_time, m.seq),
        )

    @staticmethod
    def _pending_marker_tokens(pending: list[Marker]) -> str:
        if not pending:
            return ""
        parts = []
        for m in pending:
            token = marker_start(m.type_name) if m.kind == "start" else marker_end(m.type_name)
            parts.append(token)
        return " ".join(parts)

    def _inject_markers(
        self,
        clean_text: str,
        clean_offsets: list[tuple[int, float, float]],
        markers: list[Marker],
        clip_lo: float = float("-inf"),
        clip_hi: float = float("inf"),
    ) -> str:
        """Return ``clean_text`` with marker tokens for ``markers`` (filtered
        to audio_time in [clip_lo, clip_hi]) injected at the offsets
        corresponding to their audio_time."""
        injections: list[tuple[int, str, int]] = []  # (offset, token, seq)
        for m in markers:
            if not (clip_lo <= m.audio_time <= clip_hi):
                continue
            off = self._offset_for_time_in(clean_text, clean_offsets, m.audio_time, mode="before")
            token = marker_start(m.type_name) if m.kind == "start" else marker_end(m.type_name)
            injections.append((off, token, m.seq))
        if not injections:
            return clean_text
        injections.sort(key=lambda x: (x[0], x[2]))
        out: list[str] = []
        last = 0
        for off, token, _ in injections:
            chunk = clean_text[last:off]
            out.append(chunk)
            # Add a space if the prior char isn't whitespace.
            if (out and out[-1] and not out[-1].endswith((" ", "\n"))):
                out.append(" ")
            out.append(token)
            last = off
        out.append(clean_text[last:])
        result = "".join(out)
        return re.sub(r"  +", " ", result).strip()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def cursor(self) -> int:
        """Length of clean canonical render (no markers). Used for
        clipboard-window span endpoints."""
        with self._lock:
            self._ensure_render()
            return len(self._render_text)

    def tail(self, n: int) -> str:
        """Return the last ~n characters of the canonical render with
        markers injected at their proper positions, plus any pending markers
        (audio_time still ahead of fast watermark) appended at the end so
        they appear in the widget immediately on press."""
        if n <= 0:
            return ""
        with self._lock:
            self._ensure_render()
            clean = self._render_text
            clean_off = self._render_offsets
            lo_clean = max(0, len(clean) - n)
            tail_clean = clean[lo_clean:]
            tail_offsets = [
                (off - lo_clean, ws, we) for (off, ws, we) in clean_off
                if off >= lo_clean
            ]
            if tail_offsets:
                t_lo = tail_offsets[0][1]
            else:
                t_lo = float("-inf")
            markers = self._eligible_markers_locked()
            placed = self._inject_markers(tail_clean, tail_offsets, markers, clip_lo=t_lo)
            pending = self._pending_marker_tokens(self._pending_markers_locked())
        if pending:
            return (placed + " " + pending).strip() if placed else pending
        return placed

    def tail_annotated(self, n: int) -> list[dict]:
        """Return source-tagged spans for the last ~n characters, with
        markers injected inline into each span (so the widget renders the
        marker tokens alongside the text and colored by the span's source).
        """
        with self._lock:
            self._ensure_render()
            clean_spans = list(self._render_spans)
            meta = list(self._render_span_meta)
            markers = self._eligible_markers_locked()
        # Walk spans backward, accumulating up to n chars. Track which spans
        # we keep so we can decide their audio-time bounds for marker
        # injection.
        kept: list[tuple[dict, dict, str]] = []  # (span, meta, kept_text)
        remaining = n
        for span, m in zip(reversed(clean_spans), reversed(meta)):
            text = span.get("text", "")
            if len(text) <= remaining:
                kept.insert(0, (span, m, text))
                remaining -= len(text)
            else:
                kept.insert(0, (span, m, text[-remaining:]))
                break
        pending = self._pending_marker_tokens(self._pending_markers_locked())
        if not kept:
            # No words yet but maybe pending markers from a just-pressed
            # recording — show them so the widget reflects the press.
            return [{"source": "fast", "text": pending or ""}]
        out: list[dict] = []
        for span, m, kept_text in kept:
            out.append({
                "source": span.get("source"),
                "text": self._inject_markers_into_span(
                    kept_text, m, markers, full_span_text=span.get("text", "")
                ),
            })
        if pending:
            # Append pending markers to the last (most-recent) span so they
            # show up at the right edge of the rendered tail. They'll be
            # absorbed into a placed position on the next ingest_segment.
            out[-1]["text"] = (out[-1]["text"] + " " + pending).strip()
        return out

    @staticmethod
    def _inject_markers_into_span(
        kept_text: str,
        meta: dict,
        markers: list[Marker],
        full_span_text: str,
    ) -> str:
        """Inject any markers whose audio_time falls inside this span's
        time range into ``kept_text`` (a possibly-truncated tail of the
        full span). Offsets are computed against ``full_span_text`` then
        shifted to land in ``kept_text``."""
        t_lo = meta.get("t_lo", 0.0)
        t_hi = meta.get("t_hi", 0.0)
        relevant = [m for m in markers if t_lo <= m.audio_time <= t_hi]
        if not relevant:
            return kept_text
        local_off = meta.get("local_offsets", [])
        # Map each marker to an offset in the FULL span text.
        full_injections: list[tuple[int, str, int]] = []
        for m in relevant:
            off = TranscriptTimeline._offset_for_time_in(
                full_span_text, local_off, m.audio_time, mode="before"
            )
            token = marker_start(m.type_name) if m.kind == "start" else marker_end(m.type_name)
            full_injections.append((off, token, m.seq))
        # Convert to offsets within kept_text (which is the tail
        # ``full_span_text[len(full)-len(kept):]``).
        offset_shift = len(full_span_text) - len(kept_text)
        kept_injections = [
            (off - offset_shift, token, seq)
            for (off, token, seq) in full_injections
            if off >= offset_shift
        ]
        if not kept_injections:
            return kept_text
        kept_injections.sort(key=lambda x: (x[0], x[2]))
        out_parts: list[str] = []
        last = 0
        for off, token, _ in kept_injections:
            out_parts.append(kept_text[last:off])
            if out_parts and out_parts[-1] and not out_parts[-1].endswith((" ", "\n")):
                out_parts.append(" ")
            out_parts.append(token)
            last = off
        out_parts.append(kept_text[last:])
        return re.sub(r"  +", " ", "".join(out_parts)).strip()

    def full_text(self) -> str:
        with self._lock:
            self._ensure_render()
            markers = self._eligible_markers_locked()
            placed = self._inject_markers(self._render_text, self._render_offsets, markers)
            pending = self._pending_marker_tokens(self._pending_markers_locked())
        if pending:
            return (placed + " " + pending).strip() if placed else pending
        return placed

    def full_clean_text(self) -> str:
        """Render without marker tokens. Used for chunk-flush bookkeeping."""
        with self._lock:
            self._ensure_render()
            return self._render_text

    def full_annotated(self) -> list[dict]:
        """Same as tail_annotated but covers the entire canonical render."""
        with self._lock:
            self._ensure_render()
            spans = list(self._render_spans)
            meta = list(self._render_span_meta)
            markers = self._eligible_markers_locked()
            pending = self._pending_marker_tokens(self._pending_markers_locked())
        out: list[dict] = []
        for span, m in zip(spans, meta):
            text = span.get("text", "")
            out.append({
                "source": span.get("source"),
                "text": self._inject_markers_into_span(text, m, markers, full_span_text=text),
            })
        if pending:
            if out:
                out[-1]["text"] = (out[-1]["text"] + " " + pending).strip()
            else:
                out.append({"source": "fast", "text": pending})
        return out

    def find_offset_for_time(self, t: float, mode: str = "before") -> int:
        with self._lock:
            return self._find_offset_for_time_locked(t, mode)

    def slice_for_paste_by_time(
        self,
        cursor_start: int,
        cursor_end: Optional[int],
        start_press_time: Optional[float],
        end_press_time: Optional[float],
        strip_span_types: Optional[set[str]] = None,
    ) -> str:
        """Slice the canonical render by clean-text cursor range with
        word-level press-time trimming, inject any markers whose audio_time
        falls inside the slice's time range, then strip per
        ``strip_span_types`` (default: all known marker types)."""
        with self._lock:
            self._ensure_render()
            clean = self._render_text
            clean_off = self._render_offsets
            buf_len = len(clean)
            end = cursor_end if cursor_end is not None else buf_len
            adj_start = cursor_start
            adj_end = end
            if start_press_time is not None:
                for off, w_start, w_end in clean_off:
                    if off < cursor_start or off >= end:
                        continue
                    if w_end <= start_press_time:
                        adj_start = max(adj_start, off)
                    else:
                        adj_start = off
                        break
            if end_press_time is not None:
                for off, w_start, w_end in reversed(clean_off):
                    if off < cursor_start or off >= end:
                        continue
                    if w_start >= end_press_time:
                        adj_end = off
                    else:
                        break
            if adj_end < adj_start:
                adj_end = adj_start
            slice_clean = clean[adj_start:adj_end]
            # Offsets local to slice_clean.
            slice_offsets = [
                (off - adj_start, ws, we) for (off, ws, we) in clean_off
                if adj_start <= off < adj_end
            ]
            # Determine the slice's audio-time bounds for marker filtering.
            if slice_offsets:
                lo = slice_offsets[0][1]
                hi = slice_offsets[-1][2]
            elif start_press_time is not None or end_press_time is not None:
                lo = start_press_time if start_press_time is not None else float("-inf")
                hi = end_press_time if end_press_time is not None else float("inf")
            else:
                lo, hi = float("-inf"), float("inf")
            markers = self._eligible_markers_locked()
            injected = self._inject_markers(
                slice_clean, slice_offsets, markers, clip_lo=lo, clip_hi=hi
            )
        if strip_span_types is None:
            strip_span_types = set(self.marker_types.keys())
        return _strip_markers(injected, strip_span_types)

    # ------------------------------------------------------------------
    # Chunk-file flushing
    # ------------------------------------------------------------------

    def tick(self, at_silence_boundary: bool) -> Optional[str]:
        """Called by the fast aggregator on silence boundaries (and
        periodically). Flushes canonical + raw chunk writers when their
        thresholds are met."""
        canonical_path: Optional[str] = None
        with self._lock:
            self._ensure_render()
            full = self._render_text  # use clean text for chunk bookkeeping
            pending = full[self._canonical_emitted_chars:]
            if at_silence_boundary and len(pending.split()) >= int(self._token_target * 0.75):
                # Re-render with markers injected, then take the corresponding
                # tail for the canonical chunk write.
                markers = self._eligible_markers_locked()
                with_markers = self._inject_markers(full, self._render_offsets, markers)
                # Approximate: write the WHOLE rendered-with-markers text since
                # last canonical emit. Track by clean offset (markers don't
                # alter audio-time positions).
                # Use the simple approach: write the marker-injected text
                # corresponding to the pending clean range.
                # We can compute the marker-injected slice by re-injecting
                # markers only into the pending region.
                pending_offsets = [
                    (off - self._canonical_emitted_chars, ws, we)
                    for (off, ws, we) in self._render_offsets
                    if off >= self._canonical_emitted_chars
                ]
                if pending_offsets:
                    t_lo = pending_offsets[0][1]
                else:
                    t_lo = float("-inf")
                pending_with_markers = self._inject_markers(
                    pending, pending_offsets, markers, clip_lo=t_lo
                )
                self._canonical_writer.append(pending_with_markers)
                path = self._canonical_writer.maybe_flush(at_silence_boundary=True)
                if path:
                    self._canonical_emitted_chars = len(full)
                    canonical_path = path
            for s in self._streams.values():
                if s.raw_chunk_writer is not None:
                    s.raw_chunk_writer.maybe_flush(at_silence_boundary)
        return canonical_path

    def force_flush(self) -> dict:
        paths: dict[str, Optional[str]] = {}
        with self._lock:
            # At shutdown no more segments arrive, so any markers anchored
            # past the current fast watermark would never become eligible
            # and would be dropped from the canonical chunk + final
            # transcript. Re-anchor them to the watermark so they survive.
            self.flush_pending_markers()
            self._ensure_render()
            full = self._render_text
            pending = full[self._canonical_emitted_chars:]
            if pending.strip():
                markers = self._eligible_markers_locked()
                pending_offsets = [
                    (off - self._canonical_emitted_chars, ws, we)
                    for (off, ws, we) in self._render_offsets
                    if off >= self._canonical_emitted_chars
                ]
                pending_with_markers = self._inject_markers(
                    pending, pending_offsets, markers,
                )
                self._canonical_writer.append(pending_with_markers)
            self._canonical_emitted_chars = len(full)
            paths["canonical"] = self._canonical_writer.force_flush()
            for sid, s in self._streams.items():
                if s.raw_chunk_writer is not None:
                    paths[sid] = s.raw_chunk_writer.force_flush()
        return paths

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        self._ended_at = datetime.now(tz=timezone.utc)
        out: dict[str, str] = {}
        with self._lock:
            self._ensure_render()
            clean = self._render_text
            offsets = self._render_offsets
            markers = self._eligible_markers_locked()
            text_with_markers = self._inject_markers(clean, offsets, markers)
            spans = [dict(s) for s in self._render_spans]
            streams_meta = [
                {
                    "stream_id": s.stream_id,
                    "priority": s.priority,
                    "model": s.model_label,
                    "coverage_start_time": s.coverage_start_time,
                    "high_watermark": s.high_watermark,
                    "chunk_count": s.raw_chunk_writer.chunk_count if s.raw_chunk_writer else 0,
                }
                for s in self._streams.values()
            ]
            markers_meta = [
                {
                    "audio_time": m.audio_time,
                    "type": m.type_name,
                    "kind": m.kind,
                }
                for m in sorted(self._markers, key=lambda x: (x.audio_time, x.seq))
            ]
            session_meta = {
                "session_id": self.session_id,
                "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
                "ended_at": self._ended_at.isoformat().replace("+00:00", "Z"),
                "compute": FW_COMPUTE,
                "streams": streams_meta,
                "marker_types": [
                    {"key": m.key, "type": m.type, "description": m.description}
                    for m in self.marker_types.values()
                ],
                "markers": markers_meta,
                "canonical_chunk_count": self._canonical_writer.chunk_count,
            }
        try:
            p = os.path.join(self.session_dir, "transcript_final.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text_with_markers)
                f.write("\n")
            out["transcript_final_txt"] = p
            p = os.path.join(self.session_dir, "transcript_final.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"spans": spans, "markers": markers_meta}, f, indent=2)
            out["transcript_final_json"] = p
            p = os.path.join(self.session_dir, "session_meta.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(session_meta, f, indent=2)
            out["session_meta"] = p
        except Exception:
            pass
        return out
