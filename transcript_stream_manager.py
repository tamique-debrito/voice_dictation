"""TranscriptStreamManager — merges per-stream segments and emits paste actions.

This is the only "logic" module that is NOT mocked in replay. Its inputs
are the segment streams (direct queues for ordered delivery) and the
user-action bus topic. Its outputs are:

  - ``paste.actions`` events with the text for each completed paste window.
  - ``transcript.canonical`` events with the merged canonical render delta
    (used by the live transcript view).

Completion rule for a paste window ``[start_accepted_ms, end_accepted_ms]``
with ``end_press_idx``:
  - EITHER a segment with ``content_accepted_ms_end >= end_accepted_ms``
    is observed (timestamp-based, the v1 rule), OR
  - a segment carries ``ends_on_marker_idx == end_press_idx``
    (marker-based; the new rule that removes the wait for a later segment).
  The first condition met fires the paste; HQ amendments after fast-fired
  paste are not retried in Phase 5 — that's deferred. Most live UX wants
  the snappier paste anyway.

Canonical render strategy: when a paste window fires, slice text from the
"best" stream that has data covering the range. HQ wins if HQ has a
segment whose range covers the window; otherwise fast wins; otherwise we
wait. Phase 5 keeps this simple — full v1-style segment-by-segment merge
with overlap reconciliation is deferred until needed for transcript
chunking.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .types import (
    Event, Segment, WindowCompletion,
    TOPIC_PASTE_ACTIONS, TOPIC_TRANSCRIPT_CANONICAL, TOPIC_USER_ACTIONS,
)


logger = logging.getLogger(__name__)


@dataclass
class _PendingPaste:
    paste_idx: int
    start_press_idx: int
    end_press_idx: int
    start_accepted_ms: int
    end_accepted_ms: int
    received_at_accepted_ms: int
    is_aside: bool = False
    exclude_intervals: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class _StreamBuffer:
    label: str
    segments: list[Segment] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, seg: Segment) -> None:
        # Segments arrive in window order from the transcriber; within a
        # window they're already in content order. Append; if the new
        # segment's start is earlier than the last (rare race), keep the
        # list sorted lazily — completion checks scan the whole list.
        with self.lock:
            self.segments.append(seg)

    def snapshot(self) -> list[Segment]:
        with self.lock:
            return list(self.segments)


class TranscriptStreamManager:
    def __init__(
        self,
        bus: EventBus,
        clock: AcceptedClock,
        fast_segment_q: queue.Queue,
        hq_segment_q: Optional[queue.Queue],
        stop_event: threading.Event,
        *,
        complete_tolerance_ms: int = 200,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._fast_q = fast_segment_q
        self._hq_q = hq_segment_q
        self._stop = stop_event
        self._tolerance_ms = complete_tolerance_ms

        self._fast = _StreamBuffer("fast")
        self._hq = _StreamBuffer("hq")

        # Watermarks per stream: "everything up to this accepted_ms has
        # been fully integrated into the buffer." Advanced from either:
        #   - a Segment with end > current value, OR
        #   - a WindowCompletion for a window ending past current value.
        # All advances happen on the single drain thread per stream, so
        # the watermark advance and the buffer integration are atomic by
        # construction.
        self._fast_up_to = 0
        self._hq_up_to = 0
        self._wm_lock = threading.Lock()

        self._pending: list[_PendingPaste] = []
        self._pending_lock = threading.Lock()
        self._paste_seq = 0

        self._actions_sub = None
        self._fast_thread: Optional[threading.Thread] = None
        self._hq_thread: Optional[threading.Thread] = None

        logger.info(
            "TranscriptStreamManager initialized (tolerance=%dms)",
            complete_tolerance_ms,
        )

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._actions_sub = self._bus.subscribe(
            TOPIC_USER_ACTIONS, self._on_action, name="manager.user_actions",
        )
        self._fast_thread = threading.Thread(
            target=self._drain_loop, args=(self._fast_q, self._fast),
            name="manager-fast", daemon=True,
        )
        self._fast_thread.start()
        if self._hq_q is not None:
            self._hq_thread = threading.Thread(
                target=self._drain_loop, args=(self._hq_q, self._hq),
                name="manager-hq", daemon=True,
            )
            self._hq_thread.start()
        logger.info("TranscriptStreamManager.start")

    def shutdown(self) -> None:
        # Wake the drain loops with sentinels.
        try:
            self._fast_q.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass
        if self._hq_q is not None:
            try:
                self._hq_q.put_nowait(None)  # type: ignore[arg-type]
            except queue.Full:
                pass
        if self._fast_thread is not None:
            self._fast_thread.join(timeout=2.0)
        if self._hq_thread is not None:
            self._hq_thread.join(timeout=2.0)
        if self._actions_sub is not None:
            self._actions_sub.shutdown()
        # Flush any pending pastes with whatever data is on hand.
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
        for w in pending:
            self._try_complete(w, force=True)
        logger.info("TranscriptStreamManager.shutdown")

    # ------------------------------------------------------------------
    # Segment intake
    # ------------------------------------------------------------------

    def _drain_loop(self, q: queue.Queue, buf: _StreamBuffer) -> None:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            if isinstance(item, WindowCompletion):
                # Atomic watermark advance — every Segment with content
                # in this window has already been added to ``buf`` above
                # in this same thread (queue order).
                self._advance_watermark(buf.label, item.end_accepted_ms)
                self._maybe_complete_pending()
                continue
            # Real segment.
            buf.add(item)
            # Segment-end can also advance the watermark within a window
            # (e.g. window has S1 end=5500, S2 end=7000; paste at 5400
            # fires after S1, not after S2 or window completion).
            self._advance_watermark(buf.label, item.content_accepted_ms_end)
            self._bus.publish(Event(
                topic=TOPIC_TRANSCRIPT_CANONICAL,
                emit_accepted_ms=self._clock.now_accepted_ms(),
                seq=next_seq(),
                payload={
                    "source": buf.label,
                    "text": item.text,
                    "content_accepted_ms_start": item.content_accepted_ms_start,
                    "content_accepted_ms_end": item.content_accepted_ms_end,
                    "ends_on_marker_idx": item.ends_on_marker_idx,
                },
            ))
            self._maybe_complete_pending()

    def stats(self) -> dict:
        with self._wm_lock:
            fast_up = self._fast_up_to
            hq_up = self._hq_up_to
        with self._pending_lock:
            pending_count = len(self._pending)
        return {
            "fast_transcribed_up_to_ms": fast_up,
            "hq_transcribed_up_to_ms": hq_up,
            "pending_pastes": pending_count,
            "pastes_emitted": self._paste_seq,
            "fast_segments_buffered": len(self._fast.snapshot()),
            "hq_segments_buffered": len(self._hq.snapshot()),
            "complete_tolerance_ms": self._tolerance_ms,
        }

    def _advance_watermark(self, label: str, end_ms: int) -> None:
        with self._wm_lock:
            if label == "fast":
                if end_ms > self._fast_up_to:
                    self._fast_up_to = end_ms
            elif label == "hq":
                if end_ms > self._hq_up_to:
                    self._hq_up_to = end_ms

    # ------------------------------------------------------------------
    # Pending paste windows
    # ------------------------------------------------------------------

    def _on_action(self, ev: Event) -> None:
        action = ev.payload.get("action")
        if action not in ("paste_window_complete", "aside_ended"):
            return
        self._paste_seq += 1
        is_aside = action == "aside_ended"
        aside_intervals = ev.payload.get("aside_intervals") or []
        exclude = [
            (a["start_accepted_ms"], a["end_accepted_ms"])
            for a in aside_intervals
        ]
        window = _PendingPaste(
            paste_idx=self._paste_seq,
            start_press_idx=ev.payload["start_press_idx"],
            end_press_idx=ev.payload["end_press_idx"],
            start_accepted_ms=ev.payload["start_accepted_ms"],
            end_accepted_ms=ev.payload["end_accepted_ms"],
            received_at_accepted_ms=ev.emit_accepted_ms,
            is_aside=is_aside,
            exclude_intervals=exclude,
        )
        with self._pending_lock:
            self._pending.append(window)
        # Try immediately — segments may already cover the range.
        self._try_complete(window)

    def _maybe_complete_pending(self) -> None:
        with self._pending_lock:
            pending = list(self._pending)
        for w in pending:
            self._try_complete(w)

    def _try_complete(self, w: _PendingPaste, force: bool = False) -> bool:
        text = self._slice_text(w, force=force)
        if text is None and not force:
            return False
        # Remove from pending under lock; the first completer wins.
        with self._pending_lock:
            try:
                self._pending.remove(w)
            except ValueError:
                return False
        self._emit_paste(w, text or "")
        return True

    def _slice_text(self, w: _PendingPaste, *, force: bool = False) -> Optional[str]:
        """Return paste text if the manager has processed every audio
        block up to ``w.end_accepted_ms`` (within tolerance), else None.

        Watermark rule:
          - if hq_up_to + tol >= end → use HQ text in range (higher quality)
          - elif fast_up_to + tol >= end → use FAST text in range
          - else None (wait for more progress; or force=True at shutdown)
        ``hq_up_to`` typically lags fast by a lot, so most live pastes
        come from fast.
        """
        with self._wm_lock:
            fast_up = self._fast_up_to
            hq_up = self._hq_up_to
        end = w.end_accepted_ms
        tol = self._tolerance_ms
        if hq_up + tol >= end:
            return self._concat_in_range(
                self._hq.snapshot(), w.start_accepted_ms, end, w.exclude_intervals,
            )
        if fast_up + tol >= end:
            return self._concat_in_range(
                self._fast.snapshot(), w.start_accepted_ms, end, w.exclude_intervals,
            )
        if force:
            # Shutdown — emit whatever's available, preferring fast over hq
            # for completeness (fast typically has more coverage live).
            return self._concat_in_range(
                self._fast.snapshot(), w.start_accepted_ms, end, w.exclude_intervals,
            ) or self._concat_in_range(
                self._hq.snapshot(), w.start_accepted_ms, end, w.exclude_intervals,
            )
        return None

    @staticmethod
    def _concat_in_range(
        segments: list[Segment],
        start_ms: int,
        end_ms: int,
        exclude: list[tuple[int, int]] | None = None,
    ) -> str:
        # Pick segments whose content range overlaps [start_ms, end_ms].
        # For segments fully inside the range, take whole text. For
        # segments straddling either boundary, use word-level timings to
        # trim to just the words whose midpoints fall inside the range.
        ex = exclude or []

        def _in_exclude(t: int) -> bool:
            for a, b in ex:
                if a <= t < b:
                    return True
            return False

        def _seg_fully_excluded(s: Segment) -> bool:
            for a, b in ex:
                if s.content_accepted_ms_start >= a and s.content_accepted_ms_end <= b:
                    return True
            return False

        relevant = [
            s for s in segments
            if s.content_accepted_ms_end > start_ms and s.content_accepted_ms_start < end_ms
            and not _seg_fully_excluded(s)
        ]
        relevant.sort(key=lambda s: s.content_accepted_ms_start)
        parts: list[str] = []
        for s in relevant:
            fully_inside = (
                s.content_accepted_ms_start >= start_ms
                and s.content_accepted_ms_end <= end_ms
            )
            seg_touches_exclude = any(
                s.content_accepted_ms_end > a and s.content_accepted_ms_start < b
                for a, b in ex
            )
            if fully_inside and not seg_touches_exclude:
                parts.append(s.text)
                continue
            if not s.words:
                # Without word timings we can't subtract — keep the segment
                # only if it doesn't overlap an excluded range.
                if not seg_touches_exclude:
                    parts.append(s.text)
                continue
            kept: list[str] = []
            for w in s.words:
                mid = (w.start_accepted_ms + w.end_accepted_ms) // 2
                # Lenient inclusion at both edges — Whisper stretches first-word
                # starts backwards, and pressing the close key twice across a
                # single word is implausible in practice. Include any word that
                # overlaps the window at all.
                in_window = (
                    w.end_accepted_ms > start_ms
                    and w.start_accepted_ms < end_ms
                )
                if in_window and not _in_exclude(mid):
                    kept.append(w.text)
            piece = "".join(kept).strip()
            if piece:
                parts.append(piece)
        return " ".join(p for p in parts if p).strip()

    # ------------------------------------------------------------------
    # Public read-only views for the widget server.
    #
    # The widget asks for the canonical tail (merged text up to N chars)
    # via a single endpoint. Doing the merge here, not in JS, keeps the
    # logic in one place — same code path as paste-window text slicing.
    # ------------------------------------------------------------------

    def get_canonical_tail(self, max_chars: int = 10_000) -> dict:
        """Return the canonical transcript tail.

        Merge strategy: walk fast and hq segments in content-time order;
        within any time region where HQ has a segment, prefer HQ (its
        text replaces fast's). The returned ``segments`` is a list of
        ``{source, text, start_accepted_ms, end_accepted_ms}`` dicts in
        content-time order so the widget can render each piece styled by
        its origin (fast = lighter, hq = brighter) without doing merge
        work itself.

        ``hq_covered_end_accepted_ms`` is the end of the latest contiguous
        HQ region — useful for showing an "HQ progress cursor" in the UI.
        """
        fast_segs = sorted(
            self._fast.snapshot(),
            key=lambda s: (s.content_accepted_ms_start, s.window_id),
        )
        hq_segs = sorted(
            self._hq.snapshot(),
            key=lambda s: (s.content_accepted_ms_start, s.window_id),
        )

        # HQ coverage: HQ wins over any fast segment whose range overlaps
        # a HQ segment. Build a list of HQ time ranges to test against.
        hq_ranges = [
            (s.content_accepted_ms_start, s.content_accepted_ms_end)
            for s in hq_segs
        ]

        def _overlaps_hq(s: Segment) -> bool:
            for a, b in hq_ranges:
                if s.content_accepted_ms_end > a and s.content_accepted_ms_start < b:
                    return True
            return False

        kept_fast = [s for s in fast_segs if not _overlaps_hq(s)]
        merged = sorted(
            [(s, "fast") for s in kept_fast] + [(s, "hq") for s in hq_segs],
            key=lambda it: (it[0].content_accepted_ms_start, it[0].window_id),
        )

        # Tail by char budget.
        pieces: list[dict] = []
        total = 0
        for seg, src in reversed(merged):
            piece = {
                "source": src,
                "text": seg.text,
                "start_accepted_ms": seg.content_accepted_ms_start,
                "end_accepted_ms": seg.content_accepted_ms_end,
                "words": [
                    {
                        "text": w.text,
                        "start_accepted_ms": w.start_accepted_ms,
                        "end_accepted_ms": w.end_accepted_ms,
                    } for w in seg.words
                ],
            }
            total += len(seg.text) + 1
            pieces.append(piece)
            if total >= max_chars:
                break
        pieces.reverse()

        hq_covered_end = max((b for _, b in hq_ranges), default=0)
        return {
            "segments": pieces,
            "hq_covered_end_accepted_ms": hq_covered_end,
            "total_chars": sum(len(p["text"]) + 1 for p in pieces),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _emit_paste(self, w: _PendingPaste, text: str) -> None:
        self._bus.publish(Event(
            topic=TOPIC_PASTE_ACTIONS,
            emit_accepted_ms=self._clock.now_accepted_ms(),
            seq=next_seq(),
            payload={
                "paste_idx": w.paste_idx,
                "start_press_idx": w.start_press_idx,
                "end_press_idx": w.end_press_idx,
                "start_accepted_ms": w.start_accepted_ms,
                "end_accepted_ms": w.end_accepted_ms,
                "text": text,
                "is_aside": w.is_aside,
            },
        ))
