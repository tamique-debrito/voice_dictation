"""Clipboard window state + lifecycle manager.

A "clipboard window" is the audio region between two double-taps of a
hotkey (r, aside, ...) whose transcription gets pasted into the focused
application. Each window type has slightly different semantics:

  * r       — main recording. Can be parked by an aside opening on top of it.
  * aside   — parks an active r window while it is open; appended at writer
              cursor time on end (vs. r's scheduled end marker).
  * future  — additional types can be added by appending a ClipboardWindowType.

The original implementation in voice_dictation_persistent.py duplicated
the start/end flow for r and aside. This module collapses that into one
ClipboardWindowManager driven by declarative ClipboardWindowType configs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ClipboardWindow:
    """Tracks an active clipboard paste window across park/resume cycles."""

    label: str
    start_cursor: Optional[int]
    end_press_time: float
    start_press_time: float
    spans: list[tuple[Optional[int], Optional[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.spans:
            self.spans = [(self.start_cursor, None)]

    def set_start_cursor(self, cursor: int) -> None:
        _, e = self.spans[0]
        self.spans[0] = (cursor, e)

    def park(self, cursor: int) -> None:
        s, _ = self.spans[-1]
        self.spans[-1] = (s, cursor)

    def resume(self, cursor: int) -> None:
        self.spans.append((cursor, None))

    def close(self, cursor: int) -> None:
        s, _ = self.spans[-1]
        self.spans[-1] = (s, cursor)

    def text(self, timeline) -> str:
        """Render the paste text for this window by slicing the timeline's
        canonical render at each span's cursor bounds, with word-level
        press-time trimming on the first/last span.

        The first span's start cursor is resolved against the *current*
        canonical render using ``start_press_time`` as the audio anchor.
        This is critical: if we captured a cursor offset at press time
        (when the fast watermark first crossed start_press_time) and HQ
        later substituted its own text for the same audio range, the
        canonical render rearranges underneath the saved offset, and a
        cursor that was originally "just after the last word before the
        press" lands deep inside the first post-press word in the new
        render. Resolving fresh against the current render is the only
        way to keep the slice start at the right audio-time boundary
        regardless of fast-vs-HQ re-renders between press and paste.
        Park/resume cursors (intermediate spans) are still captured at
        live time — they're rarer and re-rendering doesn't affect them
        the same way."""
        parts: list[str] = []
        for i, (s, e) in enumerate(self.spans):
            if s is None:
                if i == 0:
                    s = timeline.find_offset_for_time(
                        self.start_press_time, "before"
                    )
                else:
                    s = timeline.cursor()
                self.spans[i] = (s, e)
            sp = self.start_press_time if i == 0 else None
            ep = self.end_press_time if i == len(self.spans) - 1 else None
            parts.append(timeline.slice_for_paste_by_time(s, e, sp, ep))
        return " ".join(p for p in parts if p)


@dataclass
class ClipboardWindowType:
    """Declarative configuration for one kind of clipboard window."""

    key: str
    label: str
    marker_type: str
    can_park_others: bool
    can_be_parked: bool
    schedule_end_marker: bool


@dataclass
class ToggleResult:
    """Outcome of ClipboardWindowManager.toggle()."""

    action: str  # "noop" | "started" | "ended" | "start_blocked"
    type_label: Optional[str] = None
    closed_window: Optional[ClipboardWindow] = None
    error_msg: Optional[str] = None


@dataclass
class CancelResult:
    """Outcome of ClipboardWindowManager.cancel()."""

    nothing: bool = False
    cancelled_label: Optional[str] = None
    resumed_labels: list[str] = field(default_factory=list)


class ClipboardWindowManager:
    """Owns the live set of clipboard windows and their lifecycle.

    Handles state, marker emission, cursor-capture scheduling, and
    park/resume of overlapping windows. Does NOT trigger the paste —
    callers spawn a finisher thread on the ClipboardWindow returned in
    ToggleResult.closed_window.
    """

    def __init__(
        self,
        types: list[ClipboardWindowType],
        timeline,
        debug_log: Callable[[str, dict], None],
        request_force_flush: Callable[[], None],
        cursor_time_at: Callable[[float], float],
    ):
        self._types: dict[str, ClipboardWindowType] = {t.key: t for t in types}
        # The TranscriptTimeline owns marker storage, cursor/slice reads, and
        # cursor-capture scheduling. All marker mutations and reads go
        # through it; there is no per-stream marker bus anymore.
        self._timeline = timeline
        self._debug = debug_log
        # Callable that triggers an immediate aggregator flush on every
        # active transcription stream so trailing words reach the
        # transcriber inside the paste-drain budget.
        self._request_force_flush = request_force_flush
        self._cursor_time_at = cursor_time_at
        self._active: dict[str, ClipboardWindow] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_active(self, key: str) -> bool:
        with self._lock:
            return key in self._active

    def active_keys(self) -> list[str]:
        with self._lock:
            return list(self._active)

    def types(self) -> list[ClipboardWindowType]:
        return list(self._types.values())

    def keys(self) -> set[str]:
        return set(self._types)

    def type_for(self, key: str) -> Optional[ClipboardWindowType]:
        return self._types.get(key)

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def toggle(self, key: str) -> ToggleResult:
        """Live entry point — captures session time from monotonic clock."""
        press_ts = time.monotonic()
        return self.toggle_at(key, press_ts, self._cursor_time_at(press_ts))

    def toggle_at(
        self, key: str, press_ts: float, session_time: float,
    ) -> ToggleResult:
        """Replay-friendly entry point. Caller provides the session time
        explicitly (so recorded events can replay the original moment
        rather than relying on the live clock)."""
        cfg = self._types.get(key)
        if cfg is None:
            return ToggleResult(action="noop")
        with self._lock:
            if cfg.key in self._active:
                return self._end_locked(cfg, press_ts, session_time)
            return self._start_locked(cfg, press_ts, session_time)

    def _start_locked(
        self, cfg: ClipboardWindowType,
        press_ts: float, press_session: float,
    ) -> ToggleResult:
        # A parker is exclusive while open: refuse to start a parkable window
        # while one is active (matches original "aside is active; close it
        # with `a` first" behavior).
        if cfg.can_be_parked:
            for active_key in self._active:
                if self._types[active_key].can_park_others:
                    blocker = self._types[active_key]
                    return ToggleResult(
                        action="start_blocked",
                        error_msg=(
                            f"{blocker.label} is active; close it with "
                            f"`{blocker.key}` first"
                        ),
                    )

        # Drain any markers still pending from a prior window (most commonly
        # the end marker for a fast end-then-start sequence whose paste-
        # finish drain hasn't fired yet). Without this, the new start token
        # would land BEFORE the stale end token, producing
        # `start start ... end` in the buffer.
        drained = self._timeline.flush_pending_markers()
        if drained:
            self._debug("marker", {
                "type": cfg.marker_type,
                "action": "drained_pending",
                "key": "-",
                "count": drained,
                "trigger": "new_start",
            })
        # Always schedule at press_session — the marker is anchored to a
        # moment in audio, not to "now in the buffer." This is correct for
        # both fast and HQ canonical-render placement: the timeline injects
        # the marker at the offset corresponding to press_session using its
        # word-time index, regardless of which stream's transcription
        # delivered the words around that moment.
        self._timeline.schedule_marker(cfg.marker_type, "start", press_session)

        if cfg.can_park_others:
            for active_key, w in self._active.items():
                if self._types[active_key].can_be_parked:
                    w.park(self._timeline.cursor())

        window = ClipboardWindow(
            label=cfg.label,
            start_cursor=None,
            end_press_time=press_ts,
            start_press_time=press_session,
        )
        self._active[cfg.key] = window

        # The first span's start cursor is intentionally NOT captured here.
        # It's resolved at paste-time against the canonical render using
        # ``start_press_time`` as the audio anchor — see
        # ``ClipboardWindow.text`` for the reasoning. Capturing now (the
        # moment the fast watermark crossed ``press_session``) was the old
        # behavior and produced stale offsets once HQ substituted its
        # transcription for the same audio range, leading to words at the
        # start of a window being dropped from the paste.
        self._debug("press", {
            "key": cfg.key,
            "phase": "start",
            "session_time": round(press_session, 3),
        })
        return ToggleResult(action="started", type_label=cfg.label)

    def _end_locked(
        self, cfg: ClipboardWindowType,
        press_ts: float, press_session: float,
    ) -> ToggleResult:
        # Force the aggregator to flush its in-progress audio so trailing
        # words reach the transcriber inside the paste-drain budget instead
        # of waiting for VAD silence.
        self._request_force_flush()
        # All end markers schedule at press_session for the same audio-time
        # anchoring reason as starts. The legacy ``schedule_end_marker``
        # flag on ClipboardWindowType is retained for config compatibility
        # but no longer toggles behavior.
        self._timeline.schedule_marker(cfg.marker_type, "end", press_session)
        window = self._active.pop(cfg.key)
        window.end_press_time = press_ts
        if cfg.can_park_others:
            for active_key, w in self._active.items():
                if self._types[active_key].can_be_parked:
                    w.resume(self._timeline.cursor())
        self._debug("press", {
            "key": cfg.key,
            "phase": "end",
            "session_time": round(press_session, 3),
        })
        return ToggleResult(
            action="ended",
            type_label=cfg.label,
            closed_window=window,
        )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self) -> CancelResult:
        """Cancel the topmost active window: parker first (aside), then
        parkable (r). Scrubs the start marker and resumes any parked
        windows that should now be live again."""
        return self.cancel_at()

    def cancel_at(self) -> CancelResult:
        """Replay-friendly entry point — currently identical to ``cancel``
        since the cancel path doesn't depend on press timing. Kept as a
        sibling of ``toggle_at`` for API symmetry."""
        with self._lock:
            if not self._active:
                return CancelResult(nothing=True)
            target_key: Optional[str] = None
            for k in self._active:
                if self._types[k].can_park_others:
                    target_key = k
                    break
            if target_key is None:
                target_key = next(iter(self._active))
            cfg = self._types[target_key]
            self._timeline.remove_last_marker(cfg.marker_type, "start")
            self._active.pop(target_key)
            resumed: list[str] = []
            if cfg.can_park_others:
                for active_key, w in self._active.items():
                    if self._types[active_key].can_be_parked:
                        w.resume(self._timeline.cursor())
                        resumed.append(self._types[active_key].label)
            return CancelResult(
                cancelled_label=cfg.label,
                resumed_labels=resumed,
            )
