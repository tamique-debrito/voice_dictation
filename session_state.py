"""SessionState: single source of truth for the session's UI-visible state.

All state-mutating events flow through ``ingest(evt)``. ``snapshot()`` reads
the resulting state and builds the JSON payload served by the widget's
``/status`` endpoint.

The goal is *stream substitution*: in live mode the live callbacks (PyAudio
recorder, transcription streams, pynput listener) emit events into
``ingest``; in replay mode those same events are read from JSONL files and
fed through the same ``ingest``. The state manager doesn't know or care
which side the events came from.

This refactor is being landed in steps (see ``refactor plan.md``):

* **Step 1** (already done): every event gets a monotonic ``seq`` field.
* **Step 2** (this commit): absorb the ``DebugLog`` ring buffer + the
  ``_paste_log`` / ``_marker_open_since`` mirror state out of
  ``PersistentApp``. Snapshot still reads ``clipboard_mgr`` /
  ``audio_fanout`` directly for ``capture.*`` and ``muted`` — moving those
  into a pure event-driven mirror is Step 4 (when replay starts driving
  the state without those live objects).
* **Step 3**: per-stream JSONL files.
* **Step 4**: rewrite ``replay_session.py`` to feed events through here.
* **Step 5**: delete ``status_snapshot.py``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional


class SessionState:
    def __init__(
        self,
        timeline,
        started_at: datetime,
        device: str,
        compute: str,
        maxlen_events: int = 2000,
        maxlen_paste_log: int = 50,
    ):
        # Owned references — event-driven, fine in both live and replay.
        self.timeline = timeline

        # Immutable session metadata (set once at init).
        self._started_at = started_at
        self._device = device
        self._compute = compute

        # Mirror state: updated by ingest() from recorded events.
        self._marker_open_since: Optional[str] = None
        self._paste_log: deque = deque(maxlen=maxlen_paste_log)
        # Capture state mirror, updated by `press` (r/a phase) + `mute` events.
        # When live components are attached these stay in sync with the live
        # owners (ClipboardWindowManager / AudioFanout); when not attached
        # (replay), they're the only source of truth for snapshot() reads.
        self._r_active: bool = False
        self._aside_active: bool = False
        self._muted: bool = False

        # Event ring buffer (absorbed from DebugLog) + optional JSONL persistence.
        self._events: deque = deque(maxlen=maxlen_events)
        self._lock = threading.RLock()
        self._origin: Optional[float] = None
        self._seq = 0
        # Per-stream output files. When ``set_output_dir`` is called, every
        # event routed by ``_stream_for_kind`` is also appended to its file.
        # Replaces the single ``debug_events.jsonl`` write of the pre-refactor
        # era. ``_out_files`` is keyed by the stream-file basename (without
        # ``.jsonl``); ``_legacy_out_file`` is kept solely so ``set_output_path``
        # (single-file API) still works for tests that don't want per-stream.
        self._out_files: dict = {}
        self._legacy_out_file = None

        # Snapshot-time inputs that are not event-driven. PersistentApp pokes
        # these before each snapshot() so build doesn't need many params.
        self._stream_stats: Optional[dict] = None
        self._hq_active: bool = False
        self._recorder = None

        # Live-mode wiring used by ``_apply_user_action`` to perform the
        # state mutations that previously lived in PersistentApp callbacks.
        # All optional — replay mode passes None and SessionState skips the
        # OS side effects while still tracking mirror state.
        self._audio_fanout = None       # for live mute_toggle
        self._clipboard_mgr = None      # for both live + replay clipboard state
        self._paste_callback = None     # live-only: triggered when window closes
        self._stream_server = None      # live-only: broadcast marker over websocket
        # Stop signal — flipped by user_action(quit). PersistentApp wires
        # its own threading.Event here so the main loop can shut down.
        self._stop_event = None

    # ------------------------------------------------------------------
    # Configuration / wiring from PersistentApp
    # ------------------------------------------------------------------

    def bind_origin(self, origin_monotonic: float) -> None:
        """Set the session-monotonic origin used for ``ts`` on log() calls."""
        self._origin = origin_monotonic

    def set_output_path(self, path: str) -> None:
        """Legacy single-file output (kept for any caller that still wants a
        merged stream). Prefer ``set_output_dir`` so replay can drive a fresh
        SessionState from per-stream files."""
        try:
            self._legacy_out_file = open(path, "a", buffering=1, encoding="utf-8")
        except OSError:
            self._legacy_out_file = None

    def set_output_dir(self, session_dir: str) -> None:
        """Open the four per-stream JSONL files under ``session_dir``.

        Every event ingested is routed by kind to one of:
        - stream_fast.jsonl / stream_hq.jsonl (transcription segments)
        - stream_observability.jsonl (audio_window events)
        - stream_user.jsonl (everything keyboard/clipboard/marker-related)

        Replay rebuilds the session by merging these four files in
        ``(ts, seq)`` order and feeding events through ``ingest``.
        """
        names = ["stream_fast", "stream_hq", "stream_user",
                 "stream_observability"]
        for n in names:
            path = f"{session_dir}/{n}.jsonl"
            try:
                self._out_files[n] = open(path, "a", buffering=1, encoding="utf-8")
            except OSError:
                self._out_files[n] = None

    def close_output(self) -> None:
        if self._legacy_out_file is not None:
            try:
                self._legacy_out_file.close()
            except Exception:
                pass
            self._legacy_out_file = None
        for n, f in list(self._out_files.items()):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        self._out_files.clear()

    # Routing table: which per-stream file each event kind lands in.
    _STREAM_FOR_KIND = {
        "segment": None,           # handled specially (split fast vs hq)
        "audio_window": "stream_observability",
        "press": "stream_user",
        "marker": "stream_user",
        "mute": "stream_user",
        "paste": "stream_user",
        "cursor_capture": "stream_user",
        "debug_flag": "stream_user",
        "user_action": "stream_user",
    }

    def _file_for_event(self, evt: dict):
        kind = evt.get("kind")
        if kind == "segment":
            stream = (evt.get("data") or {}).get("stream", "")
            if stream == "hq":
                return self._out_files.get("stream_hq")
            return self._out_files.get("stream_fast")
        name = self._STREAM_FOR_KIND.get(kind, "stream_user")
        return self._out_files.get(name) if name else None

    def attach_live_components(self, *, recorder, audio_fanout, clipboard_mgr) -> None:
        """Live-mode helpers PersistentApp wires once at startup.

        - ``recorder`` provides ``started_monotonic`` for ``now_seconds``.
        - ``audio_fanout`` is the live mute owner — ``_apply_user_action``
          flips its mute when it sees a ``mute_toggle`` event.
        - ``clipboard_mgr`` is the clipboard-window state machine; it's
          owned by SessionState in both live and replay modes (replay
          mode passes a fresh instance via ``attach_clipboard_manager``)."""
        self._recorder = recorder
        self._audio_fanout = audio_fanout
        self._clipboard_mgr = clipboard_mgr

    def attach_clipboard_manager(self, clipboard_mgr) -> None:
        """Replay-mode wiring — caller constructs a clipboard manager
        with a no-op debug_log and no paste side effects, then hands it
        here. Live mode also uses this through ``attach_live_components``."""
        self._clipboard_mgr = clipboard_mgr

    def attach_paste_callback(self, cb) -> None:
        """Called by ``_apply_user_action`` when a clipboard window closes,
        with the ``ClipboardWindow`` instance whose contents should be
        pasted. Live mode wires this to the OS paste path; replay leaves
        it unset and the closed-window content is never pasted."""
        self._paste_callback = cb

    def attach_stream_server(self, server) -> None:
        """Live-only: marker broadcast over the transcript-stream WebSocket
        when a marker is opened/closed via a ``marker_press`` action."""
        self._stream_server = server

    def attach_stop_event(self, ev) -> None:
        """``user_action(quit)`` flips this event to shut the app down."""
        self._stop_event = ev

    def set_stream_stats(self, stats: dict) -> None:
        self._stream_stats = stats

    def set_hq_active(self, active: bool) -> None:
        self._hq_active = active

    # ------------------------------------------------------------------
    # Direct mirror-state updates (callers that produce data the event
    # doesn't carry yet — see refactor-plan step 4 for unification).
    # ------------------------------------------------------------------

    def append_paste_log_entry(self, entry: dict) -> None:
        """Append a paste-log entry. Today the persisted ``paste`` event
        carries ``preview`` but not ``full`` text, while the widget paste
        log wants ``full``. Step 4 will unify these so ingest() can build
        the entry from the event alone."""
        with self._lock:
            self._paste_log.append(entry)

    # ------------------------------------------------------------------
    # Event ingest (the single entry point for state mutation)
    # ------------------------------------------------------------------

    def _now(self) -> float:
        if self._origin is None:
            return 0.0
        return time.monotonic() - self._origin

    def log(self, kind: str, data: dict, ts: Optional[float] = None) -> None:
        """Back-compat shim for DebugLog.log(). Equivalent to constructing an
        event with the current ts and forwarding to ingest()."""
        evt = {
            "ts": round(ts if ts is not None else self._now(), 3),
            "kind": kind,
            "data": data,
        }
        self.ingest(evt)

    def ingest(self, evt: dict) -> None:
        """Record + apply one event. Single mutation entry point."""
        with self._lock:
            # Stamp seq if missing (live emits won't have one; replay events
            # come pre-stamped). Track max so subsequent log() calls don't
            # collide with values loaded from a file.
            if "seq" not in evt:
                self._seq += 1
                evt = {**evt, "seq": self._seq}
            else:
                if evt["seq"] > self._seq:
                    self._seq = int(evt["seq"])

            self._events.append(evt)

            if self._legacy_out_file is not None:
                try:
                    self._legacy_out_file.write(json.dumps(evt) + "\n")
                except Exception:
                    pass
            if self._out_files:
                f = self._file_for_event(evt)
                if f is not None:
                    try:
                        f.write(json.dumps(evt) + "\n")
                    except Exception:
                        pass

            self._apply(evt)

    def _apply(self, evt: dict) -> None:
        """Update mirror state in response to an event. Called under the
        ingest() lock. Side-effect-free w.r.t. external systems — only
        mutates ``self.*`` state. Live-only side effects (paste, audio
        toggles, marker injection into the timeline) stay in PersistentApp.

        In live mode the existing live owners (ClipboardWindowManager,
        AudioFanout, TranscriptTimeline) do the *real* state change as a
        side effect of the live callbacks; this mirror update keeps a
        replay-friendly copy. In replay mode there are no live owners and
        this mirror is the single source of truth for ``snapshot()``."""
        kind = evt.get("kind")
        data = evt.get("data", {})

        if kind == "segment":
            # Replay path: live mode already calls timeline.ingest_segment()
            # directly from the streaming callback, but we re-run it here
            # for idempotency. The timeline guards against duplicate ranges,
            # and replay needs us to advance the transcript state from the
            # recorded segment events.
            words = data.get("words") or []
            try:
                self.timeline.ingest_segment(
                    data.get("stream", "fast"),
                    data.get("text", ""),
                    float(data.get("start", 0.0)),
                    float(data.get("end", 0.0)),
                    words,
                )
            except Exception:
                pass
        elif kind == "user_action":
            self._apply_user_action(data)
        elif kind == "paste":
            # Paste events are still emitted by the live paste path
            # (PersistentApp._finish_paste) because they carry the resolved
            # ``full`` text + drain timing that the clipboard state machine
            # alone doesn't know. Recorded for the paste_log mirror; could
            # be derived from segments + window spans in a later refactor.
            if data.get("full") is not None:
                self._paste_log.append({
                    "ts": data.get("ts_wall", ""),
                    "label": data.get("label"),
                    "preview": data.get("preview", ""),
                    "full": data.get("full", ""),
                })
        # press / marker / mute / cursor_capture / debug_flag events are
        # legacy observability — they still flow into the event ring
        # buffer for the debug panel, but they no longer drive state.
        # State updates come from user_action exclusively.

    # ------------------------------------------------------------------
    # User-action dispatch (driven by the ``user_action`` input stream)
    # ------------------------------------------------------------------

    def _apply_user_action(self, data: dict) -> None:
        action = data.get("action")
        key = data.get("key", "")
        session_time = float(data.get("session_time", 0.0))

        if action == "marker_press":
            # ``timeline.insert_marker`` resolves the key against the
            # marker_types config it was constructed with, decides open
            # vs close, and updates the timeline's marker state. Returns
            # a list of (action, type_name) tuples. On live this is the
            # same call PersistentApp._handle_marker used to make; on
            # replay we're running against a replay-mode timeline that
            # has the same marker_types config from session_meta.
            try:
                events = self.timeline.insert_marker(key)
            except Exception:
                events = []
            for evt_action, type_name in events:
                if evt_action == "open":
                    self._marker_open_since = datetime.now().strftime("%H:%M:%S")
                elif evt_action == "close":
                    self._marker_open_since = None
                if self._stream_server is not None and evt_action != "flag":
                    try:
                        audio_time = self.timeline.fast_high_watermark()
                        self._stream_server.broadcast_marker(
                            evt_action, type_name, audio_time
                        )
                    except Exception:
                        pass

        elif action == "clipboard_toggle":
            cm = self._clipboard_mgr
            if cm is None:
                return
            try:
                # The clipboard manager owns its own clock conversion for
                # the live ``toggle()`` entry point, but for replay we feed
                # the recorded session_time directly via ``toggle_at``.
                press_ts = session_time + (
                    self._recorder.started_monotonic
                    if self._recorder is not None and self._recorder.started_monotonic
                    else 0.0
                )
                result = cm.toggle_at(key, press_ts, session_time)
            except Exception:
                return
            if key == "r":
                self._r_active = result.action == "started"
            elif key == "a":
                self._aside_active = result.action == "started"
            if result.action == "ended" and result.closed_window is not None:
                if self._paste_callback is not None:
                    try:
                        self._paste_callback(result.closed_window)
                    except Exception:
                        pass

        elif action == "discard":
            cm = self._clipboard_mgr
            if cm is None:
                return
            try:
                cm.cancel_at()
            except Exception:
                return
            # Mirror update — discard closes the topmost active window.
            # If r was the topmost (no aside), r_active flips off; if aside
            # was topmost, aside_active flips off and r stays as it was.
            self._r_active = cm.is_active("r") if hasattr(cm, "is_active") else False
            self._aside_active = cm.is_active("a") if hasattr(cm, "is_active") else False

        elif action == "mute_toggle":
            # Live: flip the fanout's mute (the actual audio drop). Replay:
            # mirror-only.
            if self._audio_fanout is not None:
                try:
                    new_muted = self._audio_fanout.toggle_muted()
                    self._muted = bool(new_muted)
                    return
                except Exception:
                    pass
            self._muted = not self._muted

        elif action == "quit":
            if self._stop_event is not None:
                try:
                    self._stop_event.set()
                except Exception:
                    pass

        # debug_flag: nothing to apply. The event already appended itself
        # to the ring buffer in ingest(); the timeline SVG renders it.

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def events_snapshot(self, limit: int = 500) -> list:
        """Recent debug events for the widget's debug-log panel."""
        with self._lock:
            if limit >= len(self._events):
                return list(self._events)
            return list(self._events)[-limit:]

    def snapshot(self, now_seconds: Optional[float] = None) -> dict:
        """Build the JSON payload served by ``/status``.

        Equivalent to the old ``build_status()``. In live mode the live
        components (clipboard_mgr, audio_fanout, recorder) are the source
        of truth for capture / muted / now_seconds — but the event-driven
        mirror tracks the same values, so omitting them (as replay does)
        still produces a coherent payload. ``now_seconds`` is taken from
        the override if provided (replay), else from the recorder's
        started_monotonic, else 0."""
        timeline = self.timeline
        with self._lock:
            paste_log = list(self._paste_log)
            marker_open_since = self._marker_open_since
            events = list(self._events)[-500:]
            r_active_mirror = self._r_active
            aside_active_mirror = self._aside_active
            muted_mirror = self._muted

        # Prefer live owners when attached (live mode); fall back to the
        # event-driven mirror (replay or any pre-attach moment).
        cm = self._clipboard_mgr
        af = self._audio_fanout
        rec = self._recorder

        r_active = cm.is_active("r") if cm is not None else r_active_mirror
        aside_active = cm.is_active("a") if cm is not None else aside_active_mirror
        muted = af.is_muted() if af is not None else muted_mirror

        if r_active and aside_active:
            mode = "r+aside"
        elif r_active:
            mode = "r"
        elif aside_active:
            mode = "aside"
        else:
            mode = "passive"

        open_marker = timeline.open_marker_type()
        if open_marker is None:
            marker_open_since = None

        if now_seconds is not None:
            now_seconds = round(float(now_seconds), 3)
        elif rec is not None:
            now_seconds = round(
                time.monotonic() - (rec.started_monotonic or time.monotonic()),
                3,
            )
        else:
            now_seconds = 0.0

        payload = {
            "session_id": timeline.session_id,
            "session_dir": timeline.session_dir,
            "model": timeline.model_label("fast"),
            "device": self._device,
            "compute": self._compute,
            "started_at": self._started_at.isoformat().replace("+00:00", "Z"),
            "uptime_seconds": (
                datetime.now(tz=timezone.utc) - self._started_at
            ).total_seconds(),
            "chunk_count": timeline.chunk_count(),
            "capture": {
                "mode": mode,
                "r_active": r_active,
                "aside_active": aside_active,
                "muted": muted,
            },
            "open_marker": open_marker,
            "open_marker_since": marker_open_since,
            "paste_log": paste_log,
            "transcript_tail": timeline.tail(2000),
            "now_seconds": now_seconds,
            "debug_events": events,
        }
        if self._hq_active and timeline.stream_exists("hq"):
            payload["hq"] = {
                "model": timeline.model_label("hq"),
                "chunk_count": timeline.chunk_count("hq"),
                "leading_edge_seconds": round(timeline.hq_high_watermark(), 3),
            }
            payload["transcript_tail_merged"] = timeline.tail_annotated(2000)
        if self._stream_stats is not None:
            payload["stream_stats"] = self._stream_stats
        return payload
