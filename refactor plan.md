# Plan: stream-substitution replay

## Concept

The live system already has the right shape: a **state manager** sits in the middle, fed by several **independent input streams** (transcription outputs, filtered user input). It maintains UI state in response to each event and serves snapshots to the widget on demand.

**Replay = substitute each live stream with a file-reader. Nothing else changes.** Same state manager, same UI, same `/status` endpoint. The state manager doesn't know or care whether a segment came from `faster-whisper` running now or from a file recorded an hour ago.

The performance model is the same as live: O(1) per event ingested, O(1) per `/status` poll (just serializes current state). No rebuild-on-tick, no quadratic anything.

## The input streams (what gets recorded → replayed)

These are the *inputs* the state manager receives. Each is structurally independent: produced by a separate live source, consumable by a separate file-reader.

1. **Fast transcription segments.** `TranscriptionStream(fast).on_segment` → `{ts, seq, kind: "segment", stream: "fast", text, start_time, end_time, words}`. Already a stream, just not recorded as its own file today.
2. **HQ transcription segments.** Same shape, `stream: "hq"`.
3. **Filtered user-action events.** This is the user's "intention" stream — every key that survives `BareDoubleTap`'s filter (the set built at `voice_dictation_persistent.py:285-289`). One event per "the user did something deliberate." Covers all configured action keys, not just clipboard:

   | key | action |
   |---|---|
   | `1`, `2`, ... (configured marker keys) | `marker_press` |
   | `r` | `clipboard_toggle` (recording) |
   | `a` | `clipboard_toggle` (aside) |
   | `x` | `discard` |
   | `q` | `quit` |
   | `m` | `mute_toggle` |
   | `e` (configurable `debug_flag_key`) | `debug_flag` |

   Recorded form: `{ts, seq, kind: "user_action", action, key, payload}`. Today these flow into `_on_action` which then fans out to specialized handlers; the recording captures the consolidated event before fanout.

   **Not in this stream:** raw `_on_press` callbacks for single (un-double-tapped) presses. Those are partial inputs that the filter may or may not turn into a completed action. Only completed/filtered actions belong here.
4. **Audio-window observability events** (VAD boundary, dropped_silent, dropped_queue_full). Already in `debug_events.jsonl`. They drive the timeline SVG but don't change state. Keep recording, but they're a passive observability stream, not a state-driving stream.

A few signals are *outputs* the state manager emits in response to (1-3), not inputs: `paste`, `marker` open/close, `cursor_capture`, capture-mode transitions. They live in the state manager and can be re-emitted on replay without recording — no need to be in the input streams.

So at minimum the recording is **three input streams + one observability stream**. The chunk text files (`chunk_NNN.txt`) and the audio wavs stay as-is — they're outputs, not inputs.

## Refactor: extract the state manager

`PersistentApp` today is both the live machinery (PyAudio recorder, transcription streams, pynput listener) *and* the state holder (timeline, paste log, marker_open_since, capture mode) *and* the snapshot producer (`get_status_snapshot` / `build_status`). Split it:

- **`SessionState`** (new module): owns `TranscriptTimeline`, `paste_log`, `marker_open_since`, `capture` state, `chunk_count`, etc. Two entry points only:
  - `ingest(evt: dict)` — dispatches on `evt["kind"]` and mutates state. Side-effect-free.
  - `snapshot() → dict` — returns the widget payload (today's `build_status()` rewritten to read from here).
- **`PersistentApp`**: keeps the audio/transcription/keyboard machinery. Each live callback (`_on_segment`, the filtered key handler, etc.) becomes a two-liner: build an event dict, call `state.ingest(evt)` *and* write the event to the matching JSONL file. Live mode also keeps the *side-effect* call paths (the paste, the keyboard event simulation) — those don't live in `SessionState`.
- **`replay_session.py`**: instantiates an empty `SessionState`, opens the recorded stream files, and feeds events through `state.ingest` in `(ts, seq)` order as the replay clock advances. `status_provider` returns `state.snapshot()`. That's the whole launcher.

The widget's `StatusServer` and dashboard JS are **unchanged**.

## Ordering ID

Add `seq: int` (monotonic, process-local counter) to every event at log time. Replay merge-sorts by `(ts, seq)` so identical-timestamp events replay in the exact order the live manager saw them. One-line addition in the event-emit path.

## Storage format

Two options:

- **(A) One file, all events**: keep `debug_events.jsonl` as the single recording. Replay reads once. Simplest.
- **(B) File per stream**: `stream_fast.jsonl`, `stream_hq.jsonl`, `stream_user.jsonl`, `stream_observability.jsonl`. Replay opens four file iterators and merge-sorts on `(ts, seq)`. Slightly more complex launcher, but each file is a clean record of one stream — easier to inspect, easier to surgically replace one stream (e.g. "replay this session but with new transcription output").

You explicitly want the stream-per-file model — it makes the substitution mental model concrete. **(B)** it is. The merge-sort is a 10-line generator with `heapq.merge`; no quadratic anywhere.

## What goes away

- `status_snapshots.jsonl` is no longer needed. The state manager builds the same payload on demand, faster and with finer temporal resolution (no 2s sampling).
- `build_status()` in `status_snapshot.py` collapses into `SessionState.snapshot()`.

## Migration steps

1. **Add `seq` field to every event emit.** No behavior change.
2. **Extract `SessionState`.** Move state and snapshot construction out of `PersistentApp` and into `SessionState`. Live mode: each callback emits an event into `SessionState.ingest()`, then performs its side effects as today. Verify widget payload is byte-identical via a snapshot diff. Still write old files (no consumer change yet).
3. **Switch to per-stream files.** Replace `debug_events.jsonl` writes with the four per-stream files. Drop `status_snapshots.jsonl` writes.
4. **Rewrite `replay_session.py`** to drive `SessionState` from the per-stream files.
5. **Delete `build_status` / `status_snapshot.py`** once the replay path is verified.

Steps 1+2 are pure refactor (no observable change). Steps 3-5 are the actual switchover. Each step lands independently and ships green.

## Refactor risks worth flagging

- **`ClipboardWindowManager` holds state too.** Its open/parked/closed windows affect later behavior. Two clean options: (i) move its state into `SessionState` and drive it via `ingest`; (ii) keep it as-is and have `SessionState` ask it for its current view at `snapshot()` time. (i) is purer but a bigger move; (ii) is fine and matches today.
- **Live-only side effects.** Pastes, clipboard reads, simulated keystrokes, force-flush at shutdown — none of these belong inside `SessionState` (replay must not perform them). Keep them in `PersistentApp`. The discipline: events drive state; state changes don't trigger side effects on their own.
- **Force-flush at shutdown** writes a final canonical chunk. Needs to become an explicit event (`{kind: "force_flush"}`) so replay reproduces the same chunk_count.
- **Audio is orthogonal.** Keep `AudioArchiver` and per-chunk wavs as a separate persistence path. The state manager is audio-blind. The dashboard's audio playback reads `now_seconds` + manifest — unaffected by this refactor.

## Relationship to other work

- **The audio-alignment fix** (silence-padding new wavs so wav duration = `t_end - t_start`) is independent. Land it whenever; no coupling to this refactor.
- **Phase B (annotation)** builds cleanly on top of this. Annotation mode uses the same `SessionState` to materialize the segment list per chunk. Random-access scrubbing (jumping to chunk N) needs either "rebuild from start up to chunk N's t_start" (O(events) on selection — fine, sessions are minutes not hours) or in-memory checkpoints every M events (cheap optimization to add later if needed). Either way, no quadratic.
