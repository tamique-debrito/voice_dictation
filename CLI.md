# Voice Dictation — CLI Reference

Two entry points live in this directory:

| Script | Purpose |
| ------ | ------- |
| `voice_dictation_persistent.py` | The always-on dictation app. Runs the recorder, fast + HQ transcription streams, clipboard windows, marker hotkeys, status widget, and chunk-file writers. |
| `replay_session.py` | Replays a previously-recorded session in the same widget UI for debugging. No mic access; reads from disk. |

Both are invoked through the project's Python venv:

```
source ../.claude-voice-venv/bin/activate
python voice_dictation_persistent.py [flags]
python replay_session.py <session_dir> [flags]
```

---

## `voice_dictation_persistent.py`

### Flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--config` / `-c PATH` | `local_config.json` next to this script | Override the runtime-config path. Schema mirrors `runtime_config.py:RuntimeConfig`. |
| `--model NAME` | from config | faster-whisper model name for the **fast** stream (e.g. `small.en`, `medium.en`). |
| `--device {auto,cpu,cuda}` | auto-detected | Compute device for the fast-stream model. |
| `--compute TYPE` | `int8` | faster-whisper compute type (`int8`, `int8_float16`, `float16`, `float32`). |
| `--no-widget` | off | Disable the HTTP status widget server entirely. The app still records and writes chunk files. |
| `--open-browser` | off | Auto-open the widget URL in the default browser at startup. Otherwise just prints the URL. |
| `--check-updates` | off | Allow HuggingFace Hub to do a network check for newer model versions on this run. Default is fully-offline model load from cache. |

HQ stream settings are not CLI flags — edit `local_config.json` or use the widget settings modal. See `runtime_config.py:PersistentConfig.hq` for the schema.

### Hotkeys

All hotkeys are **bare double-taps** of the key, with no modifier held. You have ~1s between taps. The pre-paste backspaces undo the visible keystrokes in whichever app currently has focus.

| Key | Action |
| --- | ------ |
| `r` (×2) | Toggle the **recording** clipboard window. First tap starts it; second tap pastes the captured text. |
| `a` (×2) | Toggle an **aside** clipboard window. Parks any active `r` window while it's open; resumes on close. |
| `x` (×2) | Cancel the current window (aside first, else `r`) without pasting. Scrubs the start marker. |
| `m` (×2) | Toggle mic mute. Muted = no audio reaches either stream; window state resets cleanly on un-mute. |
| `e` (×2) | **Flag an error/issue in this moment.** Writes a `debug_flag` event to `debug_events.jsonl` with the current session-time, both stream watermarks, and per-stream accepted/dropped counters. Renders as a tall red vertical line + 🚩 in the widget timeline. |
| `q` (×2) | Quit + finalize the session. Flushes all chunk writers, writes `transcript_final.txt`, closes JSONL handles. |
| user marker keys | Bare double-tap toggles a user-defined annotation marker. Configured in `local_config.json:markers` (defaults: `1` = `assistant_annotation`, `2` = `note`). |

### Output files (per session)

Every session creates a directory under `transcripts_dir` (default `voice_dictation/transcripts/<YYYYMMDD_HHMMSS>/`) containing:

| File | When written | Contents |
| ---- | ------------ | -------- |
| `chunk_NNN.txt` | At every silence boundary once the canonical render accumulates ~75% of `chunk_token_target` words | Canonical (best-priority merged) transcript with marker tokens inline. |
| `chunk_fast_NNN.txt` | Same trigger, fast-stream-tagged | Raw fast-stream text without marker injection. Debug audit trail. |
| `chunk_hq_NNN.txt` | Same trigger, hq-stream-tagged | Raw HQ-stream text without marker injection. Tends to lag fast in count because HQ is slower. |
| `transcript_final.txt` | Shutdown | Full canonical render with markers. The "official" session transcript. |
| `transcript_final.json` | Shutdown | `{spans: [...], markers: [...]}` — per-span source labels (`fast`/`hq`) and the markers list with `audio_time` / `type` / `kind`. |
| `session_meta.json` | Shutdown | Session metadata: id, timestamps, registered streams with model labels + watermarks + chunk counts, all markers with audio times. |
| `debug_events.jsonl` | **Live**, append-only, when debug recording is enabled | Every event written to the in-memory debug ring buffer, one JSON object per line. See [Debug Mode](#debug-mode). |
| `status_snapshots.jsonl` | **Live**, periodic, when debug recording is enabled | Full widget `/status` payload at `debug_snapshot_interval_s` intervals. |

No audio is saved by default — and there is currently no flag to enable saving it. The replay/test feature works purely from the post-session text + event logs.

---

## Debug mode

Debug recording is **on by default**. It produces two append-only JSONL streams under the session directory:

### `debug_events.jsonl`

Every line is one event from the debug ring buffer:

```json
{"ts": 12.345, "kind": "audio_window", "data": {"stream": "fast", "start": 10.1, "end": 12.3, "voiced_ms": 850, "voiced_frac": 0.42, "reason": "silence"}}
{"ts": 14.012, "kind": "segment",      "data": {"stream": "fast", "text": "hello world", "start": 10.2, "end": 12.0, "words": [...]}}
{"ts": 27.500, "kind": "debug_flag",   "data": {"wall_clock": "18:39:23", "fast_watermark": 27.2, "hq_watermark": 19.8, "stream_stats": {...}, "capture_mode": "passive"}}
```

`ts` is session-relative seconds. `kind` is one of:

| `kind` | Carries `data.stream`? | Meaning |
| ------ | ---------------------- | ------- |
| `audio_window` | Yes (`fast`/`hq`) | Aggregator finished a window. `data.reason` is `silence`, `max_window`, `forced`, `dropped_silent`, or `dropped_queue_full`. |
| `segment` | Yes | Transcriber returned a segment. `data.words` has per-word `(text, s, e, p)`. |
| `press` | No | Hotkey double-tap recognized. |
| `marker` | No | User-key marker open/close, or built-in marker emitted. |
| `cursor_capture` | No | Scheduled cursor-capture fired (clipboard-window start). |
| `paste` | No | Clipboard paste completed. |
| `mute` | No | Mic mute state changed. |
| `debug_flag` | No | User double-tapped `e` to flag a moment for debugging. |

Grep workflows:

```bash
# Jump to flagged moments in a session
grep '"kind": "debug_flag"' transcripts/<id>/debug_events.jsonl | jq .

# Count drops per stream
jq -r 'select(.kind=="audio_window") | .data.reason' \
   transcripts/<id>/debug_events.jsonl | sort | uniq -c
```

### `status_snapshots.jsonl`

One full status payload per `debug_snapshot_interval_s` (default 2s). Same schema as the widget's live `/status` endpoint, except `debug_events: []` is stripped (the events live in the other file). Useful for reconstructing live widget state: transcript tail at a moment, paste log history, capture mode, accumulated drop counters.

### Config knobs

In `local_config.json` under `persistent`:

```json
{
  "persistent": {
    "debug_recording": true,
    "debug_snapshot_interval_s": 2.0
  }
}
```

Both editable in the widget settings modal (gear icon → General section). Takes effect on next launch — the JSONL handles are opened in `run()`.

### Disk cost

Both files together are typically a **few hundred KB per 30-minute session**. Debug events dominate when many short windows are produced; status snapshots dominate when sessions are quiet. Cheap enough to leave on always.

---

## `replay_session.py`

Replays a recorded session in the same widget UI, driven by the saved JSONL files.

```
python replay_session.py <session_dir> [--speed FLOAT] [--port INT]
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `session_dir` (positional) | required | Path to a session directory (the timestamped folder under `transcripts/`). |
| `--speed FLOAT` | `1.0` | Playback speed multiplier. `2.0` = 2× faster, `0.5` = half. |
| `--port INT` | `0` (ephemeral) | Widget port. `0` lets the OS pick. |

The replay reuses `widget.py` unchanged. The status endpoint returns the recorded snapshot whose `uptime_seconds <= replay_time`, merged with the slice of debug events ≤ replay_time. The widget timeline + transcript tail + paste log + drop chips advance as they did live.

In replay mode, config-edit PUTs are no-ops so you can't accidentally mutate the recording.

When replay reaches the end it pauses for a few seconds, then loops to the start.

### Typical debug workflow

1. During a session, double-tap `e` whenever something feels wrong.
2. `q` (×2) to quit.
3. Look at console output for the session directory printed at shutdown.
4. `python replay_session.py transcripts/<session_id> --speed 4` to skim through.
5. In the widget timeline, scroll to red 🚩 markers — those are your flagged moments. The 🚩 tooltip shows the watermarks and drop counters at the time of the flag.
6. Drop the speed back to `--speed 1` or even `0.5` to step through the flagged region carefully.

---

## Configuration reference

The full runtime config schema lives in `runtime_config.py`. Every field is editable in the widget settings modal (gear icon). Notable debug-relevant fields:

- `persistent.debug_recording` (bool) — On/off switch for the JSONL writers.
- `persistent.debug_snapshot_interval_s` (float) — Seconds between status snapshots.
- `persistent.fast.window_q_maxsize` (int) — Fast aggregator's queue cap. Hitting full = `dropped_queue_full`. Default 8.
- `persistent.hq.window_q_maxsize` (int) — HQ aggregator's queue cap. Default 64 (HQ falls behind cold and needs the buffer).
- `vad.min_voiced_ms`, `vad.min_voiced_frac` — Thresholds below which the aggregator drops a window as `dropped_silent`. Same for both streams.

If you see frequent `dropped_silent` for HQ but not fast, the most likely cause is HQ's longer windows (`max_window_seconds=40`+) diluting voiced-fraction below the threshold. Tune `vad.min_voiced_frac` down, or lower `max_window_seconds` on HQ.
