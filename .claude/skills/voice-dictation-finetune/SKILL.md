---
name: voice-dictation-finetune
description: End-to-end pipeline for turning a folder of raw audio files into a fine-tuned Whisper model for the voice_dictation app. Use when the user wants to (a) generate transcripts from existing audio for annotation, (b) auto-annotate transcripts to produce training data, (c) review annotations in the multi-session UI, or (d) train + deploy a new fast-stream model. Trigger phrases include "fine-tune from these audio files", "regenerate sessions from this folder", "annotate this audio dump", "train a new whisper from <folder>".
argument-hint: <path/to/audio_folder>
---

# Voice-Dictation Fine-Tune from Audio Folder

Turn a folder of raw audio (or v1 transcript dirs) into a fine-tuned Whisper model the live dictation app can load. The pipeline is six steps. Each step is a single command — only step 4 (review) needs a human.

The user gives you a path to a **folder of audio**. That folder can hold:
- Loose 16 kHz mono int16 WAVs (one per session), OR
- Subdirectories each containing a chunked-WAV layout like the legacy `voice_dictation/transcripts/<ts>/audio/chunk_*.wav` (in which case use `--concat`).

If the audio format is unknown, check with `file <path>`. Non-16-kHz inputs require conversion first (`ffmpeg -i in.mp3 -ar 16000 -ac 1 -sample_fmt s16 out.wav`).

## Step 1 — Activate the venv

```bash
source /Users/tamiquedebrito/Documents/Repositories/Repository\ copy/.claude-voice-venv/bin/activate
```

All commands below assume the cwd is `Repository copy/`.

## Step 2 — Regenerate sessions from audio

This runs each audio file through the full v2 dictation pipeline offline (no mic). Output goes to `voice_dictation/regenerated_sessions/<regen_*>/` with the same JSONL streams a live session produces (`hq_segments.jsonl`, `fast_segments.jsonl`, `audio/manifest.json`, etc.), plus a `regenerated.json` provenance marker.

```bash
# Loose WAVs:
python -m voice_dictation.tools.regenerate_session \
    --parallel 4 --ct2-threads 3 \
    <folder>/*.wav

# Chunked v1-style transcript dirs:
python -m voice_dictation.tools.regenerate_session \
    --concat --parallel 4 --ct2-threads 3 \
    <folder>/*/audio
```

Tuning knobs:
- `--parallel N` — worker processes. Default 1. On a 14-core / 36 GB machine, 4 with `--ct2-threads 3` is the sweet spot. Drop to `--parallel 2 --ct2-threads 6` to leave half the box for other work.
- `--ct2-threads N` — per-worker CT2/OMP thread cap. 0 = unbounded (only sensible when `--parallel 1`).
- `--out-dir <path>` — override default `voice_dictation/regenerated_sessions/`.

**Time budget**: HQ on CPU at ~3× realtime per process; 4 in parallel → roughly **1/12 audio-duration** wall-clock plus a ~30 s drain per session at end. 6 hours of audio ≈ 35–45 minutes.

Output sanity check:
```bash
ls voice_dictation/regenerated_sessions/regen_*/regenerated.json | wc -l   # should equal input count
find voice_dictation/regenerated_sessions -name hq_segments.jsonl | wc -l
```

## Step 3 — Auto-annotate with one Sonnet sub-agent

The annotator UI loads `annotations.jsonl` per session. To pre-fill it, dispatch one Sonnet sub-agent that reads the (post-boundary-split) view of every session and writes clean transcriptions per segment.

### Step 3a — Export per-session views

```bash
python -m voice_dictation.tools.export_annotator_view \
    voice_dictation/regenerated_sessions/regen_*
```

This writes `<session>/annotator_view.json`, a compact list of chunks + segments with the **post-split** `segment_id`s the annotator UI looks up (e.g. `hq:734` or `hq:734:p1` for boundary-straddling pieces). The sub-agent must annotate against this view, NOT the raw `*_segments.jsonl`.

### Step 3b — Token-budget check + sub-agent dispatch

Measure total HQ text size first:

```bash
find voice_dictation/regenerated_sessions -name hq_segments.jsonl \
  -exec cat {} \; | python3 -c "import sys,json; print(sum(len(json.loads(l).get('text','')) for l in sys.stdin if l.strip()))"
```

If under ~200k chars (~50k tokens), one sub-agent can handle all sessions with full cross-session context. If larger, split sessions into ~50k-token groups by date or by session size.

Dispatch one (or more) **Sonnet** sub-agent via the Agent tool. The brief:

> You are auto-annotating N regenerated voice-dictation sessions to produce clean training data. For each session at `<session>/annotator_view.json`, read every chunk + segment and write `<session>/annotations.jsonl`. **One JSONL row per segment**; use the exact `segment_id` from the view (e.g. `hq:734:p1`).
>
> Schema per row:
> ```json
> {"ts": <int unix ms>, "chunk_idx": int, "segment_id": "...", "text": "...",
>  "status": "edited" | "accepted_hq" | "accepted_fast" | "rejected",
>  "source_text_fast": null, "source_text_hq": "<original HQ if present>",
>  "notes": "<short reason if edited>"}
> ```
>
> Rules:
> - `accepted_hq`/`accepted_fast` if the original is clean — `text` equals original.
> - `edited` with cleaned `text` if you stripped fillers ("um", "uh", filler "like"/"you know"), collapsed immediate repetitions, removed Whisper hallucinations ("Thanks for watching!", "♪", "[Music]", repeated 3+ times), or fixed obvious wrong words using neighboring/cross-session context.
> - `rejected` ONLY at chunk level (`segment_id: null, text: null`) — pure noise chunks.
> - If a segment is entirely hallucination: `edited` with `text: ""`.
> - Preserve content; do not paraphrase.
>
> All sessions are the same speaker talking about (skim first to absorb terminology): voice-dictation pipeline architecture, Whisper / faster-whisper inference, fine-tuning, the annotation UI, training data, audio chunks, accepted-ms clock, EventBus.
>
> Output: write each session's `annotations.jsonl` via a single Write tool call (overwrite — file doesn't exist yet). Verify line count matches segment count before reporting done.

Verify after the agent returns:
```bash
python3 -c "
import json
from pathlib import Path
for sd in sorted(Path('voice_dictation/regenerated_sessions').glob('regen_*')):
    view = json.loads((sd/'annotator_view.json').read_text())
    n_view = sum(len(c['segments']) for c in view['chunks'])
    ann = sd / 'annotations.jsonl'
    if not ann.exists():
        print(f'MISSING: {sd.name}'); continue
    rows = [json.loads(l) for l in ann.open() if l.strip()]
    seg_rows = sum(1 for r in rows if r.get('segment_id') is not None)
    flag = '' if seg_rows == n_view else '  <<< MISMATCH'
    print(f'{sd.name}: view={n_view} ann={seg_rows}{flag}')
"
```

## Step 4 — Review in the multi-session annotator UI

Launch the picker-mode annotator on the regen folder:

```bash
python -m voice_dictation.annotate_session --root voice_dictation/regenerated_sessions
```

Open the URL it prints. UI controls:
- Session dropdown at the top — picks which session to view.
- Left rail — chunks within the current session. `↑`/`↓` switches chunks.
- `[` / `]` switches sessions.
- Click any timestamp to seek the audio; space toggles play/pause.
- Edit text in the right column — auto-saves on blur. Word-level diff shows red strikethrough for removed, green underline for added.
- `r` toggles chunk-reject (skips the whole chunk from the dataset).

When the user is happy, kill the server (Ctrl-C in foreground, or `kill <pid>` if backgrounded).

Tell the user the keyboard map and let them drive review. **Don't pre-mark sessions as reviewed for them.**

## Step 5 — Build the training dataset

```bash
python -m voice_dictation.tools.build_dataset \
    --sessions voice_dictation/regenerated_sessions/regen_* \
    --out voice_dictation/datasets/<dataset_name>
```

This slices each accepted segment's audio out of the source WAV, writes per-segment WAV clips, and emits `manifest.jsonl` + `train.jsonl` + `val.jsonl` (split by session). Rejected chunks are dropped. By default, segments without an annotation row are skipped — pass `--include-unannotated` to fall back to HQ text for those. `--include-boundary` includes boundary-straddling pieces (off by default).

## Step 6 — Fine-tune + deploy

Training is on Apple Silicon MPS via a custom LoRA loop (NOT `Seq2SeqTrainer` — PEFT's seq2seq wrapper is hardcoded for T5):

```bash
# Install training deps once:
pip install -r voice_dictation/tools/requirements-training.txt

python -m voice_dictation.tools.finetune_whisper \
    --dataset voice_dictation/datasets/<dataset_name> \
    --output  voice_dictation/models/finetune_<date>_hf \
    --base-model openai/whisper-tiny.en \
    --epochs 3
```

Defaults: tiny.en (matches `local_config.json → fast.fw.model`), rank-16 alpha-32 LoRA on `q/k/v/out_proj`, batch 4 × grad-accum 4, lr 1e-4. Override `--base-model openai/whisper-small.en` to fine-tune larger.

After training, LoRA is merged into base weights and saved at `--output`. Convert to CT2:

```bash
voice_dictation/tools/convert_to_ct2.sh voice_dictation/models/finetune_<date>_hf
# writes <hf_dir>_ct2/ with int8 quantization
```

Swap in via the widget config panel: open the live app's widget URL, the model dropdown will show `📁 finetune_<date>_hf_ct2` under the canonical hub models. Or edit `voice_dictation/local_config.json` directly:

```json
"fast": {"fw": {"model": "voice_dictation/models/finetune_<date>_hf_ct2"}}
```

Restart the live app — or, when only the `fw.model` / `fw.compute` / `fw.beam_size` / `fw.condition_on_previous_text` changed, the widget will hot-reload the affected transcribers on the next window (the toast reports which streams reloaded). Other fields (window sizes, preprocessor, device) still need a restart.

## Side utility — plain-text transcript export

For pasting transcripts into an external analyzer (or feeding into a local LLM downstream), `tools/export_transcripts.py` emits one `.txt` per session with a header (session name, recorded time, source audio for regenerated sessions, exported_at). Uses the same merge logic the annotator UI does (HQ-preferred + annotation overrides + skip rejected chunks).

```bash
# Per-session files in an out dir:
python -m voice_dictation.tools.export_transcripts \
    --root voice_dictation/regenerated_sessions \
    --out  voice_dictation/transcripts_text

# Date-filter + combined into one file (good for paste-into-LLM):
python -m voice_dictation.tools.export_transcripts \
    --root voice_dictation/sessions \
    --since 2026-05-13 --until 2026-05-15 \
    --combined --out voice_dictation/transcripts_text

# Specific session dirs (positional, may be mixed with --root):
python -m voice_dictation.tools.export_transcripts \
    voice_dictation/sessions/20260515_224523 \
    --out voice_dictation/transcripts_text
```

Notes:
- `--since` / `--until` filter by the leading `YYYYMMDD_HHMMSS` stamp in the session dir name (works for both live and `regen_*` dirs). `--until` is inclusive (end-of-day).
- Segments are paragraph-broken on silence gaps ≥ 1.5 s.
- Empty-text annotations (hallucinations the annotator wiped) are dropped.
- Rejected chunks are skipped entirely.

## Quick reference — files

| Path | Role |
|---|---|
| `voice_dictation/tools/regenerate_session.py` | Step 2 — feed audio through pipeline offline |
| `voice_dictation/tools/export_annotator_view.py` | Step 3a — emit annotator view JSON |
| `voice_dictation/annotate_session.py` | Step 4 — launch annotator (single or `--root` multi mode) |
| `voice_dictation/tools/build_dataset.py` | Step 5 — slice clips + manifest |
| `voice_dictation/tools/finetune_whisper.py` | Step 6 — LoRA train on MPS |
| `voice_dictation/tools/export_transcripts.py` | Side — plain-text transcript dump (for downstream analysis / local LLM) |
| `voice_dictation/tools/convert_to_ct2.sh` | Step 6 — HF → CT2 conversion |
| `voice_dictation/FINETUNE_PLAN.md` | Deeper design notes for steps 2–6 |

## Failure modes seen in practice

- **Concat-mode name collision when multiple input dirs are named `audio`**: regenerator uses `<parent>_<name>.wav` for scratch; verify scratch dir after run if you see overwritten outputs.
- **HQ shows windows=0**: the drain loop in `regenerate_session._drain` waits up to 30 s of idle after the last completed window before shutdown. Long HQ inferences need that buffer — don't lower the idle threshold.
- **Annotator agent's IDs don't match UI**: always export via `export_annotator_view.py` first. The view applies boundary-splitting; if you skip it the agent writes `hq:734` but the UI looks up `hq:734:p1`.
- **Training crashes with "input_ids got multiple values"**: PEFT's `PeftModelForSeq2SeqLM` collides with Whisper's input_features signature. The script avoids it by **not** passing `task_type=TaskType.SEQ_2_SEQ_LM` — don't add that.
- **Drain returns before HQ inference completes**: counter only increments after inference completes; if you see `windows=0` in HQ shutdown despite a queued window, raise `idle_seconds` in `_drain`.
