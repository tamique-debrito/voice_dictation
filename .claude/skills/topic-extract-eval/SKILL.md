---
name: topic-extract-eval
description: Prompt-engineering workbench for benchmarking local LLMs (via Ollama) on extracting topics + distinctive vocabulary from voice-dictation transcripts. Two phases — per-transcript extraction, then aggregation across per-transcript summaries — graded against a hand-curated `ground_truth.md`. Use when the user wants to iterate on prompts for the local-LLM topic-extraction task, add a new model to the benchmark, or compare a run against ground truth.
argument-hint: <model-tag> [prompt-version]
---

# Topic Extraction — Local LLM Eval Workbench

A scaffold for prompt-engineering iteration: given the voice-dictation transcript corpus and a ground-truth reference, run small local LLMs through Ollama with successive prompt revisions and grade the outputs.

## Layout

```
voice_dictation/llm_eval/
├── prompts/                            # prompt templates (markdown w/ {input})
│   ├── per_transcript_v1.md
│   ├── aggregate_v1.md
│   └── ... (add new versions as v2, v3 ...)
├── runs/                               # outputs go here
│   ├── per_transcript_v1__<model>__<transcript>.md
│   └── aggregate_v1__<model>.md
└── ground_truth.md                     # human-curated reference (copied or symlinked
                                         # from scratch/llm_eval/ground_truth.md)
```

Transcripts live at `scratch/transcripts_text/` (built by `voice_dictation/tools/export_transcripts.py` + a one-off v1 dump script — see the parent `voice-dictation-finetune` skill for context).

## Two-phase workflow

**Phase 1 — per-transcript extraction.** Local LLM reads one transcript, outputs a structured per-transcript summary (Topics, Vocabulary, Summary, "no significant content?"). Iterate the prompt until quality on the benchmark set matches `ground_truth.md` Section 2.

**Phase 2 — aggregation.** Concatenate the Phase-1 outputs (across all transcripts the LLM has summarized), feed them to a different prompt that asks the LLM to merge them into a corpus-level annotation (Areas of work, Vocabulary by area, Cross-cutting themes). Iterate until output matches `ground_truth.md` Section 1.

Both phases use the same CLI — only the prompt and inputs differ.

## Step 1 — Make sure Ollama is running

```bash
pgrep -f "ollama serve" >/dev/null || ollama serve &
ollama list   # sanity-check models installed
```

Available models (see `local_llm_tests/models.yaml`): `llama3.1:8b-instruct-q4_K_M`, `phi4:latest`, `mistral-small3.2:24b-instruct-2506-q4_K_M`, `gemma4:31b-it-q4_K_M`, `gemma4:26b-a4b-it-q4_K_M`, `qwen3.6:27b-q4_K_M`, `qwen3.6:35b-a3b-q4_K_M`.

## Step 2 — Run a per-transcript prompt against one transcript

```bash
python -m voice_dictation.tools.llm_eval \
    --prompt   voice_dictation/llm_eval/prompts/per_transcript_v1.md \
    --input    scratch/transcripts_text/regen_20260515_193718_p64371_20260514_160743_audio.txt \
    --model    llama3.1:8b-instruct-q4_K_M \
    --num-ctx  8192 \
    --output   voice_dictation/llm_eval/runs/per_transcript_v1__llama31__regen_160743.md
```

Output file includes a header (model, prompt, timing, token counts) followed by the model's response.

### CLI flags

| Flag | Meaning |
|---|---|
| `--prompt <file>` | Prompt template (must contain literal `{input}` placeholder). |
| `--input <file>...` | One or more input files. Multiple files are concatenated with banner separators between them (used in Phase 2 aggregation). |
| `--model <tag>` | Ollama tag. |
| `--num-ctx N` | Ollama `num_ctx`. Default 8192. Bump for large inputs (Phase 2 may need 16384–32768). Practical ceiling for llama3.1-8b is ~16K. |
| `--temperature F` | Default 0.2 (deterministic-ish). |
| `--think` / `--no-think` | Toggle thinking on supported models (qwen3, gpt-oss). llama3.1 ignores. |
| `--output <file>` | Output markdown. Auto-creates the parent dir. |

## Step 3 — Batch over the benchmark set

The benchmark set is the 7–8 transcripts named in `ground_truth.md` Section 2. Run the current prompt against each:

```bash
PROMPT=voice_dictation/llm_eval/prompts/per_transcript_v1.md
MODEL=llama3.1:8b-instruct-q4_K_M
TAG=p1_v1__llama31

# Extract benchmark filenames from ground_truth.md (lines like "#### `xxx.txt`")
grep -oE '#### \`[^`]+\.txt`' voice_dictation/llm_eval/ground_truth.md \
  | sed 's/.*`\([^`]*\)`/\1/' | while read f; do
    out="voice_dictation/llm_eval/runs/${TAG}__${f%.txt}.md"
    python -m voice_dictation.tools.llm_eval \
        --prompt "$PROMPT" \
        --input "scratch/transcripts_text/$f" \
        --model "$MODEL" \
        --num-ctx 8192 \
        --output "$out"
done
```

## Step 4 — Grade Phase-1 against ground truth

Spot-check by eye, OR have a stronger LLM grade the outputs. A quick eyeball compare:

```bash
# Side-by-side for one transcript
diff -y --width=200 \
    <(sed -n '/^#### \`regen_X/,/^####/p' voice_dictation/llm_eval/ground_truth.md) \
    voice_dictation/llm_eval/runs/p1_v1__llama31__regen_X.md
```

Things to grade on:
- Did it correctly say "no significant content" for the minimal transcripts?
- Are the topics specific (good) or generic (bad)?
- Is the distinctive vocabulary actually drawn from the transcript?
- Is the summary accurate vs hallucinated?

When Phase-1 quality looks reasonable, move to Phase 2.

## Step 5 — Phase 2: aggregate

Feed the Phase-1 outputs (the substantive ones; ignore the "no significant content" cases) to the aggregation prompt:

```bash
python -m voice_dictation.tools.llm_eval \
    --prompt   voice_dictation/llm_eval/prompts/aggregate_v1.md \
    --input    voice_dictation/llm_eval/runs/p1_v1__llama31__regen_*.md \
    --model    llama3.1:8b-instruct-q4_K_M \
    --num-ctx  16384 \
    --output   voice_dictation/llm_eval/runs/p2_v1__llama31.md
```

Compare against `ground_truth.md` Section 1.

## Iterating prompts

When something doesn't work, copy the prompt to a new version:

```bash
cp voice_dictation/llm_eval/prompts/per_transcript_v1.md \
   voice_dictation/llm_eval/prompts/per_transcript_v2.md
# edit v2, then re-run Step 3 with the new prompt, distinct output tag.
```

Common things that help small-LLM prompts:
- Show the exact output format with a literal example (few-shot).
- Lead with the decision rule ("first, decide if this is meaningful or not").
- Use bullet markers to constrain shape.
- Add a refusal example ("if empty, output: …").
- Lower temperature.

## Things to watch out for

- **Context window**: llama-3.1-8b q4 has practical ceiling ~16 K tokens; quality drops measurably past that. The biggest single transcript is ~10 K tokens, fine. Phase-2 aggregation across all 8 per-transcript summaries is well under 8 K — no issue.
- **Thinking models**: qwen3.x is hybrid-thinking. `--think` exposes the `thinking` field separately in the output file. Don't grade the thinking; grade the response.
- **Hallucinated topics on minimal content**: the test for whether a prompt is robust. If the model invents topics for a `v1_*` trivia-question transcript, the prompt isn't strict enough.

## Quick reference — files

| Path | Role |
|---|---|
| `voice_dictation/tools/llm_eval.py` | CLI (Ollama generate, single-shot) |
| `voice_dictation/llm_eval/prompts/*.md` | prompt templates with `{input}` placeholder |
| `voice_dictation/llm_eval/runs/*.md` | run outputs (model response + run metadata) |
| `voice_dictation/llm_eval/ground_truth.md` | curated reference (or symlinked to `scratch/llm_eval/ground_truth.md` while iterating) |
| `scratch/transcripts_text/` | plain-text transcripts (`regen_*.txt`, `v1_*.txt`) |
