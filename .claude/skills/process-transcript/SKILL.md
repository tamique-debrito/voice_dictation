---
name: process-transcript
description: Convert a persistent voice-dictation session into an annotated structured timeline. Use when the user references a session_id under voice_dictation/transcripts/ or asks to process annotations, extract timeline, or build a validation dataset from a recorded session.
---

# process-transcript

Turns the chunk_*.txt files of a persistent dictation session into a structured `annotations.json` describing what an assistant agent should be expected to do at each marked moment.

## Inputs

The user typically provides a session id (a `YYYYMMDD_HHMMSS` string). The session lives at:

```
voice_dictation/transcripts/{session_id}/
    chunk_000.txt
    chunk_001.txt
    ...
    session_meta.json
```

If the user provides a path instead of a session id, use it directly.

## Procedure

1. **Locate the session.** Resolve the session directory under `voice_dictation/transcripts/`. List all `chunk_*.txt` files and sort them numerically by the `NNN` index in the filename. Read `session_meta.json` for the model info and the `marker_types` list (each entry has `key`, `type`, `description`). The descriptions are essential — they tell you how to frame each marker type's record.

2. **Concatenate** the chunk files in order into a single in-memory string. Do not write this anywhere.

3. **Find marker pairs.** Scan with the regex:

   ```
   <<<MARKER:([a-z_]+):(start|end)>>>
   ```

   Because the recorder enforces non-overlap, pairs are well-formed: every `start` is followed by a matching same-type `end` before any other marker token. Validate this assumption while scanning; on mismatch, log a warning and skip that pair.

   - **Unclosed final start** → close the span at end-of-stream.
   - **Stray end without a start** → ignore.

4. **Build per-annotation context.** For each pair:
   - `raw_span` = text strictly between the two tokens (trim whitespace).
   - `context_before` = up to ~150 words preceding the start token, capped at the previous end-marker if closer.
   - `context_after` = up to ~80 words following the end token, capped at the next start-marker if closer.
   - `type` = the marker type from the token.
   - `index` = sequential index starting at 0.

5. **Run the LLM extraction prompt** for each annotation. The prompt template lives at `prompts/extract.md`. Substitute these variables into it before sending:
   - `{type}`, `{type_description}` (from `session_meta.json`)
   - `{raw_span}`, `{context_before}`, `{context_after}`
   - `{prior_summaries}` — bulleted summaries of all earlier annotations in this session (so the model can reference them in `dependencies`)
   - `{tech_hints}` — list of likely technical identifiers to seed transcription corrections. Build by globbing `*.py` filenames in the workspace root and the `voice_dictation/` directory, plus any obvious capitalised terms from the surrounding `CLAUDE.md` / project READMEs

   The LLM must return JSON conforming to the per-type sub-schema (see `schemas/annotations.schema.json`). Common fields across types: `index`, `type`, `raw_span`, `context_before`, `context_after`, `corrected_span`, `transcription_corrections`, `dependencies`. Type-specific fields vary by `type_description`:
   - `assistant_annotation` → `trigger`, `expected_action`, `expected_inputs`, `expected_outputs`, `success_criteria`, `notes`
   - `note` → `summary`, optional `notes`
   - Unknown / new types → free-form fields decided by the LLM, framed by `type_description`

6. **Assemble** the final structure:

   ```json
   {
     "session_id": "...",
     "source_chunks": ["chunk_000.txt", ...],
     "annotations": [ ... ]
   }
   ```

7. **Validate** each annotation against `schemas/annotations.schema.json` and fix or warn on validation failures.

8. **Write** the result to `voice_dictation/transcripts/{session_id}/annotations.json` (UTF-8, indent 2). Tell the user where it landed.

## Notes

- This skill does not modify the chunk_*.txt files.
- If `annotations.json` already exists, ask the user before overwriting; offer to write to `annotations_v2.json` instead.
- Re-running on the same session should be deterministic in structure, even if the LLM output paraphrases slightly.
