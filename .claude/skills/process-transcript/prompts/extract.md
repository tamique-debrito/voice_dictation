# Annotation extraction prompt

Use this template when calling the model for each annotation. Substitute the bracketed variables before sending.

---

You are extracting a single structured annotation from a voice-dictation session.

**Marker type:** `{type}`
**Type description (defines what this annotation represents):** {type_description}

**Annotated span (verbatim transcribed speech):**
```
{raw_span}
```

**Context before the annotation (up to ~150 words):**
```
{context_before}
```

**Context after the annotation (up to ~80 words):**
```
{context_after}
```

**Prior annotations in this session (for `dependencies` references):**
{prior_summaries}

**Likely technical identifiers in this codebase (use these to fix transcription errors when applicable):**
{tech_hints}

---

## Your task

Produce a single JSON object representing this annotation. The JSON must include these common fields:

- `index` — integer, will be supplied by the caller (use the value provided)
- `type` — copy `{type}`
- `raw_span` — copy verbatim
- `context_before`, `context_after` — copy verbatim
- `corrected_span` — `raw_span` rewritten with likely transcription errors fixed; preserve the speaker's intent and word choice. If nothing needs correcting, set equal to `raw_span`.
- `transcription_corrections` — array of `{original, corrected, reason}` for each fix you made. Empty array if none.
- `dependencies` — array of integer indices of prior annotations this one depends on or references. Empty array if none.

**Then add type-specific fields based on `{type}` and `{type_description}`:**

- If the type description indicates the annotation is about what an assistant agent should do (e.g. `assistant_annotation`):
  - `trigger` — the situation, observation, or context that should prompt the action
  - `expected_action` — concretely what the assistant should do
  - `expected_inputs` — files, data, or context the assistant needs to have access to
  - `expected_outputs` — tangible deliverables (e.g. a PR, a proposal, a code change)
  - `success_criteria` — how to tell whether the assistant did it correctly
  - `notes` — anything the user said that doesn't fit elsewhere

- If the type description indicates a personal note (e.g. `note`):
  - `summary` — a one-sentence paraphrase of the note
  - `notes` — optional further detail

- For other types, choose fields appropriate to the type description. Aim to capture the *intent* of the annotation in a way a downstream system could later check whether something happened.

## Output format

Return ONLY the JSON object, no surrounding prose. Use `null` only where a field is genuinely not applicable; otherwise prefer empty strings or empty arrays. Do not invent information not implied by the span or context.
