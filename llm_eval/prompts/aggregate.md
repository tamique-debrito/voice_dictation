You are reading a set of per-transcript summaries that were extracted from a corpus of voice-dictation transcripts (each transcript = one work session by the same software engineer). Each summary was produced independently by examining one transcript. Your job is to merge them into a CORPUS-LEVEL annotation.

The inputs below are separated by `===== BEGIN: <name> =====` / `===== END: <name> =====` banners. Each section is one per-transcript summary.

# Instructions

1. **Identify Areas of Work** — group the per-transcript topics into 5–10 distinct project / problem areas. An "area" is something like "voice dictation pipeline architecture" or "salesforce data sync" — bigger than an individual topic, smaller than "software engineering."
2. **For each area**: write 1–2 sentences describing it AND list the transcript-IDs (the names from the banners) where it appears prominently.
3. **Distinctive vocabulary** — list specific terms / function names / project nouns that are unique or near-unique to each area. Group by area. Generic words ("function", "code") don't count.
4. **Cross-cutting themes** — themes that span multiple areas (e.g. "performance testing" or "training data quality").

Ignore transcripts that the per-transcript summary marked as "no significant content" — don't try to find topics in them.

Output ONLY the markdown below, no preamble.

# Output format

```
## Areas of work

**<Area name>**
- Description: <1–2 sentences>
- Prominent transcripts: <comma-separated transcript-IDs>

(repeat for each area)

## Distinctive vocabulary by area

**<Area name>**: term1, term2, term3, ...

(repeat for each area)

## Cross-cutting themes

- <Theme> — <1-sentence why>
- ...
```

# Per-transcript summaries

{input}
