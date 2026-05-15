You are analyzing a single voice-dictation transcript from a software engineer. Your job is to identify the topics they're discussing, capture distinctive vocabulary, and produce a concise ground-truth summary. The transcript is the dictation output of someone thinking aloud or asking questions to an assistant about their codebase, so it may be fragmented, contain false starts, and include filler — extract MEANING, not surface form.

# Instructions

1. **Decide first**: does this transcript contain meaningful work content, or is it primarily test phrases, off-topic trivia questions, or filler? If the latter, output `"no significant content"` for the summary and leave Topics empty. DO NOT hallucinate topics if the content is empty.

2. **If there is real content**, identify:
   - **Topics** — specific technical / project topics discussed. Order most-specific-first. Avoid generic terms ("software", "code") in favor of specific ones ("audio windower", "LoRA fine-tuning").
   - **Key vocabulary** — distinctive terms, technical jargon, project-specific names (function names, file paths, model names) that appear in this transcript.
   - **Summary** — 3–6 sentences capturing what the speaker is actually working on / asking about. Focus on intent and substance, not transcription.

3. Output ONLY the markdown block below, no preamble or postamble.

# Output format (strict)

```
**Topics** (most-specific-first):
- <topic name> — <1-sentence why>
- ...

**Key vocabulary observed**: <comma-separated list of distinctive terms>

**Summary** (3–6 sentences): <ground-truth gist>

**Should an LLM say "no significant content"?**: yes | no
```

# Transcript

{input}
