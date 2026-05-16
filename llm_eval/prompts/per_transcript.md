You are analyzing a single voice-dictation transcript from a software engineer. Your job is to identify the topics they're discussing, capture distinctive vocabulary, and produce a concise ground-truth summary. The transcript is the dictation output of someone thinking aloud or asking questions to an assistant about their codebase, so it may be fragmented, contain false starts, and include filler — extract MEANING, not surface form.

# Instructions

1. **Decide first** if the transcript has significant work content. Set `Significant content` accordingly:

   `Significant content: yes` ONLY if the transcript contains ONE OR MORE of:
   - A design discussion (architectural reasoning, trade-off analysis, plan for a refactor).
   - Active debugging or root-cause analysis with details (not just "I noticed X is broken").
   - Specific code-direction with multiple steps or constraints (not a single imperative like "start the server").

   `Significant content: no` for everything else, INCLUDING:
   - Short action requests ("start the V2 server", "run the live-play tool") even if they name specific things.
   - Brief bug observations without analysis ("there's a duplicate paste block").
   - Meta-questions about the dictation tool itself.
   - Test phrases, trivia questions, or recordings of mostly filler ("I don't know", "thank you").

   When `no`: leave `Topics` empty, leave `Key vocabulary observed` empty, write a one-sentence Summary that says what the speaker did (e.g. "asked to start the V2 live-play server") and STOP. Do not list topics from the surface mentions.

2. **Identify distinct work threads.** Sessions sometimes interleave two unrelated workstreams (e.g. a voice-dictation pipeline redesign AND a card-game UI fix in the same recording). If you see this, list each work thread as a separate top-level topic — don't collapse them.

3. **Topics**: aim for 4–8 specific topics for a substantive transcript. Order most-specific-first. Avoid generic words ("software", "code", "system"). Prefer concrete project nouns ("AudioPreprocessor", "paste-window finalization", "kitty private info").

4. **Key vocabulary**: pull distinctive terms VERBATIM from the transcript — function names, file names, model names, technical jargon. Generic words don't count. Pull more rather than fewer if you're unsure: 10–25 terms for a substantive transcript.

# Examples

## Example A — minimal content (action request)

Transcript: `Can you start the V2 live-play server for Chao Pung Yo?`

Correct output:
```
**Significant content**: no

**Topics**: (empty)
**Key vocabulary observed**: (empty)
**Summary**: Speaker asks the assistant to start the V2 live-play server for Chao Pung Yo. No design content.
```

## Example B — minimal content (brief bug note)

Transcript: `The paste block at the end of the transcript looks duplicated. Not sure why. I don't know. I don't know.`

Correct output:
```
**Significant content**: no

**Topics**: (empty)
**Key vocabulary observed**: (empty)
**Summary**: Speaker briefly notes a duplicated paste block at the end of a transcript with no further analysis.
```

# Output format (strict — no preamble, no postamble)

```
**Significant content**: yes | no

**Topics** (most-specific-first):
- <topic name> — <1-sentence why>
- ...

**Key vocabulary observed**: <comma-separated list of distinctive terms>

**Summary** (3–6 sentences for substantive; 1 sentence for minimal): <ground-truth gist>
```

# Transcript

{input}
