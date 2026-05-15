"""Run a prompt template against an Ollama-served local LLM.

Used for prompt-engineering iteration on the topic-extraction benchmark
(see ``voice_dictation/.claude/skills/topic-extract-eval/SKILL.md``).

Two phases of the workflow use the same CLI:

  Phase 1 — per-transcript: feed one transcript file to a prompt that
    asks the model to extract topics / vocabulary / summary. Compare
    against ``ground_truth.md`` per-transcript annotations.

  Phase 2 — aggregation: feed the Phase-1 outputs back in (concatenated)
    to a prompt that asks the model to merge per-transcript summaries
    into a corpus-level annotation. Compare against ``ground_truth.md``
    Section 1.

Prompts are markdown files containing a ``{input}`` placeholder that
gets replaced with the concatenated contents of the ``--input`` files
(separated by a blank line + a banner header).

Usage::

    # Phase 1:
    python -m voice_dictation.tools.llm_eval \\
        --prompt   voice_dictation/llm_eval/prompts/per_transcript_v1.md \\
        --input    scratch/transcripts_text/regen_X.txt \\
        --model    llama3.1:8b-instruct-q4_K_M \\
        --num-ctx  8192 \\
        --output   voice_dictation/llm_eval/runs/<run_name>.md

    # Phase 2 (aggregate over multiple per-transcript outputs):
    python -m voice_dictation.tools.llm_eval \\
        --prompt   voice_dictation/llm_eval/prompts/aggregate_v1.md \\
        --input    voice_dictation/llm_eval/runs/per_transcript_v1__*.md \\
        --model    llama3.1:8b-instruct-q4_K_M \\
        --num-ctx  16384 \\
        --output   voice_dictation/llm_eval/runs/aggregate_v1.md

Ollama must be running locally (``ollama serve``). Talks to its
OpenAI-compatible endpoint at ``http://127.0.0.1:11434/v1`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def _concat_inputs(paths: list[Path]) -> str:
    """Concatenate input files with a clear banner between each. The
    banner makes it obvious to the model where one source ends and the
    next begins (useful for aggregation prompts)."""
    if len(paths) == 1:
        return paths[0].read_text(encoding="utf-8")
    parts: list[str] = []
    for p in paths:
        parts.append(f"\n===== BEGIN: {p.name} =====\n")
        parts.append(p.read_text(encoding="utf-8"))
        parts.append(f"\n===== END: {p.name} =====\n")
    return "".join(parts)


def _render_prompt(prompt_template: str, input_text: str) -> str:
    if "{input}" not in prompt_template:
        raise ValueError(
            "prompt template must contain the literal placeholder {input}"
        )
    return prompt_template.replace("{input}", input_text)


def _call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    *,
    num_ctx: int,
    temperature: float,
    think: Optional[bool],
    timeout_seconds: float,
) -> dict:
    """POST to /api/generate (native Ollama). Returns the raw response dict.

    /api/generate is preferred over /v1/chat/completions here because it
    exposes ``think`` and detailed timing (``total_duration``, etc.) and
    we don't need OpenAI compat for single-turn generation.
    """
    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
        },
    }
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _approx_token_count(s: str) -> int:
    # Rough heuristic — 4 chars/token. Good enough for warnings.
    return max(1, len(s) // 4)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prompt", type=Path, required=True,
                   help="path to a prompt template (markdown with {input})")
    p.add_argument("--input", type=Path, nargs="+", required=True,
                   help="one or more input files (concatenated with banners "
                        "if more than one)")
    p.add_argument("--model", required=True,
                   help="Ollama model tag (e.g. llama3.1:8b-instruct-q4_K_M)")
    p.add_argument("--output", type=Path, required=True,
                   help="output markdown file (will be created)")
    p.add_argument("--num-ctx", type=int, default=8192,
                   help="Ollama num_ctx (default 8192). Set higher for "
                        "aggregation phase or long transcripts.")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--think", dest="think", action="store_true", default=None,
                   help="enable thinking on supported models (qwen3, gpt-oss)")
    p.add_argument("--no-think", dest="think", action="store_false",
                   help="explicitly disable thinking")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args(argv)

    for f in args.input:
        if not f.exists():
            print(f"error: input not found: {f}", file=sys.stderr)
            return 2
    if not args.prompt.exists():
        print(f"error: prompt not found: {args.prompt}", file=sys.stderr)
        return 2

    prompt_template = args.prompt.read_text(encoding="utf-8")
    input_text = _concat_inputs(args.input)
    rendered = _render_prompt(prompt_template, input_text)

    approx_in = _approx_token_count(rendered)
    if approx_in > args.num_ctx * 0.85:
        print(
            f"warning: rendered prompt ~{approx_in} tokens vs "
            f"num_ctx={args.num_ctx} — context may overflow",
            file=sys.stderr,
        )

    print(f"model={args.model}  num_ctx={args.num_ctx}  "
          f"approx_input_tokens={approx_in}  think={args.think}",
          file=sys.stderr)
    t0 = time.monotonic()
    try:
        result = _call_ollama(
            args.ollama_url, args.model, rendered,
            num_ctx=args.num_ctx, temperature=args.temperature,
            think=args.think, timeout_seconds=args.timeout,
        )
    except Exception as e:
        print(f"error: ollama call failed: {e}", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - t0

    completion = result.get("response", "")
    thinking = result.get("thinking", "") or ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Header so the run is self-documenting on disk.
    header_lines = [
        "# llm_eval run",
        f"model: {args.model}",
        f"prompt: {args.prompt}",
        f"input: {', '.join(str(p) for p in args.input)}",
        f"num_ctx: {args.num_ctx}",
        f"temperature: {args.temperature}",
        f"think: {args.think}",
        f"run_at: {datetime.now(tz=timezone.utc).isoformat()}",
        f"wall_seconds: {elapsed:.2f}",
        f"prompt_eval_count: {result.get('prompt_eval_count', '?')}",
        f"eval_count: {result.get('eval_count', '?')}",
        f"total_duration_ns: {result.get('total_duration', '?')}",
        "",
        "---",
        "",
    ]
    body_parts = ["\n".join(header_lines)]
    if thinking.strip():
        body_parts.append("## Thinking (model internal)\n\n```\n")
        body_parts.append(thinking.rstrip() + "\n")
        body_parts.append("```\n\n")
    body_parts.append("## Response\n\n")
    body_parts.append(completion.rstrip() + "\n")
    args.output.write_text("".join(body_parts), encoding="utf-8")

    print(f"wrote {args.output}  "
          f"({elapsed:.1f}s, {result.get('eval_count', '?')} output tokens)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
