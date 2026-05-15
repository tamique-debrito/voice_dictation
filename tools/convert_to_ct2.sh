#!/usr/bin/env bash
# Convert a (LoRA-merged) HuggingFace Whisper checkpoint into a CTranslate2
# directory that `faster-whisper` — and therefore `streaming_transcriber.py` —
# can load.
#
# Usage:
#   bash voice_dictation/tools/convert_to_ct2.sh <hf_dir> <ct2_out_dir> [--quantization int8]
#
# Examples:
#   # After finetune_whisper.py finishes:
#   bash voice_dictation/tools/convert_to_ct2.sh \
#       voice_dictation/models/finetune_20260514/merged_hf \
#       voice_dictation/models/finetune_20260514/ct2
#
#   # Round-trip the stock model (no fine-tune) to validate the toolchain:
#   bash voice_dictation/tools/convert_to_ct2.sh openai/whisper-small.en /tmp/whisper_ct2
#
# To wire the result into live dictation, point `local_config.json` at the
# CT2 directory:
#
#   {
#     "persistent": {
#       "fast": { "fw": { "model": "/abs/path/to/ct2" } }
#     }
#   }
#
# `faster-whisper`'s `WhisperModel(model_size_or_path=...)` treats any
# absolute directory path as a CT2 model — no code changes needed.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <hf_dir> <ct2_out_dir> [extra ct2-transformers-converter args]" >&2
  exit 2
fi

HF_DIR="$1"
OUT_DIR="$2"
shift 2

QUANT="${QUANT:-int8}"

if ! command -v ct2-transformers-converter >/dev/null 2>&1; then
  echo "ct2-transformers-converter not found on PATH." >&2
  echo "Install with: pip install -r voice_dictation/tools/requirements-training.txt" >&2
  exit 1
fi

# --copy_files: tokenizer and feature-extractor configs travel with the
# CT2 weights so faster-whisper has everything it needs from one path.
# Whisper variants split this across tokenizer.json + preprocessor_config.json.
ct2-transformers-converter \
  --model "$HF_DIR" \
  --output_dir "$OUT_DIR" \
  --quantization "$QUANT" \
  --copy_files tokenizer.json preprocessor_config.json \
  "$@"

echo
echo "wrote: $OUT_DIR"
echo "next: set persistent.fast.fw.model = $(cd "$OUT_DIR" && pwd) in local_config.json"
