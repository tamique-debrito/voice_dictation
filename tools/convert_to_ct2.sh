#!/usr/bin/env bash
# Convert a merged HuggingFace Whisper checkpoint (from finetune_whisper.py)
# into a CTranslate2 directory that the live dictation app can load via
# faster-whisper.
#
# Usage:
#   voice_dictation/tools/convert_to_ct2.sh <hf_model_dir> [out_dir]
#
# Defaults: <hf_model_dir>_ct2 next to the input.
#
# Requires:
#   pip install ctranslate2 transformers
#
# After conversion, point the live app at the new model:
#   local_config.json → fast.fw.model: "<out_dir>"
# (faster-whisper accepts a directory path transparently.)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <hf_model_dir> [out_dir]" >&2
  exit 2
fi

HF_DIR="$1"
OUT_DIR="${2:-${HF_DIR%/}_ct2}"

if [[ ! -d "$HF_DIR" ]]; then
  echo "error: not a directory: $HF_DIR" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT_DIR")"

# int8 quantization is the default for parity with faster-whisper's
# typical setup. Override by setting QUANT=int8_float16 / float16 / none.
QUANT="${QUANT:-int8}"

ct2-transformers-converter \
  --model "$HF_DIR" \
  --output_dir "$OUT_DIR" \
  --quantization "$QUANT" \
  --copy_files tokenizer.json preprocessor_config.json \
  --force

echo "wrote: $OUT_DIR"
echo
echo "next: set in voice_dictation/local_config.json"
echo '  "fast": { "fw": { "model": "'"$OUT_DIR"'" } }'
