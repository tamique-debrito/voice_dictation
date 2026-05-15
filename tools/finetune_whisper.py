"""LoRA fine-tune Whisper on a dataset built by ``build_dataset.py``.

Designed for local training on Apple Silicon (MPS). Default base model:
``openai/whisper-tiny.en`` (matches ``local_config.json`` → ``fast.fw.model``).
Override with ``--base-model openai/whisper-small.en`` etc.

Usage::

    python -m voice_dictation.tools.finetune_whisper \
        --dataset /path/to/out_dir \
        --output  /path/to/voice_dictation/models/finetune_<date>_hf \
        --epochs  3

Time budget (rough, M-series): ~2 hours of audio, LoRA rank 16, 3
epochs ≈ several hours of wall-clock.

After training, the LoRA adapter is merged back into the base weights
and saved at ``--output``. Phase E (``convert_to_ct2.sh``) then takes
those merged weights and emits a CT2 directory the live dictation app
can use directly via ``streams.fast.model`` in ``local_config.json``.

This script is not import-safe — it depends on torch / transformers /
peft / datasets which are NOT in the runtime requirements. Install
the training deps first:

    pip install -r voice_dictation/tools/requirements-training.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Ensure MPS-fallback for ops not yet implemented on Metal.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


@dataclass
class Args:
    dataset: Path
    output: Path
    base_model: str = "openai/whisper-tiny.en"
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_steps: Optional[int] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 20
    seed: int = 0
    no_eval: bool = False


def _parse_args(argv: Optional[list[str]] = None) -> Args:
    p = argparse.ArgumentParser(description="LoRA fine-tune Whisper on MPS.")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--base-model", default="openai/whisper-tiny.en")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-eval", action="store_true",
                   help="skip validation (use when val.jsonl is empty)")
    a = p.parse_args(argv)
    return Args(
        dataset=a.dataset, output=a.output, base_model=a.base_model,
        epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
        learning_rate=a.learning_rate, warmup_steps=a.warmup_steps,
        max_steps=a.max_steps, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout, save_steps=a.save_steps,
        eval_steps=a.eval_steps, logging_steps=a.logging_steps,
        seed=a.seed, no_eval=a.no_eval,
    )


# ---------------------------------------------------------------------------


def _load_manifest_dataset(path: Path) -> "Any":
    """Read a build_dataset.py manifest.jsonl into an HF Dataset.

    We keep the audio column as a plain absolute path string and decode
    it ourselves with soundfile in ``_prepare_dataset``. Avoids the
    ``datasets.Audio`` feature, which in recent versions wants
    ``torchcodec`` — extra weight we don't need for fixed 16kHz mono WAVs.
    """
    from datasets import Dataset

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "audio_path": str((path.parent / r["audio_rel"]).resolve()),
                "text": r["text"],
            })
    return Dataset.from_list(rows)


def _build_collator(feature_extractor, tokenizer):
    """Standard Whisper seq2seq collator: pad input features + labels
    separately; replace pad token ids in labels with -100."""
    import torch

    class _Collator:
        def __call__(self, features):
            input_features = [
                {"input_features": f["input_features"]} for f in features
            ]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = feature_extractor.pad(
                input_features, return_tensors="pt",
            )
            labels_batch = tokenizer.pad(
                label_features, return_tensors="pt",
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100,
            )
            # If the first token is a bos, drop it (whisper handles).
            if (labels[:, 0] == tokenizer.bos_token_id).all().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    return _Collator()


def _prepare_dataset(ds, feature_extractor, tokenizer):
    import soundfile as sf
    import numpy as np

    def _prep(batch):
        wav, sr = sf.read(batch["audio_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            # The build_dataset pipeline emits 16kHz WAVs; resample
            # defensively here in case someone hands us something else.
            import librosa  # lazy; only triggered if rate mismatched
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        batch["input_features"] = feature_extractor(
            wav, sampling_rate=16000,
        ).input_features[0]
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch
    return ds.map(_prep, remove_columns=ds.column_names, num_proc=1)


# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = _parse_args(argv)

    if not (args.dataset / "manifest.jsonl").exists():
        print(f"error: {args.dataset}/manifest.jsonl not found", file=sys.stderr)
        return 2

    # Lazy imports so --help works without the heavy deps installed.
    import time
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
        WhisperForConditionalGeneration,
    )
    from peft import LoraConfig, get_peft_model

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("device=%s base_model=%s dataset=%s",
                device, args.base_model, args.dataset)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.base_model, language="english", task="transcribe",
    )
    processor = WhisperProcessor.from_pretrained(args.base_model)

    train_path = args.dataset / "train.jsonl"
    val_path = args.dataset / "val.jsonl"
    if train_path.stat().st_size == 0:
        logger.warning("train.jsonl empty; using manifest.jsonl as train set")
        train_path = args.dataset / "manifest.jsonl"
    train_ds = _load_manifest_dataset(train_path)
    val_ds = None
    if not args.no_eval and val_path.exists() and val_path.stat().st_size > 0:
        val_ds = _load_manifest_dataset(val_path)
    logger.info("train clips=%d  val clips=%s",
                len(train_ds), len(val_ds) if val_ds is not None else "-")

    train_ds = _prepare_dataset(train_ds, feature_extractor, tokenizer)
    if val_ds is not None:
        val_ds = _prepare_dataset(val_ds, feature_extractor, tokenizer)

    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # Note: we deliberately do NOT set task_type=TaskType.SEQ_2_SEQ_LM
    # here. PEFT's PeftModelForSeq2SeqLM.forward is hardcoded to T5-style
    # signatures (input_ids), which collides with Whisper's
    # input_features-based forward. Omitting task_type uses the base
    # PeftModel which passes kwargs through unchanged.
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    collator = _build_collator(feature_extractor, tokenizer)
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )

    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )
    total_steps = (
        args.max_steps if args.max_steps is not None
        else max(1, (len(train_ds) // args.batch_size) * args.epochs
                    // args.grad_accum)
    )
    logger.info(
        "training: total_steps=%d batch_size=%d grad_accum=%d lr=%g",
        total_steps, args.batch_size, args.grad_accum, args.learning_rate,
    )

    step = 0
    accum = 0
    optim.zero_grad()
    loss_window: list[float] = []
    train_start = time.monotonic()
    while step < total_steps:
        for batch in loader:
            if step >= total_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            # Call decoder_input_ids=None to let the model shift labels.
            outputs = model(
                input_features=batch["input_features"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            loss_window.append(outputs.loss.item())
            accum += 1
            if accum >= args.grad_accum:
                optim.step()
                optim.zero_grad()
                step += 1
                accum = 0
                if step % args.logging_steps == 0 or step == 1:
                    avg = sum(loss_window) / len(loss_window)
                    elapsed = time.monotonic() - train_start
                    logger.info(
                        "step %d/%d loss=%.4f elapsed=%.1fs (%.2fs/step)",
                        step, total_steps, avg, elapsed, elapsed / max(step, 1),
                    )
                    loss_window.clear()

    train_elapsed = time.monotonic() - train_start
    logger.info("training done in %.1fs (%d steps)", train_elapsed, step)

    # Merge LoRA into the base weights and save the HF-format checkpoint.
    logger.info("merging LoRA adapter into base weights …")
    model.eval()
    merged = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    logger.info("saved merged HF model to %s", args.output)
    logger.info("next: voice_dictation/tools/convert_to_ct2.sh %s",
                args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
