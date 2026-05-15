#!/usr/bin/env python3
"""LoRA fine-tune Whisper on collected dictation data (Apple Silicon / MPS).

Pipeline:
    1. Load the dataset produced by ``tools/build_dataset.py``
       (``train.jsonl`` + ``validation.jsonl`` of ``{audio, text}`` rows).
    2. Resample on the fly to 16 kHz via ``datasets.Audio``.
    3. Wrap the base Whisper model (default ``openai/whisper-small.en``)
       with a LoRA adapter targeting q/k/v/out_proj.
    4. Train with HuggingFace ``Seq2SeqTrainer``, fp32 (MPS doesn't do
       fp16/bf16 reliably), gradient checkpointing, generation-based
       eval with WER as the model-selection metric.
    5. Merge the LoRA adapter back into the base weights and save the
       result to ``<output_dir>/merged_hf`` — this directory is what
       ``convert_to_ct2.sh`` (Phase D) feeds into CT2 conversion.

Wall-clock expectation:
    ~2 hours of training audio at r=16 / 3 epochs / per-device batch 4
    / grad-accum 4 takes a few hours on an M-series. ``--smoke`` caps
    training at 20 steps so you can validate the pipeline in minutes.

Quick start:
    pip install -r voice_dictation/tools/requirements-training.txt
    python tools/build_dataset.py transcripts/* --out datasets/v1
    python tools/finetune_whisper.py \\
        --train datasets/v1/train.jsonl \\
        --val   datasets/v1/validation.jsonl \\
        --output-dir voice_dictation/models/finetune_20260514

After training completes, run ``convert_to_ct2.sh`` against
``voice_dictation/models/finetune_20260514/merged_hf`` to produce the
CT2 directory that ``streaming_transcriber.py`` consumes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

# Required for any Whisper op that lacks an MPS kernel — falls back to CPU
# transparently. Must be set BEFORE torch is imported. Keep this above
# all heavy imports.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    # Heavy imports gated behind argparse so ``--help`` is instant and
    # doesn't require the training stack to be installed.
    import torch  # noqa: F401  (sanity: surfaces "torch not installed" early)
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    import evaluate

    device = _pick_device()
    print(f"[finetune] device={device}  base={args.base_model}  output={args.output_dir}")

    # -- 1. Dataset ----------------------------------------------------
    data_files = {"train": args.train}
    if args.val and os.path.exists(args.val):
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    print(f"[finetune] train={len(ds['train'])}  val={len(ds.get('validation', []))}")

    processor = WhisperProcessor.from_pretrained(args.base_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    # .en models don't take language/task tokens. Clear any defaults the
    # processor may have baked in so generation doesn't prepend tokens
    # that aren't in the .en tokenizer vocab.
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    # ``use_cache`` must be off when gradient_checkpointing is on, or
    # the model errors with "use_cache=True is incompatible with grad
    # checkpointing".
    base_model.config.use_cache = False

    def prepare(batch: dict) -> dict:
        audio = batch["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]
        # If the dataset row came from `build_dataset.py --per-segment`,
        # the row carries an audio sub-range to slice out of the full
        # WAV. Without this slicing, a 20-minute chunk WAV would be
        # silently truncated to its first 30s by the feature extractor
        # (Whisper's encoder window), which is degenerate training:
        # the model sees only the start of the audio but is told to
        # reproduce the entire chunk transcript.
        start_s = batch.get("audio_start_s")
        end_s = batch.get("audio_end_s")
        if start_s is not None and end_s is not None:
            s = max(0, int(float(start_s) * sr))
            e = min(len(arr), int(float(end_s) * sr))
            if e > s:
                arr = arr[s:e]
        batch["input_features"] = processor.feature_extractor(
            arr, sampling_rate=sr,
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1)

    # -- 2. LoRA -------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # -- 3. Collator + metric -----------------------------------------
    collator = _DataCollator(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # Replace -100 with the pad id so decode doesn't crash.
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100.0 * wer_metric.compute(predictions=pred_str, references=label_str)}

    # -- 4. Train -----------------------------------------------------
    has_val = "validation" in ds and len(ds["validation"]) > 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
        gradient_checkpointing=True,
        # MPS doesn't reliably support fp16/bf16 mixed precision in
        # transformers' Trainer path; force fp32 to avoid silent NaNs.
        fp16=False,
        bf16=False,
        eval_strategy=("steps" if has_val else "no"),
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=has_val,
        generation_max_length=225,
        save_total_limit=3,
        load_best_model_at_end=has_val,
        metric_for_best_model=("wer" if has_val else None),
        greater_is_better=(False if has_val else None),
        report_to=[],
        # Keep MPS deterministic-ish; CUDA-only options skipped.
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_val else None,
        data_collator=collator,
        compute_metrics=compute_metrics if has_val else None,
        tokenizer=processor.feature_extractor,  # for input padding
    )

    # Persist the processor next to the adapter checkpoints so a
    # mid-training kill still leaves a usable checkpoint.
    processor.save_pretrained(args.output_dir)
    trainer.train()

    # -- 5. Merge + save ---------------------------------------------
    print("[finetune] merging LoRA adapter into base weights")
    peft_model: PeftModel = trainer.model
    merged = peft_model.merge_and_unload()
    merged_dir = os.path.join(args.output_dir, "merged_hf")
    merged.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"[finetune] merged model written to: {merged_dir}")
    print(f"[finetune] next: bash voice_dictation/tools/convert_to_ct2.sh {merged_dir} <ct2_out>")
    return 0


# ---------------------------------------------------------------------------
# Data collator — standard Whisper-finetune pattern
# ---------------------------------------------------------------------------

@dataclass
class _DataCollator:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        # Mel features and labels need separate padding because their
        # tensors have different shapes (features are pre-padded to 30s
        # log-mel; labels are variable-length token sequences).
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # Mask pad tokens in the loss so the model isn't penalized for
        # predicting padding. -100 is HF's "ignore" sentinel.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer prepends a BOS that the Trainer's shift_right
        # adds back at training time — strip it here if present so we
        # don't end up with two.
        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and (labels[:, 0] == bos).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train", required=True, help="Path to train.jsonl from build_dataset.py.")
    p.add_argument("--val", default=None, help="Path to validation.jsonl (optional).")
    p.add_argument("--output-dir", required=True, help="Where to write checkpoints and the merged model.")
    p.add_argument("--base-model", default="openai/whisper-small.en")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4).")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument(
        "--max-steps", type=int, default=0,
        help="Cap training at N optimizer steps. 0 = unlimited (use --num-epochs instead).",
    )
    p.add_argument("--num-epochs", type=int, default=3, help="Used only when --max-steps=0.")
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument(
        "--max-train-samples", type=int, default=0,
        help="Subset the training split to this many rows (smoke testing).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Shortcut: --max-steps=20 --logging-steps=2 --eval-steps=10 --save-steps=10.",
    )
    args = p.parse_args(argv)
    if args.smoke:
        args.max_steps = max(args.max_steps, 20)
        args.logging_steps = min(args.logging_steps, 2)
        args.eval_steps = min(args.eval_steps, 10)
        args.save_steps = min(args.save_steps, 10)
    return args


def _pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    sys.exit(main())
