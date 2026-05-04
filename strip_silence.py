#!/usr/bin/env python3
"""Strip silence from WAV recordings using ffmpeg's silenceremove filter."""

import argparse
import glob
import os
import random
import subprocess
import sys

INPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings_trimmed")

DEFAULT_THRESHOLD = "-35dB"
DEFAULT_MIN_DURATION = "0.5"


def build_filter(threshold, duration):
    return (
        f"silenceremove="
        f"stop_periods=-1:"
        f"stop_duration={duration}:"
        f"stop_threshold={threshold}"
    )


def get_size(path):
    """Get total size of a file or directory in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Strip silence from WAV recordings")
    parser.add_argument("--test", type=int, nargs="?", const=5, default=None,
                        help="Test mode: process N random files (default 5)")
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD,
                        help=f"Silence threshold in dB (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--min-duration", default=DEFAULT_MIN_DURATION,
                        help=f"Min silence duration in seconds to strip (default: {DEFAULT_MIN_DURATION}s)")
    args = parser.parse_args()

    af_filter = build_filter(args.threshold, args.min_duration)
    print(f"Settings: threshold={args.threshold}, min_duration={args.min_duration}s")

    wav_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {INPUT_DIR}")
        sys.exit(1)

    if args.test:
        n = min(args.test, len(wav_files))
        wav_files = random.sample(wav_files, n)
        print(f"TEST MODE: processing {n} random files")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(wav_files)
    print(f"Found {total} WAV files to process")

    errors = []
    for i, inpath in enumerate(wav_files, 1):
        filename = os.path.basename(inpath)
        outpath = os.path.join(OUTPUT_DIR, filename)
        print(f"[{i}/{total}] {filename}", end=" ... ", flush=True)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", inpath,
                "-af", af_filter,
                outpath,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("ERROR")
            errors.append((filename, result.stderr))
        else:
            orig = os.path.getsize(inpath)
            new = os.path.getsize(outpath)
            pct = (1 - new / orig) * 100 if orig > 0 else 0
            print(f"done ({human_size(orig)} -> {human_size(new)}, {pct:.0f}% reduced)")

    # Summary
    print("\n--- Summary ---")
    input_size = get_size(INPUT_DIR)
    output_size = get_size(OUTPUT_DIR)
    saved = input_size - output_size
    print(f"Input:  {human_size(input_size)}")
    print(f"Output: {human_size(output_size)}")
    print(f"Saved:  {human_size(saved)} ({(saved / input_size * 100) if input_size else 0:.1f}%)")

    if errors:
        print(f"\n{len(errors)} file(s) had errors:")
        for fname, err in errors:
            print(f"  {fname}: {err[:200]}")


if __name__ == "__main__":
    main()
