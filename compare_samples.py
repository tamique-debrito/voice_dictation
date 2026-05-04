#!/usr/bin/env python3
"""Compare original vs trimmed WAV files by playing them back-to-back.

Usage:
    python compare_samples.py              # Pick 3 random samples
    python compare_samples.py 5            # Pick 5 random samples
    python compare_samples.py file1.wav file2.wav  # Compare specific files
"""

import glob
import os
import random
import subprocess
import sys

INPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "recordings_trimmed")


def get_duration(path):
    """Get duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def play_file(path, label):
    """Play a WAV file using afplay (macOS)."""
    print(f"  Playing {label}...", flush=True)
    subprocess.run(["afplay", path])


def compare(filename):
    orig = os.path.join(INPUT_DIR, filename)
    trimmed = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(orig):
        print(f"  Original not found: {orig}")
        return
    if not os.path.exists(trimmed):
        print(f"  Trimmed not found: {trimmed}")
        return

    orig_dur = get_duration(orig)
    trim_dur = get_duration(trimmed)
    orig_size = os.path.getsize(orig)
    trim_size = os.path.getsize(trimmed)

    print(f"\n{'=' * 60}")
    print(f"File: {filename}")
    print(f"  Duration: {orig_dur:.1f}s -> {trim_dur:.1f}s (removed {orig_dur - trim_dur:.1f}s)")
    print(f"  Size:     {human_size(orig_size)} -> {human_size(trim_size)}")
    print()

    play_file(orig, "ORIGINAL")
    input("  Press Enter to hear the trimmed version...")
    play_file(trimmed, "TRIMMED")
    input("  Press Enter to continue to next sample (or Ctrl+C to stop)...")


def main():
    args = sys.argv[1:]

    if args and all(a.endswith(".wav") for a in args):
        filenames = args
    else:
        n = int(args[0]) if args else 3
        all_trimmed = [os.path.basename(f) for f in glob.glob(os.path.join(OUTPUT_DIR, "*.wav"))]
        if not all_trimmed:
            print(f"No trimmed files found in {OUTPUT_DIR}. Run strip_silence.py first.")
            sys.exit(1)
        filenames = random.sample(all_trimmed, min(n, len(all_trimmed)))

    print(f"Comparing {len(filenames)} file(s): original vs trimmed")
    print("You'll hear the original first, then the trimmed version.")

    for fname in filenames:
        try:
            compare(fname)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
