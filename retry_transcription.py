#!/usr/bin/env python3
"""CLI tool to retry transcription of saved audio recordings."""

import sys
import subprocess
import tempfile
import argparse
from pathlib import Path
import config  # Load .env file
from assemblyai_client import AssemblyAIClient
from clipboard_manager import ClipboardManager
from rich.console import Console


def strip_silence(input_path: Path, console: Console) -> Path:
    """Strip silence from audio using ffmpeg, return path to processed file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    output_path = Path(tmp.name)

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-35dB",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:300]}")

    orig_mb = input_path.stat().st_size / 1024 / 1024
    new_mb = output_path.stat().st_size / 1024 / 1024
    console.print(f"[dim]Stripped silence: {orig_mb:.1f}MB → {new_mb:.1f}MB[/dim]")
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Retry transcription of a saved audio file"
    )
    parser.add_argument(
        "filename",
        help="Audio file to transcribe (e.g., recording_20260504_125453.wav)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show debug information"
    )
    parser.add_argument(
        "--no-strip",
        action="store_true",
        help="Skip silence stripping before upload"
    )

    args = parser.parse_args()
    console = Console()

    # Find the file in recordings directory
    recordings_dir = Path(__file__).parent / "recordings"
    file_path = recordings_dir / args.filename

    if not file_path.exists():
        console.print(f"[red]❌ File not found: {file_path}[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Transcribing: {args.filename}[/bold cyan]")

    upload_path = file_path
    tmp_path = None
    if not args.no_strip:
        try:
            tmp_path = strip_silence(file_path, console)
            upload_path = tmp_path
        except Exception as e:
            console.print(f"[yellow]⚠ Silence stripping failed ({e}), using original[/yellow]")

    console.print("[bold yellow]⏳ Transcribing...[/bold yellow]")

    try:
        transcriber = AssemblyAIClient()
        text = transcriber.transcribe_file(str(upload_path), verbose=args.verbose)

        if text:
            console.print(f"\n[bold green]✅ Success![/bold green]\n")
            console.print(f"[bold]Transcribed text:[/bold]\n{text}\n")

            # Copy to clipboard and paste
            clipboard = ClipboardManager()
            clipboard.copy_and_paste(text)
            console.print("[green]📋 Copied to clipboard and pasted[/green]")
            return 0
        else:
            console.print("[bold red]❌ Transcription returned no text[/bold red]")
            return 1

    except Exception as e:
        console.print(f"[bold red]❌ Transcription failed: {str(e)}[/bold red]")
        if args.verbose:
            import traceback
            console.print("\n[yellow]Stack trace:[/yellow]")
            console.print(traceback.format_exc())
        return 1
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
