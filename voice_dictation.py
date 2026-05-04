#!/usr/bin/env python3
"""
Voice Dictation App with AssemblyAI

Global keyboard shortcuts:
- Ctrl+Cmd+R (double-tap within 1 second): Toggle recording (start/stop and transcribe)
- Ctrl+Cmd+X (double-tap within 1 second): Discard current recording without transcribing

Transcribed text is automatically copied to clipboard and pasted.
"""

import sys
import time
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from audio_recorder import AudioRecorder
from assemblyai_client import AssemblyAIClient
from clipboard_manager import ClipboardManager
from transcriber import Transcriber


class VoiceDictation:
    """Main application class for voice dictation."""

    def __init__(self, transcriber: Transcriber = None, verbose: bool = False):
        self.console = Console()
        self.recorder = AudioRecorder()
        self.transcriber = transcriber or AssemblyAIClient()
        self.clipboard = ClipboardManager()
        self.verbose = verbose

        # Keyboard state tracking
        self.cmd_pressed = False
        self.ctrl_pressed = False

        # Double-press tracking
        self.last_trigger_time = 0
        self.last_discard_time = 0
        self.DOUBLE_PRESS_WINDOW = 1.0  # seconds

        # Application state
        self.is_recording = False

    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return

        self.is_recording = True
        self.recorder.start_recording()
        self.console.print("[bold red]🔴 Recording...[/bold red]")

    def stop_recording(self):
        """Stop recording and trigger transcription."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.console.print("[bold yellow]⏳ Transcribing...[/bold yellow]")

        # Stop recording and get audio file
        audio_file = self.recorder.stop_recording()

        if not audio_file:
            self.console.print("[bold red]❌ No audio recorded[/bold red]")
            self.show_ready_status()
            return

        try:
            # Save audio file before transcription
            saved_path = self.recorder.save_recording()
            if saved_path:
                self.console.print(f"[dim]💾 Saved: {saved_path}[/dim]")

            # Transcribe audio
            text = self.transcriber.transcribe_file(audio_file, verbose=self.verbose)

            if text:
                # Copy to clipboard and paste
                self.clipboard.copy_and_paste(text)

                # Show preview (truncate if too long)
                preview = text[:100] + "..." if len(text) > 100 else text
                self.console.print(f"[bold green]✅ Copied: {preview}[/bold green]")
            else:
                self.console.print("[bold red]❌ Transcription failed[/bold red]")

        except Exception as e:
            self.console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")

        finally:
            # Clean up temporary file
            self.recorder.cleanup()
            self.show_ready_status()

    def discard_recording(self):
        """Stop recording and discard without transcribing."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.console.print("[bold yellow]🗑️  Discarding recording...[/bold yellow]")

        # Stop recording and discard the audio file
        self.recorder.stop_recording()
        self.recorder.cleanup()

        self.console.print("[bold cyan]✗ Recording discarded[/bold cyan]")
        self.show_ready_status()

    @staticmethod
    def _is_r_key(key):
        """Check if the pressed key is 'r' regardless of held modifiers."""
        # When modifiers are held, key.char can be mangled (e.g. '\x12' with Ctrl).
        # The vk (virtual key code) is reliable — 15 is 'r' on macOS.
        if hasattr(key, 'vk') and key.vk == 15:
            return True
        if hasattr(key, 'char') and key.char in ('r', 'R'):
            return True
        return False

    @staticmethod
    def _is_x_key(key):
        """Check if the pressed key is 'x' regardless of held modifiers."""
        # vk 7 is 'x' on macOS
        if hasattr(key, 'vk') and key.vk == 7:
            return True
        if hasattr(key, 'char') and key.char in ('x', 'X'):
            return True
        return False

    def on_press(self, key):
        """Handle key press events."""
        try:
            # Track modifier keys
            if key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.cmd_pressed = True
            elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = True
            elif self._is_r_key(key):
                # Detect Ctrl+Cmd+R (double-tap to toggle recording)
                if self.cmd_pressed and self.ctrl_pressed:
                    now = time.time()
                    if now - self.last_trigger_time <= self.DOUBLE_PRESS_WINDOW:
                        # Second press within window — toggle recording
                        self.last_trigger_time = 0  # reset
                        if self.is_recording:
                            self.stop_recording()
                        else:
                            self.start_recording()
                    else:
                        # First press — record the time
                        self.last_trigger_time = now
            elif self._is_x_key(key):
                # Detect Ctrl+Cmd+X (double-tap to discard recording)
                if self.cmd_pressed and self.ctrl_pressed:
                    now = time.time()
                    if now - self.last_discard_time <= self.DOUBLE_PRESS_WINDOW:
                        # Second press within window — discard recording
                        self.last_discard_time = 0  # reset
                        self.discard_recording()
                    else:
                        # First press — record the time
                        self.last_discard_time = now

        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            # Reset modifier state
            if key == keyboard.Key.cmd or key == keyboard.Key.cmd_r:
                self.cmd_pressed = False
            elif key == keyboard.Key.ctrl or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = False

        except AttributeError:
            pass

    def show_ready_status(self):
        """Display ready status in terminal."""
        self.console.print("[bold green]✓ Ready[/bold green] - Double-press Ctrl+Cmd+R to start/stop recording")

    def run(self):
        """Start the voice dictation app."""
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold cyan]Voice Dictation App[/bold cyan]\n\n"
            "Shortcuts:\n"
            "  [bold]Ctrl+Cmd+R (x2)[/bold] - Start/Stop recording & transcribe\n"
            "  [bold]Ctrl+Cmd+X (x2)[/bold] - Discard recording (no transcription)\n"
            "  [bold]Ctrl+C[/bold] - Exit\n",
            title="🎤 Voice Dictation",
            border_style="cyan"
        ))

        self.show_ready_status()

        # Start keyboard listener
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

        self.console.print("\n[bold yellow]Shutting down...[/bold yellow]")
        self.recorder.shutdown()


def main():
    """Main entry point."""
    try:
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        app = VoiceDictation(verbose=verbose)
        app.run()
    except ValueError as e:
        console = Console()
        console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
