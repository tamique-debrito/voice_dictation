#!/usr/bin/env python3
"""
Voice Dictation App

Global keyboard shortcuts (configurable via local_config.json):
- Default (Mac):    Ctrl+Cmd+R  (double-tap): Toggle recording
- Default (Mac):    Ctrl+Cmd+X  (double-tap): Discard recording (or aside)
- Default (Mac):    Ctrl+Cmd+A  (double-tap): Aside recording

Pass --config path/to/config.json to override, or place local_config.json
in the same directory for automatic loading.

Pass --no-modifiers (or set ``"no_modifiers": true`` in the config) to use
bare double-taps of R/A/X with no modifier keys held. Useful when you don't
want to hold Ctrl+Cmd, but means R/A/X cannot be focused into a text input.
"""

import argparse
import time

from pynput import keyboard
from rich.console import Console
from rich.panel import Panel

from audio_recorder import AudioRecorder
from assemblyai_client import AssemblyAIClient
from clipboard_manager import ClipboardManager
from transcriber import Transcriber
from hotkeys import (
    DEFAULT_CONFIG,
    BareDoubleTap,
    ModifierTracker,
    key_char,
    load_config,
)


class VoiceDictation:
    def __init__(self, transcriber: Transcriber = None, verbose: bool = False,
                 save_recordings: bool = False, config: dict = None,
                 no_modifiers: bool = False):
        self.console = Console()
        self.recorder = AudioRecorder()
        self.transcriber = transcriber or AssemblyAIClient()
        self.clipboard = ClipboardManager()
        self.verbose = verbose
        self.save_recordings = save_recordings

        cfg = config or DEFAULT_CONFIG
        self._action_keys: dict[str, str] = cfg["keys"]
        self._required_modifiers: list[str] = cfg["modifiers"]
        self._no_modifiers = bool(no_modifiers or cfg.get("no_modifiers"))
        self._mods = ModifierTracker(self._required_modifiers)

        # Double-press tracking (modifier mode)
        self.last_trigger_time = 0.0
        self.last_discard_time = 0.0
        self.last_aside_time = 0.0
        self.DOUBLE_PRESS_WINDOW = 1.0

        # Bare double-tap (no-modifiers mode) — built lazily once we know keys
        self._bare = None
        if self._no_modifiers:
            tracked = {
                self._action_keys["toggle_recording"],
                self._action_keys["discard_recording"],
                self._action_keys["toggle_aside"],
            }
            self._bare = BareDoubleTap(
                window=self.DOUBLE_PRESS_WINDOW,
                keys=tracked,
                on_double_tap=self._handle_action,
            )

        self.is_recording = False
        self.aside_active = False
        self.stashed_main_frames = None

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.recorder.start_recording()
        self.console.print("[bold red]🔴 Recording...[/bold red]")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.aside_active = False
        self.stashed_main_frames = None
        self.console.print("[bold yellow]⏳ Transcribing...[/bold yellow]")
        audio_file = self.recorder.stop_recording()
        if not audio_file:
            self.console.print("[bold red]❌ No audio recorded[/bold red]")
            self.show_ready_status()
            return
        try:
            if self.save_recordings:
                saved_path = self.recorder.save_recording()
                if saved_path:
                    self.console.print(f"[dim]💾 Saved: {saved_path}[/dim]")
            text = self.transcriber.transcribe_file(audio_file, verbose=self.verbose)
            if text:
                self.clipboard.copy_and_paste(text)
                preview = text[:100] + "..." if len(text) > 100 else text
                self.console.print(f"[bold green]✅ Copied: {preview}[/bold green]")
            else:
                self.console.print("[bold red]❌ Transcription failed[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        finally:
            self.recorder.cleanup()
            self.show_ready_status()

    def toggle_aside(self):
        if not self.is_recording:
            self.console.print("[bold yellow]⚠️  No active recording to aside from[/bold yellow]")
            return
        if not self.aside_active:
            self.stashed_main_frames = self.recorder.pause_recording()
            self.is_recording = False
            self.aside_active = True
            self.recorder.start_recording()
            self.is_recording = True
            self.console.print("[bold magenta]↪ Aside recording...[/bold magenta] (main paused)")
            return
        self.console.print("[bold yellow]⏳ Transcribing aside...[/bold yellow]")
        audio_file = self.recorder.stop_recording()
        self.is_recording = False
        self.aside_active = False
        try:
            if not audio_file:
                self.console.print("[bold red]❌ No aside audio recorded[/bold red]")
            else:
                text = self.transcriber.transcribe_file(audio_file, verbose=self.verbose)
                if text:
                    self.clipboard.copy_and_paste(text)
                    preview = text[:100] + "..." if len(text) > 100 else text
                    self.console.print(f"[bold green]✅ Aside pasted: {preview}[/bold green]")
                else:
                    self.console.print("[bold red]❌ Aside transcription failed[/bold red]")
        except Exception as e:
            self.console.print(f"[bold red]❌ Aside error: {str(e)}[/bold red]")
        finally:
            self.recorder.cleanup()
        resume_frames = self.stashed_main_frames or []
        self.stashed_main_frames = None
        self.recorder.start_recording(initial_frames=resume_frames)
        self.is_recording = True
        self.console.print("[bold red]🔴 Resumed main recording...[/bold red]")

    def cancel_aside(self):
        """Discard the aside-only audio and resume the parked main recording."""
        if not self.aside_active:
            return
        self.recorder.stop_recording()
        self.recorder.cleanup()
        self.aside_active = False
        self.is_recording = False
        resume_frames = self.stashed_main_frames or []
        self.stashed_main_frames = None
        self.recorder.start_recording(initial_frames=resume_frames)
        self.is_recording = True
        self.console.print("[bold cyan]✗ Aside cancelled[/bold cyan] — main recording resumed")

    def discard_recording(self):
        # If we're currently in an aside, cancel only the aside and resume main.
        if self.aside_active:
            self.cancel_aside()
            return
        if not self.is_recording:
            return
        self.is_recording = False
        self.aside_active = False
        self.stashed_main_frames = None
        self.console.print("[bold yellow]🗑️  Discarding recording...[/bold yellow]")
        self.recorder.stop_recording()
        self.recorder.cleanup()
        self.console.print("[bold cyan]✗ Recording discarded[/bold cyan]")
        self.show_ready_status()

    # ------------------------------------------------------------------
    # Key dispatch
    # ------------------------------------------------------------------

    def _handle_action(self, char: str) -> None:
        """Run the action mapped to a hotkey character (already debounced)."""
        if char == self._action_keys["toggle_recording"]:
            self.stop_recording() if self.is_recording else self.start_recording()
        elif char == self._action_keys["discard_recording"]:
            self.discard_recording()
        elif char == self._action_keys["toggle_aside"]:
            self.toggle_aside()

    def on_press(self, key):
        try:
            if self._no_modifiers:
                self._bare.feed(key_char(key))
                return
            # Modifier-keyed mode: track modifier state and require it.
            if self._mods.update(key, True):
                return
            if not self._mods.all_required_active():
                return
            char = key_char(key)
            if char is None:
                return
            now = time.time()
            if char == self._action_keys["toggle_recording"]:
                if now - self.last_trigger_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_trigger_time = 0.0
                    self.stop_recording() if self.is_recording else self.start_recording()
                else:
                    self.last_trigger_time = now
            elif char == self._action_keys["discard_recording"]:
                if now - self.last_discard_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_discard_time = 0.0
                    self.discard_recording()
                else:
                    self.last_discard_time = now
            elif char == self._action_keys["toggle_aside"]:
                if now - self.last_aside_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_aside_time = 0.0
                    self.toggle_aside()
                else:
                    self.last_aside_time = now
        except Exception:
            pass

    def on_release(self, key):
        try:
            if not self._no_modifiers:
                self._mods.update(key, False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _trigger_label(self) -> str:
        r = self._action_keys["toggle_recording"].upper()
        if self._no_modifiers:
            return r
        mods = "+".join(m.capitalize() for m in self._required_modifiers)
        return f"{mods}+{r}"

    def show_ready_status(self):
        self.console.print(
            f"[bold green]✓ Ready[/bold green] - Double-press {self._trigger_label()} to start/stop recording"
        )

    def run(self):
        r = self._action_keys["toggle_recording"].upper()
        x = self._action_keys["discard_recording"].upper()
        a = self._action_keys["toggle_aside"].upper()
        if self._no_modifiers:
            r_lbl, x_lbl, a_lbl = r, x, a
            mode_note = (
                "[dim]Mode: [bold]no-modifiers[/bold] — "
                "double-tap the bare letter (don't hold Ctrl/Cmd). "
                "Avoid pressing these letters while a text input is focused.[/dim]"
            )
        else:
            mods = "+".join(m.capitalize() for m in self._required_modifiers)
            r_lbl, x_lbl, a_lbl = f"{mods}+{r}", f"{mods}+{x}", f"{mods}+{a}"
            mode_note = ""
        self.console.clear()
        body = (
            "[bold cyan]Voice Dictation App[/bold cyan]\n\n"
            "Shortcuts:\n"
            f"  [bold]{r_lbl} (x2)[/bold] - Start/Stop recording & transcribe\n"
            f"  [bold]{x_lbl} (x2)[/bold] - Cancel: aside if in aside, else discard recording\n"
            f"  [bold]{a_lbl} (x2)[/bold] - Aside: pause main, record/paste aside, then resume\n"
            "  [bold]Ctrl+C[/bold] - Exit\n"
        )
        if mode_note:
            body += "\n" + mode_note
        self.console.print(Panel.fit(body, title="🎤 Voice Dictation", border_style="cyan"))
        self.show_ready_status()
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass
        self.console.print("\n[bold yellow]Shutting down...[/bold yellow]")
        self.recorder.shutdown()


def build_transcriber(provider: str, verbose: bool) -> Transcriber:
    if provider == "assemblyai":
        return AssemblyAIClient()
    if provider == "whisper":
        from whisper_client import WhisperClient
        if verbose:
            print("[DEBUG] Loading local Whisper model...")
        return WhisperClient()
    raise ValueError(f"Unknown transcription provider: {provider}")


def main():
    import sys
    parser = argparse.ArgumentParser(description="Voice dictation app")
    parser.add_argument("--provider", "-p", choices=["assemblyai", "whisper"],
                        default="assemblyai")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to JSON key binding config (default: auto-loads local_config.json)")
    parser.add_argument("--no-modifiers", action="store_true",
                        help="Use bare double-tap of R/A/X (no Ctrl/Cmd held). "
                             "Avoid pressing these letters while a text input is focused.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        transcriber = build_transcriber(args.provider, args.verbose)
        app = VoiceDictation(
            transcriber=transcriber,
            verbose=args.verbose,
            save_recordings=args.save,
            config=config,
            no_modifiers=args.no_modifiers,
        )
        app.run()
    except ValueError as e:
        Console().print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        Console().print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
