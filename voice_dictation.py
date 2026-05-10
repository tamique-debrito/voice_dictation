#!/usr/bin/env python3
"""
Voice Dictation App

Global keyboard shortcuts (configurable via local_config.json):
- Default (Mac):    Ctrl+Cmd+R  (double-tap): Toggle recording
- Default (Mac):    Ctrl+Cmd+X  (double-tap): Discard recording
- Default (Mac):    Ctrl+Cmd+A  (double-tap): Aside recording

Pass --config path/to/config.json to override, or place local_config.json
in the same directory for automatic loading.
"""

import argparse
import json
import os
import sys
import time
from pynput import keyboard
from rich.console import Console
from rich.panel import Panel
from audio_recorder import AudioRecorder
from assemblyai_client import AssemblyAIClient
from clipboard_manager import ClipboardManager
from transcriber import Transcriber


import platform as _platform

if _platform.system() == "Windows":
    _DEFAULT_MODIFIERS = ["cmd", "ctrl"]  # Win+Shift
else:  # macOS
    _DEFAULT_MODIFIERS = ["ctrl", "cmd"]   # Ctrl+Cmd

DEFAULT_CONFIG = {
    "modifiers": _DEFAULT_MODIFIERS,
    "keys": {
        "toggle_recording": "r",
        "discard_recording": "x",
        "toggle_aside": "a",
    },
}

# Maps config modifier names → sets of pynput Key objects
_MODIFIER_MAP = {
    "ctrl":  {keyboard.Key.ctrl, keyboard.Key.ctrl_r, keyboard.Key.ctrl_l},
    "cmd":   {keyboard.Key.cmd, keyboard.Key.cmd_r},
    "alt":   {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_r},
}


def load_config(path: str | None) -> dict:
    if path is None:
        auto = os.path.join(os.path.dirname(__file__), "local_config.json")
        path = auto if os.path.exists(auto) else None
    if path is None:
        return DEFAULT_CONFIG
    with open(path) as f:
        cfg = json.load(f)
    return {**DEFAULT_CONFIG, **cfg, "keys": {**DEFAULT_CONFIG["keys"], **cfg.get("keys", {})}}


def _key_char(key) -> str | None:
    try:
        c = key.char
        if not c:
            return None
        # When Ctrl is held, pynput gives a control character (e.g. \x12 for R).
        # Decode it back to the plain letter so hotkey matching works on Windows.
        if len(c) == 1 and ord(c) < 32:
            return chr(ord(c) + 96)
        return c.lower()
    except AttributeError:
        return None


class VoiceDictation:
    def __init__(self, transcriber: Transcriber = None, verbose: bool = False,
                 save_recordings: bool = False, config: dict = None):
        self.console = Console()
        self.recorder = AudioRecorder()
        self.transcriber = transcriber or AssemblyAIClient()
        self.clipboard = ClipboardManager()
        self.verbose = verbose
        self.save_recordings = save_recordings

        cfg = config or DEFAULT_CONFIG
        self._required_modifiers: list[str] = cfg["modifiers"]
        self._action_keys: dict[str, str] = cfg["keys"]
        self._pressed_modifiers: set[str] = set()

        # Double-press tracking
        self.last_trigger_time = 0
        self.last_discard_time = 0
        self.last_aside_time = 0
        self.DOUBLE_PRESS_WINDOW = 1.0

        self.is_recording = False
        self.aside_active = False
        self.stashed_main_frames = None

    def _modifiers_active(self) -> bool:
        return all(m in self._pressed_modifiers for m in self._required_modifiers)

    def _update_modifier(self, key, pressed: bool):
        for name, keys in _MODIFIER_MAP.items():
            if key in keys:
                if pressed:
                    self._pressed_modifiers.add(name)
                else:
                    self._pressed_modifiers.discard(name)
                return True
        return False

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

    def discard_recording(self):
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

    def on_press(self, key):
        try:
            if self._update_modifier(key, True):
                return
            if not self._modifiers_active():
                return
            char = _key_char(key)
            if char is None:
                return
            now = time.time()
            if char == self._action_keys["toggle_recording"]:
                if now - self.last_trigger_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_trigger_time = 0
                    self.stop_recording() if self.is_recording else self.start_recording()
                else:
                    self.last_trigger_time = now
            elif char == self._action_keys["discard_recording"]:
                if now - self.last_discard_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_discard_time = 0
                    self.discard_recording()
                else:
                    self.last_discard_time = now
            elif char == self._action_keys["toggle_aside"]:
                if now - self.last_aside_time <= self.DOUBLE_PRESS_WINDOW:
                    self.last_aside_time = 0
                    self.toggle_aside()
                else:
                    self.last_aside_time = now
        except Exception:
            pass

    def on_release(self, key):
        try:
            self._update_modifier(key, False)
        except Exception:
            pass

    def show_ready_status(self):
        mods = "+".join(m.capitalize() for m in self._required_modifiers)
        r = self._action_keys["toggle_recording"].upper()
        self.console.print(f"[bold green]✓ Ready[/bold green] - Double-press {mods}+{r} to start/stop recording")

    def run(self):
        mods = "+".join(m.capitalize() for m in self._required_modifiers)
        r = self._action_keys["toggle_recording"].upper()
        x = self._action_keys["discard_recording"].upper()
        a = self._action_keys["toggle_aside"].upper()
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold cyan]Voice Dictation App[/bold cyan]\n\n"
            "Shortcuts:\n"
            f"  [bold]{mods}+{r} (x2)[/bold] - Start/Stop recording & transcribe\n"
            f"  [bold]{mods}+{x} (x2)[/bold] - Discard recording (no transcription)\n"
            f"  [bold]{mods}+{a} (x2)[/bold] - Aside: pause main, record/paste aside, then resume\n"
            "  [bold]Ctrl+C[/bold] - Exit\n",
            title="🎤 Voice Dictation",
            border_style="cyan"
        ))
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
    parser = argparse.ArgumentParser(description="Voice dictation app")
    parser.add_argument("--provider", "-p", choices=["assemblyai", "whisper"],
                        default="assemblyai")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to JSON key binding config (default: auto-loads local_config.json)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        transcriber = build_transcriber(args.provider, args.verbose)
        app = VoiceDictation(transcriber=transcriber, verbose=args.verbose,
                             save_recordings=args.save, config=config)
        app.run()
    except ValueError as e:
        Console().print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        Console().print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
