"""Clipboard and paste automation functionality."""

import platform
import time

import pyperclip
from pynput import keyboard


if platform.system() == "Windows":
    _PASTE_MODIFIER = keyboard.Key.ctrl
else:  # macOS / Linux
    _PASTE_MODIFIER = keyboard.Key.cmd


class ClipboardManager:
    """Manages clipboard operations and paste automation."""

    def __init__(self):
        self._kb = keyboard.Controller()

    def copy_and_paste(self, text: str):
        """Copy text to clipboard and simulate Cmd+V (or Ctrl+V) to paste."""
        pyperclip.copy(text)
        # Delay to ensure clipboard is updated before paste fires.
        time.sleep(0.03)
        try:
            with self._kb.pressed(_PASTE_MODIFIER):
                self._kb.tap("v")
        except Exception as e:
            print(f"Warning: Could not simulate paste: {e}")
            print("Text is copied to clipboard - you can paste manually with Cmd+V")
