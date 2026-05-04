"""Clipboard and paste automation functionality."""

import pyperclip
import pyautogui
import time


class ClipboardManager:
    """Manages clipboard operations and paste automation."""

    @staticmethod
    def copy_and_paste(text: str):
        """
        Copy text to clipboard and simulate Cmd+V to paste.

        Args:
            text: Text to copy and paste
        """
        # Copy to clipboard
        pyperclip.copy(text)

        # Delay to ensure clipboard is updated
        time.sleep(0.03)

        try:
            # # Simulate right-click at current mouse position
            # pyautogui.rightClick()
            #
            # # Wait for context menu to appear
            # time.sleep(0.03)
            #
            # # Press 'p' to select Paste option
            # pyautogui.press("p")
            #
            # # Small delay before confirming
            # time.sleep(0.03)
            #
            # # Press Enter to confirm paste
            # pyautogui.press("enter")

            # Paste using Cmd+V
            pyautogui.hotkey("command", "v")

        except Exception as e:
            print(f"Warning: Could not simulate paste: {e}")
            print("Text is copied to clipboard - you can paste manually with Cmd+V")
