"""Run this and press keys to see what pynput reports."""
from pynput import keyboard

def on_press(key):
    print(f"PRESS   key={key!r:30}  char={getattr(key, 'char', None)!r:10}  vk={getattr(key, 'vk', None)}")

def on_release(key):
    print(f"release key={key!r:30}  char={getattr(key, 'char', None)!r:10}  vk={getattr(key, 'vk', None)}")
    if key == keyboard.Key.esc:
        return False  # stop listener

print("Press keys to inspect them. Esc to quit.\n")
with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
    l.join()
