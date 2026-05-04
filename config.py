"""Configuration for voice dictation app."""

import os
from dotenv import load_dotenv

# Load .env file from same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Audio recording configuration
SAMPLE_RATE = 16000  # 16kHz (AssemblyAI recommended)
CHANNELS = 1  # Mono
FORMAT = "int16"  # 16-bit PCM
CHUNK_SIZE = 1024  # Frames per buffer

# Audio file storage
AUDIO_SAVE_DIR = os.path.join(os.path.dirname(__file__), "recordings")

# Paste configuration
PASTE_DELAY = 0.1  # Delay between copy and paste (seconds)
