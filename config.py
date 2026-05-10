"""Configuration for voice dictation app."""

import os
from dotenv import load_dotenv

# Load .env file from same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Audio recording configuration
SAMPLE_RATE = 16000  # 16kHz (AssemblyAI recommended; also valid for webrtcvad)
CHANNELS = 1  # Mono
FORMAT = "int16"  # 16-bit PCM
CHUNK_SIZE = 1024  # Frames per buffer

# Audio file storage (regular start/stop app only)
AUDIO_SAVE_DIR = os.path.join(os.path.dirname(__file__), "recordings")

# Paste configuration
PASTE_DELAY = 0.1  # Delay between copy and paste (seconds)

# ---------------------------------------------------------------------------
# Persistent streaming mode
# ---------------------------------------------------------------------------

# Where chunk_*.txt and session_meta.json get written.
TRANSCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "transcripts")

# Approximate target tokens per chunk file (~0.75 tokens per English word).
# Cuts only happen at silence boundaries, so chunks may exceed this slightly.
CHUNK_TOKEN_TARGET = 10000

# Minimum continuous silence (ms) considered a "boundary" for chunk flushing
# and transcription window alignment.
SILENCE_MS = 500

# Token format used for inline marker insertion in chunk_*.txt files.
# Type-name capture group must match keys used by skill / session_writer.
MARKER_TOKEN_RE = r"<<<MARKER:([a-z_]+):(start|end)>>>"

def marker_start(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:start>>>"

def marker_end(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:end>>>"

# Default marker types when local_config.json doesn't specify them.
DEFAULT_MARKERS = [
    {
        "key": "1",
        "type": "assistant_annotation",
        "description": "What an assistant should do at this point, or in reaction to the most recent utterance.",
    },
    {
        "key": "2",
        "type": "note",
        "description": "A personal note for the user; not directed at any agent.",
    },
]

# faster-whisper config. CTranslate2 has no Metal backend on macOS, so
# device=cpu is the only option here. small.en runs comfortably above
# real-time on M-series CPUs at int8.
FW_MODEL = os.getenv("FW_MODEL", "small.en")
FW_DEVICE = os.getenv("FW_DEVICE", "cpu")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")
