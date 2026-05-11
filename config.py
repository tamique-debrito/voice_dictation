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
# ~2000 tokens ≈ ~1500 words ≈ ~10 min of speech.
CHUNK_TOKEN_TARGET = 2000

# Minimum continuous silence (ms) considered a "boundary" for chunk flushing
# and transcription window alignment.
SILENCE_MS = int(os.getenv("SILENCE_MS", "500"))

# webrtcvad aggressiveness (0..3). Higher = more frames classified as
# non-speech. See voice_dictation/README.md "Tuning" section.
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))

# Window-gating thresholds: skip transcribing a window when its voiced content
# falls below BOTH of these. Prevents Whisper hallucinations on near-silent
# audio (e.g. "okay okay okay", "thank you thank you"). A window must have
# at least MIN_VOICED_MS of voiced audio AND at least MIN_VOICED_FRAC of its
# frames marked voiced to be transcribed.
MIN_VOICED_FRAC = float(os.getenv("MIN_VOICED_FRAC", "0.15"))
MIN_VOICED_MS = int(os.getenv("MIN_VOICED_MS", "300"))

# Token format used for inline marker insertion in chunk_*.txt files.
# Type-name capture group must match keys used by skill / session_writer.
MARKER_TOKEN_RE = r"<<<MARKER:([a-z_]+):(start|end)>>>"

def marker_start(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:start>>>"

def marker_end(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:end>>>"

# Built-in marker types for recording/aside clipboard windows. These are
# kept separate from user-configurable DEFAULT_MARKERS because they're
# part of the core capture machinery, not user-configurable annotations.
# Format is the same so downstream parsers can recognize them uniformly.
RECORDING_MARKER_TYPE = "recording"
ASIDE_MARKER_TYPE = "aside"

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
# darwin always falls back to CPU. On a CUDA system we auto-detect and
# pick GPU. Override via env var if you want to force cpu or cuda.
FW_MODEL = os.getenv("FW_MODEL", "small.en")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")

# faster-whisper hallucination-mitigation thresholds. Defaults are tightened
# vs. faster-whisper's own defaults (0.6 / -1.0) to drop low-confidence
# decodes of near-silent audio. See voice_dictation/README.md "Tuning".
FW_NO_SPEECH_THRESHOLD = float(os.getenv("FW_NO_SPEECH_THRESHOLD", "0.8"))
FW_LOG_PROB_THRESHOLD = float(os.getenv("FW_LOG_PROB_THRESHOLD", "-0.8"))


def _auto_detect_device() -> str:
    """Pick 'cuda' when a CUDA-capable GPU is visible to CTranslate2, else 'cpu'.

    CTranslate2's CUDA build links against cuBLAS/cuDNN; on a CPU-only
    system the import still succeeds but get_cuda_device_count() returns 0.
    Any failure path falls back to cpu so the app keeps working.
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


FW_DEVICE = os.getenv("FW_DEVICE") or _auto_detect_device()

TRANSCRIPT_STREAM_PORT = int(os.getenv("TRANSCRIPT_STREAM_PORT", "8766"))
