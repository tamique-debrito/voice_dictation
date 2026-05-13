"""Legacy configuration facade.

All real config now lives in ``runtime_config.py`` (loaded from
``local_config.json`` with env-var overrides on top). This module
re-exports the legacy module-level constants so existing callers keep
working without changes during the Phase 0/1 transition. New code
should read directly from ``runtime_config.CFG``.
"""

import os
from dotenv import load_dotenv

# Load .env file from same directory as this file (env vars need to be in
# the environment before runtime_config.load_runtime_config runs).
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from runtime_config import CFG  # noqa: E402  (after dotenv)


# Audio recording configuration
SAMPLE_RATE = CFG.audio.sample_rate
CHANNELS = CFG.audio.channels
FORMAT = CFG.audio.format
CHUNK_SIZE = CFG.audio.chunk_size

# Audio file storage (regular start/stop app only)
AUDIO_SAVE_DIR = CFG.persistent.audio_save_dir

# Paste configuration
PASTE_DELAY = CFG.paste.delay

# Persistent streaming mode
TRANSCRIPTS_DIR = CFG.persistent.transcripts_dir
CHUNK_TOKEN_TARGET = CFG.persistent.chunk_token_target
SILENCE_MS = CFG.vad.silence_ms
VAD_AGGRESSIVENESS = CFG.vad.aggressiveness
MIN_VOICED_FRAC = CFG.vad.min_voiced_frac
MIN_VOICED_MS = CFG.vad.min_voiced_ms

# Marker token format. The capture-group regex must match keys used by
# session_writer / the skill output parsers.
MARKER_TOKEN_RE = r"<<<MARKER:([a-z_]+):(start|end)>>>"


def marker_start(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:start>>>"


def marker_end(type_name: str) -> str:
    return f"<<<MARKER:{type_name}:end>>>"


# Built-in marker types for recording/aside clipboard windows.
RECORDING_MARKER_TYPE = "recording"
ASIDE_MARKER_TYPE = "aside"

DEFAULT_MARKERS = CFG.markers

# faster-whisper config (fast stream — Phase 3 adds HQ).
FW_MODEL = CFG.persistent.fast.fw.model
FW_COMPUTE = CFG.persistent.fast.fw.compute
FW_DEVICE = CFG.persistent.fast.fw.device
FW_NO_SPEECH_THRESHOLD = CFG.persistent.fast.fw.no_speech_threshold
FW_LOG_PROB_THRESHOLD = CFG.persistent.fast.fw.log_prob_threshold

TRANSCRIPT_STREAM_PORT = CFG.persistent.transcript_stream_port
WIDGET_PORT = CFG.persistent.widget_port
