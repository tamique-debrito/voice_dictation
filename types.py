"""Shared typed records exchanged between modules.

Two distinct categories:

1. Hot-path PCM payloads that flow over direct queues:
   - ``AcceptedAudioSegment``: AudioPreprocessor → AudioWindower / AudioPersister
   - ``AudioWindow``:           AudioWindower → Transcriber

2. Event envelopes that flow over the EventBus (low bandwidth, fan-out):
   - ``Event``: every bus message. ``topic`` decides which subscribers see it;
     ``emit_accepted_ms`` + ``seq`` form the deterministic replay key.

Every record uses accepted-ms (the AudioPreprocessor's voiced-only clock).
``real_ms`` only appears on the press-event payload for forensic purposes —
nothing downstream of the preprocessor schedules off wall-clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# PCM payloads (direct queues, not on the bus)
# ---------------------------------------------------------------------------


@dataclass
class AcceptedAudioSegment:
    """A run of consecutive voiced sub-frames emitted by the preprocessor.

    Boundaries here are sub-frame granular (20ms). Multiple segments may
    arrive in quick succession during voiced audio; the gap between two
    segments in accepted-ms is zero (silence has no duration here) even
    though wall-clock may have elapsed.
    """

    pcm: bytes
    start_accepted_ms: int
    end_accepted_ms: int


@dataclass
class AudioWindow:
    """A finalized window the Transcriber will hand to Whisper as one call."""

    pcm: bytes
    start_accepted_ms: int
    end_accepted_ms: int
    voiced_ms: int
    flush_reason: str            # "max_window" | "marker" | "marker_end_clipboard" | "shutdown"
    ends_on_marker_idx: Optional[int] = None
    window_id: int = 0


@dataclass
class Word:
    text: str
    start_accepted_ms: int
    end_accepted_ms: int
    probability: float = 0.0


@dataclass
class WindowCompletion:
    """Atomic 'window N is fully processed up to end_accepted_ms' marker.

    Emitted by the Transcriber on ``segment_q`` after all segments for a
    window have been pushed (or immediately if the window was skipped /
    dropped). The manager's drain loop processes segments and completions
    in order; when it pops a completion, every segment from windows with
    earlier ``end_accepted_ms`` has already been added to the buffer.
    That ordering is what makes ``transcribed_up_to`` a true watermark.
    """

    stream_label: str
    window_id: int
    end_accepted_ms: int
    dropped: bool = False  # True = windower/transcriber skipped this window


@dataclass
class Segment:
    """Output of one Whisper inference call on one AudioWindow."""

    text: str
    content_accepted_ms_start: int
    content_accepted_ms_end: int
    words: list[Word] = field(default_factory=list)
    window_id: int = 0
    ends_on_marker_idx: Optional[int] = None


# ---------------------------------------------------------------------------
# Event bus envelope
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """One message on the EventBus.

    ``payload`` is a JSON-serializable dict — the persisters write it
    directly. ``emit_accepted_ms`` + ``seq`` form the replay sort key.
    """

    topic: str
    emit_accepted_ms: int
    seq: int
    payload: dict[str, Any]


# Topic constants (string-typed for forward compatibility).

TOPIC_AUDIO_SILENCE = "audio.silence"
TOPIC_AUDIO_CHUNK_COMMITTED = "audio.chunk_committed"
TOPIC_WINDOWS_FAST = "windows.fast"
TOPIC_WINDOWS_HQ = "windows.hq"
TOPIC_SEGMENTS_FAST = "segments.fast"
TOPIC_SEGMENTS_HQ = "segments.hq"
TOPIC_USER_PRESSES = "user.presses"
TOPIC_USER_ACTIONS = "user.actions"
TOPIC_PASTE_ACTIONS = "paste.actions"
TOPIC_TRANSCRIPT_CANONICAL = "transcript.canonical"

TOPIC_PROGRESS_FAST = "progress.fast"
TOPIC_PROGRESS_HQ = "progress.hq"

# Status topics are namespaced by module label, e.g. "status.preprocessor".
TOPIC_STATUS_PREFIX = "status."
