"""v2 runtime config.

Mirrors voice_dictation/runtime_config.py but adds the v2-specific knobs:
- audio.persist.format / chunk_seconds
- preprocessor.voiced_gate_*
- per-stream min_window_seconds and flush_on_markers
- bus queue sizes

JSON file: local_config.json next to this module (separate from the v1 file
so the two apps don't fight over it during the transition).
"""

from __future__ import annotations

import json
import os
import platform as _platform
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


_DEFAULT_MARKERS = [
    {"key": "1", "type": "assistant_annotation",
     "description": "Assistant action for this point in the transcript."},
    {"key": "2", "type": "note",
     "description": "Personal note; not directed at any agent."},
]


def _default_hotkey_modifiers() -> list[str]:
    return ["cmd", "shift"] if _platform.system() == "Windows" else ["ctrl", "cmd"]


@dataclass
class AudioCaptureConfig:
    sample_rate: int = 16000     # webrtcvad requires 8/16/32/48 kHz
    channels: int = 1
    format: str = "int16"
    chunk_size: int = 1024       # ~64 ms at 16 kHz


@dataclass
class AudioPersistConfig:
    enabled: bool = True
    format: str = "mp3"          # "wav" | "mp3" — "wav" needed for replay-driven transcriber re-runs
    chunk_seconds: int = 300     # 5 min
    mp3_bitrate_kbps: int = 64


@dataclass
class PreprocessorConfig:
    """Single VAD + silence-strip + accepted-ms clock."""
    aggressiveness: int = 2              # webrtcvad 0..3
    # Edge-triggered "silence boundary" — same semantics as v1's VadConfig.silence_ms.
    silence_boundary_ms: int = 500
    # Sliding-window voiced gate over the most recent N sub-frames. A sub-frame
    # is passed through as accepted audio iff the gate considers the window
    # voiced. Low-latency: worst added latency ≈ VAD_FRAME_MS (20 ms).
    voiced_gate_frames: int = 5          # 5 frames × 20 ms = 100 ms window
    voiced_gate_min_voiced: int = 2      # need ≥2/5 voiced sub-frames to admit


@dataclass
class FasterWhisperConfig:
    model: str = "small.en"
    compute: str = "int8"
    device: str = ""                     # empty = auto-detect
    beam_size: int = 1
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.9
    log_prob_threshold: float = -0.5


@dataclass
class StreamConfig:
    """Per-stream windowing + transcription. Same class for fast and hq."""
    enabled: bool = True
    label: str = "fast"
    max_window_seconds: float = 25.0
    # Don't flush on a marker until this much accepted audio has accumulated.
    # Defaults to ~50% of max_window_seconds, which keeps windows in a
    # range Whisper handles well (long enough for context, short enough
    # to not stall transcription). Clipboard-end markers bypass this so
    # end-press paste is always immediate regardless of window size.
    min_window_seconds: float = 12.5
    # If True, marker stream events (silence_boundary / clipboard_end_marker)
    # cause an early flush when min_window_seconds has been reached. Fast = True,
    # HQ = False by default (HQ waits for max_window_seconds).
    flush_on_markers: bool = True
    window_q_maxsize: int = 8
    # Skip windows whose RMS amplitude (float32, normalized [-1, 1]) is
    # below this threshold. The silence-strip preprocessor admits anything
    # webrtcvad calls voiced, but webrtcvad has a much looser threshold
    # than Whisper benefits from — sub-speech background noise burns 4-9s
    # of inference time per window and tends to produce hallucinated text.
    # Empirical separation in real recordings: speech ≥ 0.018, noise ≤ 0.010.
    min_rms: float = 0.012
    fw: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)


def _default_hq_stream() -> "StreamConfig":
    return StreamConfig(
        enabled=True,
        label="hq",
        max_window_seconds=40.0,
        min_window_seconds=20.0,
        flush_on_markers=False,
        window_q_maxsize=16,
        fw=FasterWhisperConfig(
            model="distil-large-v3",
            compute="int8",
            beam_size=5,
            condition_on_previous_text=True,
        ),
    )


@dataclass
class BusConfig:
    persister_max_queued: int = 4096
    widget_max_queued: int = 8192


@dataclass
class ManagerConfig:
    """TranscriptStreamManager tunables."""
    # Paste fires when the fast stream's watermark advances to within
    # this many ms of the press's accepted_ms. Whisper's word-boundary
    # rounding can leave the last segment ending ~hundreds of ms short
    # of the press time; this absorbs that without making the user wait
    # for an extra silence boundary or window flush.
    complete_tolerance_ms: int = 200


@dataclass
class HotkeyConfig:
    no_modifiers: bool = False
    modifiers: list[str] = field(default_factory=_default_hotkey_modifiers)
    keys: dict[str, str] = field(default_factory=lambda: {
        "toggle_recording": "r",
        "discard_recording": "x",
        "toggle_aside": "a",
    })


@dataclass
class PasteConfig:
    delay: float = 0.1
    drain_timeout: float = 6.0


@dataclass
class AppConfig:
    sessions_dir: str = ""               # empty = <module_dir>/sessions
    widget_port: int = 0                 # 0 = let OS pick
    transcript_stream_port: int = 8767   # different from v1's 8766
    hf_hub_offline: bool = True
    debug_flag_key: str = "e"
    # Persist post-filter committed press events. Raw pre-filter keystrokes
    # are NOT persisted unless this is True (then they go to
    # user_keystrokes_raw.jsonl).
    persist_raw_keystrokes: bool = False


@dataclass
class RuntimeConfig:
    audio: AudioCaptureConfig = field(default_factory=AudioCaptureConfig)
    audio_persist: AudioPersistConfig = field(default_factory=AudioPersistConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    fast: StreamConfig = field(default_factory=StreamConfig)
    hq: StreamConfig = field(default_factory=_default_hq_stream)
    bus: BusConfig = field(default_factory=BusConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    paste: PasteConfig = field(default_factory=PasteConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    markers: list[dict] = field(default_factory=lambda: list(_DEFAULT_MARKERS))
    app: AppConfig = field(default_factory=AppConfig)


def _merge_into(target: Any, src: dict) -> None:
    for key, val in src.items():
        if not hasattr(target, key):
            continue
        cur = getattr(target, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _merge_into(cur, val)
        else:
            setattr(target, key, val)


def _auto_detect_device() -> str:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve(cfg: RuntimeConfig, module_dir: str) -> None:
    if not cfg.app.sessions_dir:
        cfg.app.sessions_dir = os.path.join(module_dir, "sessions")
    for stream in (cfg.fast, cfg.hq):
        if not stream.fw.device:
            stream.fw.device = _auto_detect_device()
    # Enforce label so internal code can rely on cfg.fast.label == "fast" etc.
    cfg.fast.label = "fast"
    cfg.hq.label = "hq"


def load_runtime_config(
    path: Optional[str] = None,
    module_dir: Optional[str] = None,
) -> RuntimeConfig:
    if module_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = RuntimeConfig()
    if path is None:
        candidate = os.path.join(module_dir, "local_config.json")
        if os.path.exists(candidate):
            path = candidate
    if path and os.path.exists(path):
        with open(path) as f:
            _merge_into(cfg, json.load(f))
    _resolve(cfg, module_dir)
    return cfg


def to_dict(cfg: RuntimeConfig) -> dict:
    return asdict(cfg)


def apply_dict_to_config(cfg: RuntimeConfig, data: dict) -> None:
    """In-place merge of a dict (parsed from widget PUT body) into ``cfg``.

    Unknown keys are silently ignored so old / partial bodies keep working.
    Type coercion is intentionally minimal.
    """
    _merge_into(cfg, data)


def save_runtime_config(
    cfg: RuntimeConfig,
    path: Optional[str] = None,
    module_dir: Optional[str] = None,
) -> str:
    if module_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(module_dir, "local_config.json")
    data = to_dict(cfg)
    data["app"]["sessions_dir"] = ""
    for stream in ("fast", "hq"):
        data[stream]["fw"]["device"] = ""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path
