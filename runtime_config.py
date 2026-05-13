"""Single source of truth for persistent-app runtime configuration.

All tunables (audio, VAD, faster-whisper, ports, markers, hotkeys) live
in ``local_config.json`` next to this file. The widget settings page
can edit + persist this file at runtime; model swaps and other live-
applicable changes take effect immediately.

Environment variables are NOT used for configuration. The only env vars
this module sets (HF_HUB_OFFLINE) are implementation details forwarded
to third-party libraries that require an env var; they're driven by
fields in this config.

``config.py`` is a thin legacy facade that re-exports module-level
constants from the singleton ``CFG`` so older imports keep working
during the transition.
"""

from __future__ import annotations

import json
import os
import platform as _platform
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Default values (canonical place — JSON file and env vars override these).
# ---------------------------------------------------------------------------

_DEFAULT_MARKERS = [
    {
        "key": "1",
        "type": "assistant_annotation",
        "description": (
            "What an assistant should do at this point, or in reaction to "
            "the most recent utterance."
        ),
    },
    {
        "key": "2",
        "type": "note",
        "description": (
            "A personal note for the user; not directed at any agent."
        ),
    },
]


def _default_hotkey_modifiers() -> list[str]:
    return ["cmd", "shift"] if _platform.system() == "Windows" else ["ctrl", "cmd"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    format: str = "int16"
    chunk_size: int = 1024


@dataclass
class VadConfig:
    silence_ms: int = 500
    aggressiveness: int = 2          # 0..3 (webrtcvad)
    min_voiced_ms: int = 750
    min_voiced_frac: float = 0.35


@dataclass
class FasterWhisperConfig:
    """Per-stream faster-whisper settings."""

    model: str = "small.en"
    compute: str = "int8"
    device: str = ""                 # empty = auto-detect
    beam_size: int = 1
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.9
    log_prob_threshold: float = -0.5


@dataclass
class StreamConfig:
    """Per-stream window + transcription settings."""

    enabled: bool = True
    max_window_seconds: float = 25.0
    # Max number of unprocessed AudioWindows the aggregator may buffer
    # before it starts dropping with reason=dropped_queue_full. Tradeoff:
    # larger values protect against transient transcriber lag (HQ model
    # backlog, cold-start) but consume more RAM (~3MB/window worst case).
    window_q_maxsize: int = 8
    fw: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)


def _default_hq_stream() -> "StreamConfig":
    # HQ stream is off until Phase 3 wires it in; this default reflects the
    # eventual intended settings (medium.en, beam search, prev-text context,
    # larger windows). Override via local_config.json or env vars.
    return StreamConfig(
        enabled=False,
        # 40s gives faster-whisper more cross-window context than fast's 25s
        # without bumping into the 30s-30s context window boundary penalty.
        # The extra context past ~40s helps less than buffer-fill cost.
        max_window_seconds=40.0,
        # HQ falls behind realtime on cold-start (model load) and during
        # bursts; give it room to absorb backlog without dropping. 64 × ~3MB
        # = ~200MB worst case, which buys ~15-20 minutes of buffered speech.
        window_q_maxsize=64,
        fw=FasterWhisperConfig(
            model="medium.en",
            compute="int8",
            beam_size=5,
            condition_on_previous_text=True,
        ),
    )


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
class PersistentConfig:
    chunk_token_target: int = 2000
    transcripts_dir: str = ""        # empty = <module_dir>/transcripts
    audio_save_dir: str = ""         # empty = <module_dir>/recordings
    transcript_stream_port: int = 8766
    widget_port: int = 0             # 0 = let OS pick
    # When True, sets HF_HUB_OFFLINE=1 before faster-whisper imports so the
    # model load skips the "is there a newer version?" HF check (which can
    # stall 60s+ on a blocked network). Cached models load instantly.
    # Set False to allow downloads of new models (e.g. enabling an HQ model
    # for the first time). --check-updates always forces online for one run.
    hf_hub_offline: bool = True
    # Persist debug_events.jsonl + status_snapshots.jsonl under the session
    # directory so the entire widget state (timeline, transcript tail, paste
    # log, drops, capture mode) can be replayed post-session for debugging.
    # Off by default would lose information we usually want; cheap on disk.
    debug_recording: bool = True
    # Seconds between status-snapshot writes during recording. Smaller =
    # finer replay temporal resolution; larger = less I/O.
    debug_snapshot_interval_s: float = 2.0
    fast: StreamConfig = field(default_factory=StreamConfig)
    hq: StreamConfig = field(default_factory=_default_hq_stream)


@dataclass
class RuntimeConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    paste: PasteConfig = field(default_factory=PasteConfig)
    persistent: PersistentConfig = field(default_factory=PersistentConfig)
    markers: list[dict] = field(default_factory=lambda: list(_DEFAULT_MARKERS))
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)


# ---------------------------------------------------------------------------
# JSON load + merge
# ---------------------------------------------------------------------------


def _merge_into(target: Any, src: dict) -> None:
    """Recursively merge a JSON dict into a dataclass instance in-place.

    Unknown keys are silently ignored so old configs keep loading after
    schema additions. Type coercion is intentionally minimal — JSON
    int/float/string/bool/list/dict are the only allowed inputs.
    """
    for key, val in src.items():
        if not hasattr(target, key):
            continue
        cur = getattr(target, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _merge_into(cur, val)
        else:
            setattr(target, key, val)


def _auto_detect_device() -> str:
    """Pick 'cuda' when a CUDA-capable GPU is visible to CTranslate2,
    else 'cpu'. Any failure path falls back to cpu."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_paths_and_device(cfg: RuntimeConfig, module_dir: str) -> None:
    if not cfg.persistent.transcripts_dir:
        cfg.persistent.transcripts_dir = os.path.join(module_dir, "transcripts")
    if not cfg.persistent.audio_save_dir:
        cfg.persistent.audio_save_dir = os.path.join(module_dir, "recordings")
    # Device auto-detect applies per-stream when empty.
    for stream in (cfg.persistent.fast, cfg.persistent.hq):
        if not stream.fw.device:
            stream.fw.device = _auto_detect_device()


def load_runtime_config(
    path: Optional[str] = None,
    module_dir: Optional[str] = None,
) -> RuntimeConfig:
    """Load + merge config JSON, apply env overrides, resolve defaults."""
    if module_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = RuntimeConfig()
    if path is None:
        candidate = os.path.join(module_dir, "local_config.json")
        if os.path.exists(candidate):
            path = candidate
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        _merge_into(cfg, data)
    _resolve_paths_and_device(cfg, module_dir)
    return cfg


def to_dict(cfg: RuntimeConfig) -> dict:
    """Convert a RuntimeConfig back to a JSON-safe dict (used by Phase 6
    widget /config GET)."""
    return asdict(cfg)


def save_runtime_config(cfg: RuntimeConfig, path: Optional[str] = None,
                        module_dir: Optional[str] = None) -> str:
    """Write ``cfg`` to ``path`` (default: ``<module_dir>/local_config.json``).

    Returns the path written. Used by the widget /config PUT endpoint.

    INVARIANT: every field in ``RuntimeConfig`` (and therefore everything
    written here) must be editable via the widget settings modal. When
    adding new dataclass fields, add matching form controls in
    ``widget.py`` so the user never has to hand-edit this file.
    """
    if module_dir is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
    if path is None:
        path = os.path.join(module_dir, "local_config.json")
    data = to_dict(cfg)
    # Strip resolved-path defaults so the file stays portable across machines.
    data["persistent"]["transcripts_dir"] = ""
    data["persistent"]["audio_save_dir"] = ""
    for stream in ("fast", "hq"):
        data["persistent"][stream]["fw"]["device"] = ""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


def apply_dict_to_config(cfg: RuntimeConfig, data: dict) -> None:
    """In-place merge of a dict (parsed from widget PUT body) into ``cfg``."""
    _merge_into(cfg, data)


# Singleton — module-level constants in config.py and load_config in
# hotkeys.py both pull from this so the load happens exactly once per
# process.
CFG: RuntimeConfig = load_runtime_config()
