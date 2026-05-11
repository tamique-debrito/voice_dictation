# Voice Dictation

Records speech via global hotkeys, transcribes it, and automatically copies to clipboard + pastes the result into the focused application.

## Hotkeys

All hotkeys are double-tap (press twice within 1 second) while holding the platform modifiers (Mac: Cmd+Shift, Windows: Win+Ctrl).

- **R** — Toggle recording. Second toggle stops, transcribes, and pastes.
- **X** — Discard the active recording without transcribing.
- **A** — Aside: pause the main recording, record an aside, transcribe and paste it, then resume the main recording where it left off.

## Transcription providers

Selected with `--provider`:

- `assemblyai` (default) — AssemblyAI REST API, slam-1 model. Requires `ASSEMBLYAI_API_KEY` (env var or AWS Secrets Manager entry `assemblyai_api_key` in `us-west-2`).
- `whisper` — Local Whisper model.

## Key binding configuration

Hotkeys can be overridden with a local JSON file (gitignored). Place `local_config.json` in the repo root or pass `--config path/to/file.json`.

```json
{
  "modifiers": ["shift", "ctrl"],
  "keys": {
    "toggle_recording": "r",
    "discard_recording": "x",
    "toggle_aside": "a"
  }
}
```

Supported modifiers: `ctrl`, `cmd`, `alt`, `shift`.

## Flags

- `--provider, -p` — `assemblyai` or `whisper`.
- `--config, -c` — Path to key binding config JSON (auto-loads `local_config.json` if present).
- `--save` — Save recordings to disk after transcription (off by default).
- `--verbose, -v` — Print debug info.

## Setup

```bash
brew install portaudio
pip install -r requirements.txt
```

macOS requires Microphone and Accessibility permissions for the terminal running the app.

## Tuning hallucinations / silence behavior (persistent mode)

The always-on persistent app (`voice_dictation_persistent.py`) can occasionally emit Whisper hallucinations during silence — degenerate repeats like `okay okay okay` or `thank you. thank you.` These come from faster-whisper decoding near-silent audio. Three layers of defense are in place; each is configurable via env var (or `.env`) and read at startup from `config.py`.

| Knob | Default | What it controls |
|------|---------|------------------|
| `SILENCE_MS` | 500 | Continuous silence (ms) that triggers a window flush boundary. Raise to require longer pauses before transcribing. |
| `VAD_AGGRESSIVENESS` | 3 | webrtcvad level (0–3). Higher = more frames classified as non-speech. Lower to 2 if real speech is being clipped. |
| `MIN_VOICED_FRAC` | 0.25 | Minimum fraction of voiced frames in a window for it to be transcribed. Raise to drop more low-content windows; lower if real short utterances are being suppressed. |
| `MIN_VOICED_MS` | 500 | Absolute minimum voiced audio (ms) per window. Independent of fraction — protects against single-blip windows. Both this and `MIN_VOICED_FRAC` must be met for a window to be transcribed. |
| `FW_NO_SPEECH_THRESHOLD` | 0.9 | faster-whisper no-speech probability cutoff (its default is 0.6). Raise toward 1.0 to discard more low-confidence segments; lower if real speech is being dropped. |
| `FW_LOG_PROB_THRESHOLD` | -0.5 | faster-whisper avg-log-prob cutoff (its default is -1.0). Raise toward 0 to require higher-confidence decodes. |
| `FW_MODEL` | `small.en` | faster-whisper model name. Options include `tiny.en`, `base.en`, `small.en`, `medium.en`. |
| `FW_COMPUTE` | `int8` | CTranslate2 quantization (`int8`, `int16`, `float16`, `float32`). |
| `FW_DEVICE` | auto | `cpu` or `cuda`. Auto-detects CUDA; macOS always falls back to CPU. |

Set any of these in `voice_dictation/.env` or as shell env vars before launching the persistent app. Example to make silence handling stricter:

```bash
export VAD_AGGRESSIVENESS=3
export MIN_VOICED_FRAC=0.25
export FW_NO_SPEECH_THRESHOLD=0.9
```

## Files

- `voice_dictation.py` — entry point, hotkey handling, app state
- `audio_recorder.py` — PyAudio recording, pause/resume for asides
- `assemblyai_client.py`, `whisper_client.py` — transcription backends
- `transcriber.py` — provider interface
- `clipboard_manager.py` — clipboard + paste automation
- `config.py` — API key loading
