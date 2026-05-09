# Voice Dictation

Records speech via global hotkeys, transcribes it, and automatically copies to clipboard + pastes the result into the focused application.

## Hotkeys

All hotkeys are double-tap (press twice within 1 second) while holding Ctrl+Cmd.

- **Ctrl+Cmd+R** — Toggle recording. Second toggle stops, transcribes, and pastes.
- **Ctrl+Cmd+X** — Discard the active recording without transcribing.
- **Ctrl+Cmd+A** — Aside: pause the main recording, record an aside, transcribe and paste it, then resume the main recording where it left off.

## Transcription providers

Selected with `--provider`:

- `assemblyai` (default) — AssemblyAI REST API, slam-1 model. Requires `ASSEMBLYAI_API_KEY` (env var or AWS Secrets Manager entry `assemblyai_api_key` in `us-west-2`).
- `whisper` — Local Whisper model.

## Flags

- `--provider, -p` — `assemblyai` or `whisper`.
- `--save` — Save recordings to disk after transcription (off by default).
- `--verbose, -v` — Print debug info.

## Setup

```bash
brew install portaudio
pip install -r requirements.txt
```

macOS requires Microphone and Accessibility permissions for the terminal running the app.

## Files

- `voice_dictation.py` — entry point, hotkey handling, app state
- `audio_recorder.py` — PyAudio recording, pause/resume for asides
- `assemblyai_client.py`, `whisper_client.py` — transcription backends
- `transcriber.py` — provider interface
- `clipboard_manager.py` — clipboard + paste automation
- `config.py` — API key loading
