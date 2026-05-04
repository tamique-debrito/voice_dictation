# Voice Dictation App

A Python application that records speech via global keyboard shortcuts, transcribes using AssemblyAI, and automatically pastes the transcribed text.

## Features

- **Global keyboard shortcuts**: Control recording from any application
- **AssemblyAI transcription**: High-quality speech-to-text using the slam-1 model
- **Automatic paste**: Transcribed text is copied to clipboard and pasted automatically
- **Real-time feedback**: Terminal UI shows recording and transcription status

## Prerequisites

- Python 3.8+
- macOS (for Cmd key shortcuts)
- AssemblyAI API key
- AWS credentials (optional, for Secrets Manager)
- athean-data repository (already present in this repo)

## Installation

### 1. Install Python dependencies

```bash
cd voice_dictation
pip install -r requirements.txt
```

### 2. Install PortAudio (required for PyAudio)

```bash
brew install portaudio
```

### 3. Configure API key

**Option A: Environment variable (recommended for development)**

```bash
export ASSEMBLYAI_API_KEY="your-api-key-here"
```

**Option B: AWS Secrets Manager (uses existing athean-data utility)**

Store your API key in AWS Secrets Manager with the name `assemblyai_api_key` in the `us-west-2` region.

The app will automatically use the `get_secret()` function from `athean-data/mwaa-task/secretsmanager.py`.

## Usage

### Start the application

```bash
python voice_dictation.py
```

Or make it executable:

```bash
chmod +x voice_dictation.py
./voice_dictation.py
```

### Keyboard shortcuts

- **Cmd+R**: Toggle recording (press once to start, press again to stop and transcribe)
- **Ctrl+C**: Exit the application

### Workflow

1. Launch the app in a terminal window
2. Focus on the application where you want to paste text (e.g., TextEdit, VS Code, Slack)
3. Press **Cmd+R** to start recording
4. Speak your text
5. Press **Cmd+R** again to stop recording and transcribe
6. The transcribed text will be automatically pasted into the focused application

## Permissions

On first run, macOS will prompt for permissions:

1. **Microphone access**: Required for recording audio
2. **Accessibility access**: Required for automatic paste functionality
   - Go to System Preferences → Security & Privacy → Privacy → Accessibility
   - Add Terminal (or your terminal app) to the list

## Troubleshooting

### "No module named 'pyaudio'"

Install PortAudio first:

```bash
brew install portaudio
pip install pyaudio
```

### "AssemblyAI API key not found"

Set the environment variable:

```bash
export ASSEMBLYAI_API_KEY="your-api-key-here"
```

### "Permission denied" for microphone

1. Go to System Preferences → Security & Privacy → Privacy → Microphone
2. Enable access for Terminal (or your terminal app)

### Paste not working

1. Go to System Preferences → Security & Privacy → Privacy → Accessibility
2. Add Terminal (or your terminal app) to the list
3. Restart the application

### No audio recorded

- Ensure your microphone is working
- Check that you're pressing Cmd+R properly
- Try speaking closer to the microphone

## Architecture

```
voice_dictation/
├── voice_dictation.py          # Main application entry point
├── audio_recorder.py           # PyAudio recording logic
├── assemblyai_client.py        # AssemblyAI API client
├── clipboard_manager.py        # Clipboard + paste automation
├── config.py                   # Configuration and API key loading
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

- **Audio format**: 16kHz mono WAV (AssemblyAI recommended)
- **Speech model**: slam-1 (latest multi-channel model)
- **Recording**: PyAudio with background threading
- **Hotkeys**: pynput keyboard listener
- **Transcription**: AssemblyAI REST API with polling
- **Clipboard**: pyperclip + pyautogui for paste automation

## Known Limitations

- Requires internet connection for transcription
- Transcription latency: 2-5 seconds depending on audio length
- Terminal must stay open while app is running
- Cmd+R may conflict with browser reload in some apps (global hotkey takes precedence)
- Temporary audio files saved to /tmp/ and cleaned up after transcription

## Future Enhancements

- Configurable hotkeys via config file
- Menu bar app (run as background service)
- Real-time streaming transcription
- Transcript history with timestamps
- Custom vocabulary for domain-specific terms
- Edit transcript before pasting
- Multiple language support
