"""Local Whisper transcription provider."""

import os
import ssl
from typing import Optional
from transcriber import Transcriber

DEFAULT_MODEL = "base"


class WhisperClient(Transcriber):
    """Transcribe audio files locally using OpenAI Whisper."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Allow Whisper to download model files on macOS without cert issues.
        ssl._create_default_https_context = ssl._create_unverified_context

        import whisper  # imported lazily so AssemblyAI users don't need it installed
        self._whisper = whisper
        self.model_name = model_name
        self.model = whisper.load_model(model_name)

    def transcribe_file(self, file_path: str, verbose: bool = False) -> Optional[str]:
        if verbose:
            file_size = os.path.getsize(file_path)
            print(f"[DEBUG] File: {file_path} ({file_size} bytes)")
            print(f"[DEBUG] Transcribing locally with Whisper model '{self.model_name}'...")

        result = self.model.transcribe(file_path, language="en", fp16=False)
        text = result.get("text", "").strip()

        if verbose:
            print(f"[DEBUG] Whisper produced {len(text)} chars")

        return text or None
