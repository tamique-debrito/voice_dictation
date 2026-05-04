"""Abstract base class for transcription providers."""

from abc import ABC, abstractmethod
from typing import Optional


class Transcriber(ABC):
    """Interface for audio transcription providers."""

    @abstractmethod
    def transcribe_file(self, file_path: str, verbose: bool = False) -> Optional[str]:
        """
        Transcribe an audio file to text.

        Args:
            file_path: Path to audio file
            verbose: If True, print debug info

        Returns:
            str: Transcribed text, or None if transcription failed
        """
        ...
