"""Audio recording functionality using PyAudio."""

import os
import pyaudio
import shutil
import wave
import tempfile
import threading
from datetime import datetime
from typing import Optional
from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, AUDIO_SAVE_DIR


class AudioRecorder:
    """Records audio from microphone using PyAudio."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames = []
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.temp_file: Optional[str] = None

    def start_recording(self, initial_frames=None):
        """Start recording audio from microphone.

        If initial_frames is provided, the new recording will be appended
        to those frames (used to resume a paused recording).
        """
        if self.is_recording:
            return

        # Close any existing stream first
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.frames = list(initial_frames) if initial_frames else []
        self.is_recording = True

        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        # Start recording in background thread
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.start()

    def _record(self):
        """Background thread for recording audio frames."""
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
                break

    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save to temporary WAV file.

        Returns:
            str: Path to temporary WAV file, or None if no frames recorded
        """
        if not self.is_recording:
            return None

        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join()

        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Check if we have any audio data
        if not self.frames:
            return None

        # Save to temporary WAV file
        temp_fd, self.temp_file = tempfile.mkstemp(suffix='.wav')

        with wave.open(self.temp_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

        return self.temp_file

    def pause_recording(self) -> list:
        """Stop the audio stream but return the captured frames without
        writing a temp file. Frames are also cleared from the recorder so
        a fresh recording can be started.

        Returns:
            list: The frames captured so far (may be empty).
        """
        if not self.is_recording:
            return list(self.frames)

        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        frames = self.frames
        self.frames = []
        return frames

    def save_recording(self) -> Optional[str]:
        """
        Save the current recording to the persistent recordings directory.

        Returns:
            str: Path to the saved WAV file, or None if no temp file exists.
        """
        if not self.temp_file:
            return None

        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        save_path = os.path.join(AUDIO_SAVE_DIR, filename)
        shutil.copy2(self.temp_file, save_path)
        return save_path

    def cleanup(self):
        """Clean up temporary files (called after each recording)."""
        if self.temp_file:
            try:
                os.remove(self.temp_file)
                self.temp_file = None
            except Exception:
                pass

    def shutdown(self):
        """Shutdown PyAudio (called when app exits)."""
        # Close stream if still open
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass

        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()
