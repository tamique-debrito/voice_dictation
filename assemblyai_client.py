"""AssemblyAI transcription provider."""

import os
import requests
import time
from typing import Optional
from transcriber import Transcriber

ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
SPEECH_MODEL = "best"
POLL_INTERVAL = 1.0
UPLOAD_TIMEOUT = 10.0
UPLOAD_MAX_ATTEMPTS = 5


class AssemblyAIClient(Transcriber):
    """Client for interacting with AssemblyAI API."""

    def __init__(self):
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ASSEMBLYAI_API_KEY not set. "
                "Add it to your .env file or set it as an environment variable."
            )
        self.headers = {
            "Authorization": self.api_key,
        }

    def upload_file(self, file_path: str, verbose: bool = False) -> str:
        """Upload audio file to AssemblyAI, retrying if a request hangs."""
        last_exc = None
        for attempt in range(1, UPLOAD_MAX_ATTEMPTS + 1):
            try:
                with open(file_path, 'rb') as f:
                    response = requests.post(
                        ASSEMBLYAI_UPLOAD_URL,
                        headers=self.headers,
                        data=f,
                        timeout=UPLOAD_TIMEOUT
                    )
                response.raise_for_status()
                return response.json()['upload_url']
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exc = e
                if verbose:
                    print(f"[DEBUG] Upload attempt {attempt}/{UPLOAD_MAX_ATTEMPTS} failed ({type(e).__name__}); retrying...")
        raise Exception(f"Upload failed after {UPLOAD_MAX_ATTEMPTS} attempts: {last_exc}")

    def request_transcription(self, audio_url: str) -> str:
        """Request transcription of uploaded audio."""
        json_data = {
            "audio_url": audio_url,
            "speech_model": SPEECH_MODEL
        }

        response = requests.post(
            ASSEMBLYAI_TRANSCRIPT_URL,
            headers={**self.headers, "Content-Type": "application/json"},
            json=json_data,
            timeout=30
        )

        response.raise_for_status()
        return response.json()['id']

    def poll_transcription(self, transcript_id: str, verbose: bool = False, timeout: float = 120.0) -> Optional[str]:
        """Poll for transcription completion."""
        url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
        start_time = time.time()
        poll_count = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(f"Transcription timed out after {timeout}s (polled {poll_count} times)")

            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            result = response.json()
            status = result['status']
            poll_count += 1

            if verbose:
                print(f"[DEBUG] Poll #{poll_count} ({elapsed:.1f}s): status={status}")

            if status == 'completed':
                return result['text']
            elif status == 'error':
                error_msg = result.get('error', 'Unknown error')
                raise Exception(f"Transcription failed: {error_msg}")

            time.sleep(POLL_INTERVAL)

    def transcribe_file(self, file_path: str, verbose: bool = False) -> Optional[str]:
        """Complete transcription workflow: upload, request, and poll."""
        if verbose:
            file_size = os.path.getsize(file_path)
            print(f"[DEBUG] File: {file_path} ({file_size} bytes)")

        if verbose:
            print("[DEBUG] Uploading file...")
        audio_url = self.upload_file(file_path, verbose=verbose)
        if verbose:
            print(f"[DEBUG] Upload complete: {audio_url[:80]}...")

        if verbose:
            print("[DEBUG] Requesting transcription...")
        transcript_id = self.request_transcription(audio_url)
        if verbose:
            print(f"[DEBUG] Transcript ID: {transcript_id}")

        if verbose:
            print("[DEBUG] Polling for completion...")
        return self.poll_transcription(transcript_id, verbose=verbose)
