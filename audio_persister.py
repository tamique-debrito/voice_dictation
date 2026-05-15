"""AudioPersister — writes fixed-size audio chunks + manifest.json.

Consumes ``AcceptedAudioSegment`` via a direct put-queue from the
AudioPreprocessor (NOT via the bus). Accumulates PCM until
``chunk_seconds * 1000`` ms of accepted audio is collected, then writes
one file and publishes ``audio.chunk_committed`` on the bus.

The manifest maps ``[start_accepted_ms, end_accepted_ms]`` ranges to file
paths so replay / annotation can seek by accepted-ms.

Format: WAV is always supported; MP3 requires ``lameenc`` (preferred) or
``pydub``+ffmpeg. If MP3 is configured but no encoder is importable, falls
back to WAV with a logged warning.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import struct
import threading
from typing import Optional

from .clock import AcceptedClock, next_seq
from .event_bus import EventBus
from .runtime_config import AudioCaptureConfig, AudioPersistConfig
from .types import AcceptedAudioSegment, Event, TOPIC_AUDIO_CHUNK_COMMITTED


logger = logging.getLogger(__name__)


def _mp3_encoder_available() -> Optional[str]:
    try:
        import lameenc  # noqa: F401
        return "lameenc"
    except Exception:
        pass
    try:
        from pydub import AudioSegment  # noqa: F401
        return "pydub"
    except Exception:
        pass
    return None


def _write_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    """Write a minimal RIFF/WAVE file. Mono int16 only."""
    n_samples = len(pcm) // 2
    byte_rate = sample_rate * 2  # mono * 2 bytes
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # fmt chunk size
        f.write(struct.pack("<H", 1))            # PCM
        f.write(struct.pack("<H", 1))            # channels (mono)
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", 2))            # block align
        f.write(struct.pack("<H", 16))           # bits per sample
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm)))
        f.write(pcm)


def _write_mp3(path: str, pcm: bytes, sample_rate: int, bitrate_kbps: int) -> bool:
    encoder = _mp3_encoder_available()
    if encoder == "lameenc":
        import lameenc
        enc = lameenc.Encoder()
        enc.set_bit_rate(bitrate_kbps)
        enc.set_in_sample_rate(sample_rate)
        enc.set_channels(1)
        enc.set_quality(2)
        data = enc.encode(pcm) + enc.flush()
        with open(path, "wb") as f:
            f.write(data)
        return True
    if encoder == "pydub":
        from pydub import AudioSegment
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=sample_rate, channels=1)
        seg.export(path, format="mp3", bitrate=f"{bitrate_kbps}k")
        return True
    return False


class AudioPersister:
    def __init__(
        self,
        config: AudioPersistConfig,
        audio_cfg: AudioCaptureConfig,
        bus: EventBus,
        clock: AcceptedClock,
        session_dir: str,
        stop_event: threading.Event,
    ) -> None:
        self._cfg = config
        self._audio = audio_cfg
        self._bus = bus
        self._clock = clock
        self._stop = stop_event
        self._session_dir = session_dir
        self._audio_dir = os.path.join(session_dir, "audio")
        self._manifest_path = os.path.join(self._audio_dir, "manifest.json")
        os.makedirs(self._audio_dir, exist_ok=True)

        # Pick format with graceful fallback.
        self._format = config.format
        if self._format == "mp3" and _mp3_encoder_available() is None:
            logger.warning(
                "AudioPersister: format=mp3 but no encoder available "
                "(install lameenc or pydub+ffmpeg); falling back to wav",
            )
            self._format = "wav"

        # Chunk accumulator.
        self._chunk_pcm = bytearray()
        self._chunk_start_accepted_ms: Optional[int] = None
        self._chunk_end_accepted_ms = 0
        self._chunk_target_ms = config.chunk_seconds * 1000
        self._chunk_idx = 0

        # Manifest in-memory.
        self._manifest: dict = {
            "session_dir": session_dir,
            "format": self._format,
            "sample_rate": audio_cfg.sample_rate,
            "chunks": [],
        }
        self._write_manifest()

        # Input queue + worker thread.
        self.input_q: queue.Queue[AcceptedAudioSegment] = queue.Queue(maxsize=512)
        self._thread: Optional[threading.Thread] = None

        logger.info(
            "AudioPersister initialized (format=%s, chunk_seconds=%d, dir=%s)",
            self._format, config.chunk_seconds, self._audio_dir,
        )

    # ------------------------------------------------------------------
    # Sink protocol (called by AudioPreprocessor)
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "enabled": self._cfg.enabled,
            "format": self._format,
            "chunks_committed": self._chunk_idx,
            "pending_pcm_bytes": len(self._chunk_pcm),
            "pending_ms": (
                (self._chunk_end_accepted_ms - self._chunk_start_accepted_ms)
                if self._chunk_start_accepted_ms is not None else 0
            ),
            "input_q_size": self.input_q.qsize(),
            "input_q_maxsize": self.input_q.maxsize,
        }

    def put(self, segment: AcceptedAudioSegment, timeout: float = 0.5) -> None:
        if not self._cfg.enabled:
            return
        try:
            self.input_q.put(segment, timeout=timeout)
        except queue.Full:
            logger.warning("AudioPersister input queue full; dropping segment")

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        if not self._cfg.enabled:
            logger.info("AudioPersister disabled by config")
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="AudioPersister", daemon=True,
        )
        self._thread.start()
        logger.info("AudioPersister.start")

    def shutdown(self) -> None:
        if self._thread is None:
            return
        # Drain remaining audio into a final chunk so nothing is lost.
        # Worker will see _stop and exit after one more pass.
        try:
            self.input_q.put_nowait(None)  # sentinel  # type: ignore[arg-type]
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)
        self._flush_current_chunk()
        self._thread = None
        logger.info("AudioPersister.shutdown (chunks=%d)", self._chunk_idx)

    def _loop(self) -> None:
        while True:
            try:
                seg = self.input_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if seg is None:
                return
            self._accept(seg)

    def _accept(self, seg: AcceptedAudioSegment) -> None:
        if self._chunk_start_accepted_ms is None:
            self._chunk_start_accepted_ms = seg.start_accepted_ms
        self._chunk_pcm.extend(seg.pcm)
        self._chunk_end_accepted_ms = seg.end_accepted_ms
        accumulated = self._chunk_end_accepted_ms - self._chunk_start_accepted_ms
        if accumulated >= self._chunk_target_ms:
            self._flush_current_chunk()

    def _flush_current_chunk(self) -> None:
        if not self._chunk_pcm or self._chunk_start_accepted_ms is None:
            return
        idx = self._chunk_idx
        ext = self._format
        filename = f"chunk_{idx:04d}.{ext}"
        path = os.path.join(self._audio_dir, filename)
        try:
            if ext == "wav":
                _write_wav(path, bytes(self._chunk_pcm), self._audio.sample_rate)
            elif ext == "mp3":
                if not _write_mp3(
                    path, bytes(self._chunk_pcm),
                    self._audio.sample_rate, self._cfg.mp3_bitrate_kbps,
                ):
                    logger.error("MP3 encode failed; writing WAV instead")
                    path = path.rsplit(".", 1)[0] + ".wav"
                    _write_wav(path, bytes(self._chunk_pcm), self._audio.sample_rate)
                    ext = "wav"
            else:
                raise ValueError(f"unknown format {ext!r}")
        except Exception:
            logger.exception("AudioPersister write failed for %s", path)
            self._reset_chunk()
            return

        record = {
            "file": os.path.basename(path),
            "format": ext,
            "start_accepted_ms": self._chunk_start_accepted_ms,
            "end_accepted_ms": self._chunk_end_accepted_ms,
            "duration_ms": self._chunk_end_accepted_ms - self._chunk_start_accepted_ms,
        }
        self._manifest["chunks"].append(record)
        self._write_manifest()

        self._bus.publish(Event(
            topic=TOPIC_AUDIO_CHUNK_COMMITTED,
            emit_accepted_ms=self._clock.now_accepted_ms(),
            seq=next_seq(),
            payload=record,
        ))

        self._chunk_idx += 1
        self._reset_chunk()

    def _reset_chunk(self) -> None:
        self._chunk_pcm.clear()
        self._chunk_start_accepted_ms = None
        self._chunk_end_accepted_ms = 0

    def _write_manifest(self) -> None:
        tmp = self._manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp, self._manifest_path)
