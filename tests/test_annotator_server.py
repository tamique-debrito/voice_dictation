"""End-to-end smoke tests for AnnotatorServer routes."""

from __future__ import annotations

import http.client
import json
import struct
import tempfile
import unittest
from pathlib import Path

from voice_dictation.annotator.server import AnnotatorServer


def _write_wav(path: Path, n_samples: int = 16000) -> None:
    """Minimal 16kHz mono int16 WAV with `n_samples` zero samples."""
    sr, ch, bits = 16000, 1, 16
    byte_rate = sr * ch * (bits // 8)
    block = ch * (bits // 8)
    data_bytes = n_samples * block
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_bytes))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, ch, sr, byte_rate, block, bits))
        f.write(b"data")
        f.write(struct.pack("<I", data_bytes))
        f.write(b"\x00\x00" * n_samples)


def _build_session(td: Path) -> Path:
    sess = td / "s"
    (sess / "audio").mkdir(parents=True)
    _write_wav(sess / "audio" / "chunk_0000.wav", n_samples=8000)  # 0.5s
    manifest = {
        "format": "wav", "sample_rate": 16000,
        "chunks": [{
            "file": "chunk_0000.wav", "format": "wav",
            "start_accepted_ms": 0, "end_accepted_ms": 500, "duration_ms": 500,
        }],
    }
    (sess / "audio" / "manifest.json").write_text(json.dumps(manifest))
    seg = {
        "topic": "segments.fast", "emit_accepted_ms": 600, "seq": 1,
        "window_id": 1, "text": "hello world",
        "content_accepted_ms_start": 50, "content_accepted_ms_end": 450,
        "words": [
            {"text": " hello", "start_accepted_ms": 50, "end_accepted_ms": 250, "probability": 0.95},
            {"text": " world", "start_accepted_ms": 250, "end_accepted_ms": 450, "probability": 0.95},
        ],
    }
    (sess / "fast_segments.jsonl").write_text(json.dumps(seg) + "\n")
    return sess


class TestAnnotatorServer(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        sess = _build_session(Path(self._td.name))
        self.server = AnnotatorServer(sess, port=0)
        self.server.start()
        self.port = self.server.actual_port

    def tearDown(self):
        self.server.shutdown()
        self._td.cleanup()

    def _conn(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)

    def test_index_html_served(self):
        c = self._conn(); c.request("GET", "/annotate"); r = c.getresponse()
        self.assertEqual(r.status, 200)
        body = r.read().decode("utf-8")
        self.assertIn("voice_dictation", body.lower())
        self.assertIn("/annotation/chunks", body)

    def test_chunks_payload(self):
        c = self._conn(); c.request("GET", "/annotation/chunks"); r = c.getresponse()
        self.assertEqual(r.status, 200)
        body = json.loads(r.read())
        self.assertEqual(len(body["chunks"]), 1)
        self.assertEqual(body["chunks"][0]["segments"][0]["text"], "hello world")

    def test_state_initially_empty(self):
        c = self._conn(); c.request("GET", "/annotation/state"); r = c.getresponse()
        body = json.loads(r.read())
        self.assertEqual(body, {"segments": {}, "rejected_chunks": []})

    def test_audio_range_returns_206(self):
        c = self._conn()
        c.request("GET", "/annotation/audio/0", headers={"Range": "bytes=0-99"})
        r = c.getresponse()
        self.assertEqual(r.status, 206)
        self.assertTrue(r.getheader("Content-Range").startswith("bytes 0-99/"))
        self.assertEqual(len(r.read()), 100)

    def test_audio_full_returns_200(self):
        c = self._conn()
        c.request("GET", "/annotation/audio/0")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Accept-Ranges"), "bytes")

    def test_audio_bad_range_416(self):
        c = self._conn()
        c.request("GET", "/annotation/audio/0", headers={"Range": "bytes=9999999-"})
        r = c.getresponse()
        self.assertEqual(r.status, 416)

    def test_put_segment_then_state_reflects_it(self):
        c = self._conn()
        body = json.dumps({"text": "HELLO WORLD", "status": "edited"})
        c.request("PUT", "/annotation/0/fast:1",
                  body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        out = json.loads(r.read())
        self.assertTrue(out["ok"])
        c = self._conn(); c.request("GET", "/annotation/state"); r = c.getresponse()
        st = json.loads(r.read())
        self.assertEqual(st["segments"]["0|fast:1"]["text"], "HELLO WORLD")

    def test_put_chunk_reject(self):
        c = self._conn()
        body = json.dumps({"status": "rejected"})
        c.request("PUT", "/annotation/0", body=body,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse(); self.assertEqual(r.status, 200)
        c = self._conn(); c.request("GET", "/annotation/state"); r = c.getresponse()
        st = json.loads(r.read())
        self.assertEqual(st["rejected_chunks"], [0])

    def test_invalid_status_400(self):
        c = self._conn()
        body = json.dumps({"text": "x", "status": "bogus"})
        c.request("PUT", "/annotation/0/fast:1", body=body,
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        self.assertEqual(r.status, 400)

    def test_unknown_route_404(self):
        c = self._conn(); c.request("GET", "/no/such/thing"); r = c.getresponse()
        self.assertEqual(r.status, 404)


if __name__ == "__main__":
    unittest.main()
