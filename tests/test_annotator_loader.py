"""Loader + boundary splitting for the annotation view."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from voice_dictation.annotator.loader import load_chunks


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _seg(seq: int, start: int, end: int, words: list[tuple[str, int, int]]) -> dict:
    """Helper to build a segments.fast.jsonl-style row."""
    return {
        "topic": "segments.fast",
        "emit_accepted_ms": end + 100,
        "seq": seq,
        "window_id": seq,
        "text": "".join(w[0] for w in words).strip(),
        "content_accepted_ms_start": start,
        "content_accepted_ms_end": end,
        "words": [
            {"text": t, "start_accepted_ms": s, "end_accepted_ms": e, "probability": 0.9}
            for (t, s, e) in words
        ],
    }


def _setup_session(tmp: Path, chunks: list[tuple[int, int]], fast_rows: list[dict],
                   hq_rows: list[dict] | None = None) -> Path:
    sess = tmp / "session"
    (sess / "audio").mkdir(parents=True)
    manifest = {
        "format": "wav",
        "sample_rate": 16000,
        "chunks": [
            {
                "file": f"chunk_{i:04d}.wav",
                "format": "wav",
                "start_accepted_ms": s,
                "end_accepted_ms": e,
                "duration_ms": e - s,
            } for i, (s, e) in enumerate(chunks)
        ],
    }
    (sess / "audio" / "manifest.json").write_text(json.dumps(manifest))
    _write_jsonl(sess / "fast_segments.jsonl", fast_rows)
    if hq_rows is not None:
        _write_jsonl(sess / "hq_segments.jsonl", hq_rows)
    return sess


class TestFullyContained(unittest.TestCase):
    def test_segment_inside_one_chunk_is_not_split(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seg = _seg(1, 200, 800, [(" hello", 200, 400), (" world", 400, 800)])
            sess = _setup_session(tmp, [(0, 1000), (1000, 2000)], [seg])
            chunks = load_chunks(sess)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(len(chunks[0].segments), 1)
            s = chunks[0].segments[0]
            self.assertEqual(s.segment_id, "fast:1")
            self.assertIsNone(s.boundary)
            self.assertEqual(s.text, "hello world")
            self.assertEqual(len(chunks[1].segments), 0)


class TestBoundarySplit(unittest.TestCase):
    def test_segment_straddles_two_chunks_splits_on_word_midpoint(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Boundary at 1000ms.
            # word A midpoint=200  -> chunk 0
            # word B midpoint=600  -> chunk 0
            # word C midpoint=1100 -> chunk 1
            # word D midpoint=1500 -> chunk 1
            seg = _seg(7, 100, 1700, [
                (" alpha", 100, 300),
                (" beta",  500, 700),
                (" gamma", 1000, 1200),
                (" delta", 1400, 1600),
            ])
            sess = _setup_session(tmp, [(0, 1000), (1000, 2000)], [seg])
            chunks = load_chunks(sess)
            self.assertEqual(len(chunks[0].segments), 1)
            self.assertEqual(len(chunks[1].segments), 1)
            left, right = chunks[0].segments[0], chunks[1].segments[0]
            self.assertEqual(left.segment_id, "fast:7:p0")
            self.assertEqual(right.segment_id, "fast:7:p1")
            self.assertEqual(left.boundary, "trailing")
            self.assertEqual(right.boundary, "leading")
            self.assertEqual(left.text, "alpha beta")
            self.assertEqual(right.text, "gamma delta")

    def test_segment_spanning_three_chunks_has_middle_piece(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seg = _seg(9, 100, 2900, [
                (" a", 100, 200),     # mid=150 -> chunk 0
                (" b", 1100, 1200),   # mid=1150 -> chunk 1
                (" c", 2100, 2200),   # mid=2150 -> chunk 2
            ])
            sess = _setup_session(
                tmp, [(0, 1000), (1000, 2000), (2000, 3000)], [seg],
            )
            chunks = load_chunks(sess)
            self.assertEqual([len(c.segments) for c in chunks], [1, 1, 1])
            self.assertEqual(chunks[0].segments[0].boundary, "trailing")
            self.assertEqual(chunks[1].segments[0].boundary, "middle")
            self.assertEqual(chunks[2].segments[0].boundary, "leading")
            self.assertEqual(chunks[1].segments[0].segment_id, "fast:9:p1")


class TestOutOfRangeDropped(unittest.TestCase):
    def test_segment_past_last_chunk_end_is_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Two chunks ending at 600. A segment at [700, 900] has no
            # audio backing; the loader must drop it.
            seg = _seg(1, 700, 900, [(" orphan", 700, 900)])
            sess = _setup_session(tmp, [(0, 300), (300, 600)], [seg])
            chunks = load_chunks(sess)
            self.assertEqual([len(c.segments) for c in chunks], [0, 0])

    def test_partial_overhang_drops_outside_words(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # Last chunk ends at 600. Segment with one in-range word and
            # one past-the-end word: only the in-range word survives.
            seg = _seg(2, 500, 700, [
                (" inside", 500, 580),    # mid=540 inside chunk
                (" outside", 600, 700),   # mid=650 outside
            ])
            sess = _setup_session(tmp, [(0, 300), (300, 600)], [seg])
            chunks = load_chunks(sess)
            self.assertEqual(len(chunks[0].segments), 0)
            self.assertEqual(len(chunks[1].segments), 1)
            kept = chunks[1].segments[0]
            self.assertEqual(kept.text, "inside")
            # Some words were dropped — flag for review.
            self.assertEqual(kept.boundary, "trailing")


class TestMissingManifest(unittest.TestCase):
    def test_raises_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_chunks(Path(td))

    def test_raises_when_manifest_empty(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sess = tmp / "s"
            (sess / "audio").mkdir(parents=True)
            (sess / "audio" / "manifest.json").write_text(
                '{"format":"wav","sample_rate":16000,"chunks":[]}'
            )
            with self.assertRaises(FileNotFoundError):
                load_chunks(sess)


class TestHqPreferredMerge(unittest.TestCase):
    def test_hq_wins_when_overlapping_fast(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fast = _seg(1, 100, 400, [(" hi", 100, 400)])
            hq = _seg(2, 100, 400, [(" hello", 100, 400)])
            hq["topic"] = "segments.hq"
            sess = _setup_session(tmp, [(0, 1000)], [fast], [hq])
            chunks = load_chunks(sess)
            self.assertEqual(len(chunks[0].segments), 1)
            self.assertEqual(chunks[0].segments[0].stream, "hq")
            self.assertEqual(chunks[0].segments[0].text, "hello")

    def test_fast_kept_when_no_hq_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fast_a = _seg(1, 100, 400, [(" alpha", 100, 400)])
            fast_b = _seg(2, 500, 800, [(" beta", 500, 800)])
            hq = _seg(3, 500, 800, [(" beta", 500, 800)])
            hq["topic"] = "segments.hq"
            sess = _setup_session(tmp, [(0, 1000)], [fast_a, fast_b], [hq])
            chunks = load_chunks(sess)
            streams = [(s.stream, s.text) for s in chunks[0].segments]
            self.assertEqual(streams, [("fast", "alpha"), ("hq", "beta")])


if __name__ == "__main__":
    unittest.main()
