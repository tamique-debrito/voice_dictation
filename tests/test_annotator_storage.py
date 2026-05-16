"""AnnotationStore: append + last-write-wins resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from voice_dictation.annotator.storage import AnnotationStore


class TestAnnotationStore(unittest.TestCase):
    def test_segment_last_write_wins(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            store.append_segment(
                chunk_idx=0, segment_id="fast:1", text="first",
                status="edited",
            )
            store.append_segment(
                chunk_idx=0, segment_id="fast:1", text="second",
                status="edited",
            )
            state = store.resolved_state()
            self.assertEqual(state["segments"]["0|fast:1"]["text"], "second")

    def test_chunk_reject_then_unreject(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            store.append_chunk(chunk_idx=3, status="rejected")
            self.assertEqual(store.resolved_state()["rejected_chunks"], [3])
            store.append_chunk(chunk_idx=3, status="unrejected")
            self.assertEqual(store.resolved_state()["rejected_chunks"], [])

    def test_invalid_status_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            with self.assertRaises(ValueError):
                store.append_segment(
                    chunk_idx=0, segment_id="fast:1", text="x", status="bogus",
                )
            with self.assertRaises(ValueError):
                store.append_chunk(chunk_idx=0, status="bogus")

    def test_accepted_hq_round_trips_with_source(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            store.append_segment(
                chunk_idx=1, segment_id="hq:9", text="hello",
                status="accepted_hq",
                source_text_fast="hi",
                source_text_hq="hello",
            )
            row = store.resolved_state()["segments"]["1|hq:9"]
            self.assertEqual(row["status"], "accepted_hq")
            self.assertEqual(row["source_text_fast"], "hi")
            self.assertEqual(row["source_text_hq"], "hello")

    def test_segment_save_in_rejected_chunk_keeps_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            store.append_chunk(chunk_idx=0, status="rejected")
            store.append_segment(
                chunk_idx=0, segment_id="fast:1", text="x", status="edited",
            )
            state = store.resolved_state()
            self.assertIn(0, state["rejected_chunks"])
            self.assertEqual(state["segments"]["0|fast:1"]["text"], "x")

    def test_empty_dir_returns_empty_state(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            state = store.resolved_state()
            self.assertEqual(state, {"segments": {}, "rejected_chunks": []})

    def test_path_property(self):
        with tempfile.TemporaryDirectory() as td:
            store = AnnotationStore(td)
            self.assertEqual(store.path, Path(td) / "annotations.jsonl")


if __name__ == "__main__":
    unittest.main()
