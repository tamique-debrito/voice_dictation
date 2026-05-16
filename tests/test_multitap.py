"""BareMultiTap detector — double-tap vs triple-tap disambiguation."""

from __future__ import annotations

import time
import unittest

from voice_dictation.hotkeys import BareMultiTap


def _build(disambiguation_delay: float = 0.05):
    doubles: list[str] = []
    triples: list[str] = []
    cleanups: list[tuple[str, int]] = []
    mt = BareMultiTap(
        window=1.0,
        keys={"r", "a"},
        on_double_tap=doubles.append,
        on_triple_tap=triples.append,
        on_visual_cleanup=lambda c, n: cleanups.append((c, n)),
        disambiguation_delay=disambiguation_delay,
        quiet_before=0.0,  # disable typing-guard for these tests
    )
    return mt, doubles, triples, cleanups


class TestMultiTap(unittest.TestCase):
    def test_double_tap_fires_after_disambiguation_delay(self):
        mt, doubles, triples, cleanups = _build(disambiguation_delay=0.05)
        mt.feed("r")
        mt.feed("r")
        # Cleanup runs synchronously.
        self.assertEqual(cleanups, [("r", 2)])
        # Double-tap callback is deferred.
        self.assertEqual(doubles, [])
        time.sleep(0.15)
        self.assertEqual(doubles, ["r"])
        self.assertEqual(triples, [])

    def test_triple_tap_cancels_pending_double(self):
        mt, doubles, triples, cleanups = _build(disambiguation_delay=0.2)
        mt.feed("r")
        mt.feed("r")
        mt.feed("r")
        # One extra cleanup for the 3rd tap.
        self.assertEqual(cleanups, [("r", 2), ("r", 1)])
        # Triple-tap callback is synchronous.
        self.assertEqual(triples, ["r"])
        # The deferred double-tap must not fire.
        time.sleep(0.3)
        self.assertEqual(doubles, [])

    def test_cross_key_resets_sequence(self):
        mt, doubles, triples, _cleanups = _build(disambiguation_delay=0.05)
        mt.feed("r")
        mt.feed("a")  # different tracked key — resets r's count
        mt.feed("r")
        time.sleep(0.15)
        # Only the second r is a single press → no double-tap.
        self.assertEqual(doubles, [])
        self.assertEqual(triples, [])

    def test_untracked_typing_clears_sequence(self):
        mt, doubles, _triples, _cleanups = _build(disambiguation_delay=0.05)
        mt.feed("r")
        mt.feed("z")  # untracked → clears history
        mt.feed("r")
        time.sleep(0.15)
        self.assertEqual(doubles, [])

    def test_two_separate_double_taps_both_fire(self):
        mt, doubles, _triples, _cleanups = _build(disambiguation_delay=0.05)
        mt.feed("r")
        mt.feed("r")
        time.sleep(0.15)
        mt.feed("r")
        mt.feed("r")
        time.sleep(0.15)
        self.assertEqual(doubles, ["r", "r"])


if __name__ == "__main__":
    unittest.main()
