"""Regenerate a v2-format session from one or more existing audio files.

Each input audio file produces one session directory under ``--out-dir``
(default: ``voice_dictation/regenerated_sessions/``). A ``regenerated.json``
provenance marker is written alongside the usual JSONL streams so these
sessions can be told apart from live-recorded ones.

Usage::

    python -m voice_dictation.tools.regenerate_session \
        --out-dir voice_dictation/regenerated_sessions \
        path/to/audio1.wav path/to/audio2.wav ...

Audio files must be 16 kHz mono int16 WAV. Multi-chunk v1 sessions (e.g.
``voice_dictation/transcripts/<ts>/audio/chunk_*.wav``) can be passed via
``--concat`` which concatenates all WAVs in a directory in sort order
before feeding.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..app import App
from ..runtime_config import load_runtime_config


logger = logging.getLogger(__name__)


def _concat_dir_to_wav(dir_path: Path, out_path: Path) -> Path:
    """Concatenate all WAVs in ``dir_path`` (sorted) into ``out_path``."""
    wavs = sorted(dir_path.glob("*.wav"))
    if not wavs:
        raise ValueError(f"no WAVs in {dir_path}")
    with wave.open(str(wavs[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for w in wavs[1:]:
        with wave.open(str(w), "rb") as wf:
            if wf.getparams()[:3] != params[:3]:
                raise ValueError(
                    f"param mismatch: {w} vs {wavs[0]} "
                    f"({wf.getparams()} vs {params})"
                )
            frames.append(wf.readframes(wf.getnframes()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
    return out_path


def _drain(app: App, max_seconds: float = 7200.0) -> None:
    """Drive the pipeline to completion offline:

      1. Wait for the preprocessor's raw input queue to empty (audio fed).
      2. Force-flush both windowers so trailing audio reaches the transcribers.
      3. Wait for transcribers' in-queues to empty AND ``windows_processed``
         to stop advancing.
    """
    transcribers = [app.fast_transcriber]
    if app.hq_transcriber is not None:
        transcribers.append(app.hq_transcriber)
    windowers = [app.fast_windower]
    if app.hq_windower is not None:
        windowers.append(app.hq_windower)

    start = time.monotonic()
    # Step 1: raw audio drained from preprocessor input.
    while app.preprocessor.raw_input_q.qsize() > 0:
        if time.monotonic() - start > max_seconds:
            logger.warning("drain timeout waiting for raw_input_q")
            return
        time.sleep(0.1)
    # A short settle so the final raw chunk reaches the windowers.
    time.sleep(0.5)

    # Step 2: force-flush windowers so trailing windows hit the transcribers.
    for w in windowers:
        w._flush("regenerate_drain")  # type: ignore[attr-defined]

    # Step 3: wait until transcribers drain and stop advancing.
    last_progress = time.monotonic()
    last_counts = tuple(t.windows_processed for t in transcribers)
    # HQ inference can run for tens of seconds on long windows; keep this
    # idle threshold above typical per-window inference time.
    idle_seconds = 30.0
    while True:
        counts = tuple(t.windows_processed for t in transcribers)
        in_q_sizes = [t._in_q.qsize() for t in transcribers]
        if counts != last_counts:
            last_counts = counts
            last_progress = time.monotonic()
        idle_for = time.monotonic() - last_progress
        if all(s == 0 for s in in_q_sizes) and idle_for >= idle_seconds:
            logger.info(
                "drained: windows_processed=%s idle=%.1fs",
                counts, idle_for,
            )
            return
        if time.monotonic() - start > max_seconds:
            logger.warning(
                "drain timeout (counts=%s, in_q=%s)", counts, in_q_sizes,
            )
            return
        time.sleep(0.5)


def _regenerate_one(audio_path: Path, out_dir: Path, source_label: str) -> Path:
    cfg = load_runtime_config()
    # Override sessions_dir so this run lands under out_dir.
    cfg.app.sessions_dir = str(out_dir)

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    # pid suffix so parallel workers don't collide on same-second stamps.
    session_name = f"regen_{stamp}_p{os.getpid()}_{audio_path.stem}"
    session_dir = out_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.info("regenerating: %s -> %s", audio_path, session_dir)
    app = App(cfg, str(session_dir))
    t0 = time.monotonic()
    try:
        app.start_offline(with_widget=False)
        app.feed_wav(str(audio_path))
        _drain(app)
    finally:
        app.shutdown()
    elapsed = time.monotonic() - t0
    logger.info("done in %.1fs", elapsed)

    provenance = {
        "source_audio": str(audio_path.resolve()),
        "source_label": source_label,
        "regenerated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }
    (session_dir / "regenerated.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    return session_dir


def _worker(audio_path_str: str, out_dir_str: str, source_label: str,
            log_level: str, ct2_threads: int) -> dict:
    """Subprocess entry: re-init logging + thread caps, then regenerate one."""
    if ct2_threads > 0:
        # CT2 reads these to limit per-process thread count, which lets
        # multiple workers actually parallelise rather than thrashing.
        os.environ["OMP_NUM_THREADS"] = str(ct2_threads)
        os.environ["MKL_NUM_THREADS"] = str(ct2_threads)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=f"%(asctime)s [pid %(process)d] %(levelname)s %(name)s | %(message)s",
        force=True,
    )
    try:
        session_dir = _regenerate_one(Path(audio_path_str), Path(out_dir_str),
                                      source_label)
        return {"ok": True, "audio": audio_path_str,
                "session_dir": str(session_dir)}
    except Exception as e:
        logger.exception("worker failed for %s", audio_path_str)
        return {"ok": False, "audio": audio_path_str, "error": str(e)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "audio", nargs="+", type=Path,
        help="audio file(s) (16 kHz mono int16 WAV) or directories with --concat",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "regenerated_sessions",
    )
    p.add_argument(
        "--concat", action="store_true",
        help="treat each input as a directory; concatenate its WAVs first",
    )
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--parallel", type=int, default=1,
                   help="run N audio files concurrently in worker processes")
    p.add_argument("--ct2-threads", type=int, default=0,
                   help="per-worker CT2/OMP thread cap (0 = leave default); "
                        "useful with --parallel > 1 to avoid thread thrash")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve all inputs to (audio_path, label) up front. Concat happens in
    # the parent so workers see ready-to-feed WAVs.
    work: list[tuple[Path, str]] = []
    for input_path in args.audio:
        if args.concat:
            if not input_path.is_dir():
                print(f"error: --concat requires directories: {input_path}",
                      file=sys.stderr)
                return 2
            scratch = args.out_dir / "_concat_scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            # Use parent.name so concat targets are unique when every source
            # dir is named "audio" (the v1 transcript layout).
            concat_tag = f"{input_path.parent.name}_{input_path.name}"
            concat_wav = scratch / f"{concat_tag}.wav"
            _concat_dir_to_wav(input_path, concat_wav)
            work.append((concat_wav, f"concat:{input_path}"))
        else:
            if not input_path.exists():
                print(f"error: missing: {input_path}", file=sys.stderr)
                return 2
            work.append((input_path, str(input_path)))

    results: list[dict] = []
    if args.parallel <= 1:
        for audio_path, label in work:
            session_dir = _regenerate_one(audio_path, args.out_dir, label)
            results.append({"ok": True, "audio": str(audio_path),
                            "session_dir": str(session_dir)})
    else:
        ctx = mp.get_context("spawn")  # avoid fork pitfalls w/ threads + CT2
        with cf.ProcessPoolExecutor(
            max_workers=args.parallel, mp_context=ctx,
        ) as ex:
            futs = {
                ex.submit(_worker, str(ap), str(args.out_dir), lbl,
                          args.log_level, args.ct2_threads): (ap, lbl)
                for ap, lbl in work
            }
            for fut in cf.as_completed(futs):
                results.append(fut.result())

    print(json.dumps(results, indent=2))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
