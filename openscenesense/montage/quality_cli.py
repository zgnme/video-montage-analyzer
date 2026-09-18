from __future__ import annotations

import argparse
import math
import signal
import sys
from pathlib import Path

from .cli import load_transcript, lock
from .media import run
from .provider import DEFAULT_CONFIG, load_config
from .quality_plan import prepare
from .quality_report import render
from .quality_runner import Runner


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="video-montage precision",
        description="All source frames, parallel analysis and independent boundary checks",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--core-frames", type=int, default=48)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=960)
    parser.add_argument("--threshold", type=float, default=27)
    parser.add_argument(
        "--source-offset",
        type=float,
        default=0,
        help="Seconds to add to input presentation times; never resamples frames",
    )
    parser.add_argument("--max-duration", type=float, default=1800)
    parser.add_argument("--max-frames-total", type=int, default=120000)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=300,
        help="Cumulative across resumes in this output directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20000000,
        help="Cumulative known tokens plus conservative reservations; not a provider billing cap",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model")
    parser.add_argument("--reasoning", choices=["none", "low", "high", "max"])
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument(
        "--transcript",
        type=Path,
        help="Timed transcript on the output/source clock, including --source-offset",
    )
    speech.add_argument(
        "--transcribe",
        action="store_true",
        help="Local Whisper; adjusts clip-relative timestamps by --source-offset",
    )
    args = parser.parse_args(argv)
    if not (
        1 <= args.workers <= 10
        and 4 <= args.core_frames <= 60
        and 1 <= args.context_frames <= 16
        and args.core_frames + 2 * args.context_frames <= 64
        and 160 <= args.frame_size <= 2048
        and math.isfinite(args.source_offset)
        and args.source_offset >= 0
        and math.isfinite(args.threshold)
        and 0 < args.threshold <= 255
        and math.isfinite(args.max_duration)
        and args.max_duration > 0
        and args.max_calls > 0
        and args.max_tokens > 0
        and args.max_frames_total > 0
    ):
        parser.error("Invalid limits; core frames plus both context margins must not exceed 64")
    args.output = args.output.expanduser().resolve()
    try:
        with lock(args.output):
            plan = prepare(args)
            print(
                f"Prepared {len(plan['frames'])} source frames, "
                f"{len(plan['windows'])} windows, "
                f"{len(plan['candidates'])} boundary candidates",
                flush=True,
            )
            if args.prepare_only:
                return 0
            config = load_config(args.config)
            if args.model:
                config["MONTAGE_MODEL"] = args.model
            if args.reasoning:
                config["MONTAGE_REASONING_EFFORT"] = args.reasoning
            if args.transcribe:
                if not plan["metadata"]["has_audio"]:
                    raise ValueError("--transcribe requested but input has no audio")
                python, script = config.get("MONTAGE_ASR_PYTHON"), config.get("MONTAGE_ASR_SCRIPT")
                if not python or not script:
                    raise ValueError("Local ASR is not configured")
                args.transcript = args.output / "precision-transcript.txt"
                if not args.transcript.exists():
                    temp = args.output / "precision-transcript.pending.txt"
                    run([python, script, plan["source"], "-o", str(temp)], timeout=14400)
                    temp.replace(args.transcript)
            transcript = load_transcript(args.transcript)
            if args.transcribe:
                transcript = [
                    {
                        **s,
                        "start": s["start"] + args.source_offset,
                        "end": s["end"] + args.source_offset,
                    }
                    for s in transcript
                ]
            runner = Runner(args, plan, config, transcript)
            previous = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, lambda *_: runner.stop.set())
            try:
                result = runner.analyze()
                render(plan, result, args.output)
            finally:
                runner.store.close()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            print(
                f"{result['status']}; boundaries {result['quality_status']}; "
                f"{args.output / 'report.html'}",
                flush=True,
            )
            return (
                1
                if result["status"] != "complete"
                else (2 if result["quality_status"] == "needs_review" else 0)
            )
    except Exception as exc:
        print(f"video-montage precision: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
