from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .analyzer import VideoAnalyzer
from .diagnostics import check_environment
from .frame_selectors import DynamicFrameSelector, UniformFrameSelector
from .models import AnalysisPrompts, ModelConfig, analysis_result_schema
from .openrouter_analyzer import OpenRouterAnalyzer
from .progress import ProgressEvent


def _load_prompts(path: str | None) -> AnalysisPrompts | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Prompts file must contain a JSON object.")
    return AnalysisPrompts(**payload)


def _atomic_json_write(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(value, temporary, indent=2, ensure_ascii=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _progress(event: ProgressEvent) -> None:
    suffix = f" ({event.current}/{event.total})" if event.total else ""
    print(f"[{event.stage}]{suffix} {event.message}".rstrip(), file=sys.stderr)


def _selector(args: argparse.Namespace):
    if args.frame_selector == "uniform":
        return UniformFrameSelector()
    return DynamicFrameSelector(
        scene_change_threshold=args.scene_change_threshold,
        scene_scan_fps=args.scene_scan_fps,
        min_scene_gap=args.min_scene_gap,
    )


def _add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video_path")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--vision-model", default="gpt-4o")
    parser.add_argument("--summary-model", default="gpt-4o-mini")
    parser.add_argument("--audio-model", default="whisper-1")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--frame-selector", choices=["dynamic", "uniform"], default="dynamic")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--frames-per-minute", type=float, default=4.0)
    parser.add_argument("--scene-change-threshold", type=float, default=0.18)
    parser.add_argument("--scene-scan-fps", type=float, default=2.0)
    parser.add_argument("--min-scene-gap", type=float, default=0.75)
    parser.add_argument("--max-image-dimension", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-frame-failure-ratio", type=float, default=0.25)
    parser.add_argument(
        "--api-mode",
        choices=["responses", "chat_completions", "auto"],
        default="responses",
    )
    parser.add_argument("--prompts-file")
    parser.add_argument("--cache-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--quiet", action="store_true")


def _analyze(args: argparse.Namespace) -> int:
    model_config = ModelConfig(
        vision_model=args.vision_model,
        text_model=args.summary_model,
        audio_model=args.audio_model,
    )
    common: dict[str, Any] = {
        "model_config": model_config,
        "frame_selector": _selector(args),
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "frames_per_minute": args.frames_per_minute,
        "prompts": _load_prompts(args.prompts_file),
        "max_workers": args.max_workers,
        "enable_audio": not args.no_audio,
        "strict": args.strict,
        "max_frame_failure_ratio": args.max_frame_failure_ratio,
        "on_progress": None if args.quiet else _progress,
        "max_image_dimension": args.max_image_dimension,
        "jpeg_quality": args.jpeg_quality,
        "timeout": args.timeout,
        "cache_dir": args.cache_dir,
        "resume": args.resume and not args.force,
    }
    if args.provider == "openrouter":
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is required for the OpenRouter provider.")
        analyzer = OpenRouterAnalyzer(
            openrouter_key=openrouter_key,
            openai_key=os.environ.get("OPENAI_API_KEY"),
            **common,
        )
    else:
        analyzer = VideoAnalyzer(
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_mode=args.api_mode,
            **common,
        )
    result = analyzer.analyze_video_structured(args.video_path)
    payload = result.to_dict() if args.structured_output else result.to_legacy_dict()
    if args.output:
        _atomic_json_write(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze videos with OpenAI or OpenRouter.")
    parser.add_argument("--schema", action="store_true", help="Print the v1.2 result schema")
    parser.add_argument("--check", action="store_true", help="Check local FFmpeg requirements")
    subparsers = parser.add_subparsers(dest="command")
    analyze = subparsers.add_parser("analyze", help="Analyze a video")
    _add_analyze_arguments(analyze)
    subparsers.add_parser("schema", help="Print the v1.2 structured-result schema")
    subparsers.add_parser("check", help="Check local FFmpeg requirements")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        if args.schema or args.command == "schema":
            print(json.dumps(analysis_result_schema(), indent=2, ensure_ascii=True))
            return 0
        if args.check or args.command == "check":
            report = check_environment()
            print(json.dumps(report, indent=2, ensure_ascii=True))
            return 0 if report["ok"] else 1
        if args.command is None:
            parser.error("choose 'analyze', 'schema', or 'check'")
        return _analyze(args)
    except Exception as exc:
        print(f"openscenesense: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
