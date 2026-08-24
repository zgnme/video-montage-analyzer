"""Minimal structured OpenAI example."""

from __future__ import annotations

import argparse
import os

from openscenesense import ModelConfig, VideoAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    analyzer = VideoAnalyzer(
        api_key=api_key,
        model_config=ModelConfig(
            vision_model=os.environ.get("OPENAI_VISION_MODEL") or "gpt-5.6-luna",
            text_model=os.environ.get("OPENAI_SUMMARY_MODEL") or "gpt-5.6-luna",
            audio_model=os.environ.get("OPENAI_AUDIO_MODEL") or "whisper-1",
        ),
        min_frames=4,
        max_frames=12,
        frames_per_minute=4,
        enable_audio=not args.no_audio,
    )
    result = analyzer.analyze_video_structured(args.video_path)
    print(result.summary.brief)
    for event in result.timeline:
        print(f"{event.start_time:7.2f}s  {event.description}")


if __name__ == "__main__":
    main()
