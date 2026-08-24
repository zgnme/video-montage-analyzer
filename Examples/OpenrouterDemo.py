"""Minimal structured OpenRouter example with audio disabled."""

from __future__ import annotations

import argparse
import os

from openscenesense import ModelConfig, OpenRouterAnalyzer


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} before running this example.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    args = parser.parse_args()

    analyzer = OpenRouterAnalyzer(
        openrouter_key=_required("OPENROUTER_API_KEY"),
        openai_key=None,
        model_config=ModelConfig(
            vision_model=_required("OPENROUTER_VISION_MODEL"),
            text_model=_required("OPENROUTER_SUMMARY_MODEL"),
        ),
        min_frames=4,
        max_frames=12,
        frames_per_minute=4,
        enable_audio=False,
    )
    result = analyzer.analyze_video_structured(args.video_path)
    print(result.summary.brief)
    for event in result.timeline:
        print(f"{event.start_time:7.2f}s  {event.description}")


if __name__ == "__main__":
    main()
