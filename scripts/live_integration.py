"""Opt-in, low-cost provider integration using a generated two-scene video."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from jsonschema import validate

from openscenesense import (
    ModelConfig,
    OpenRouterAnalyzer,
    UniformFrameSelector,
    VideoAnalyzer,
    analysis_result_schema,
)


def _fixture(directory: str) -> Path:
    path = Path(directory) / "integration.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        2.0,
        (96, 64),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the integration fixture.")
    for color in ((0, 0, 0), (0, 0, 0), (255, 255, 255), (255, 255, 255)):
        writer.write(np.full((64, 96, 3), color, dtype=np.uint8))
    writer.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=["openai", "openrouter"])
    args = parser.parse_args()
    if args.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required.")
        models = ModelConfig(
            vision_model=os.environ.get("OPENAI_VISION_MODEL") or "gpt-5.6-luna",
            text_model=os.environ.get("OPENAI_SUMMARY_MODEL") or "gpt-5.6-luna",
        )
        analyzer = VideoAnalyzer(
            api_key=api_key,
            model_config=models,
            frame_selector=UniformFrameSelector(),
            min_frames=2,
            max_frames=2,
            frames_per_minute=60,
            max_workers=1,
            enable_audio=False,
            strict=True,
        )
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        vision_model = os.environ.get("OPENROUTER_VISION_MODEL")
        summary_model = os.environ.get("OPENROUTER_SUMMARY_MODEL")
        if not api_key or not vision_model or not summary_model:
            raise SystemExit(
                "OPENROUTER_API_KEY, OPENROUTER_VISION_MODEL, and "
                "OPENROUTER_SUMMARY_MODEL are required."
            )
        analyzer = OpenRouterAnalyzer(
            openrouter_key=api_key,
            openai_key=None,
            model_config=ModelConfig(
                vision_model=vision_model,
                text_model=summary_model,
            ),
            frame_selector=UniformFrameSelector(),
            min_frames=2,
            max_frames=2,
            frames_per_minute=60,
            max_workers=1,
            enable_audio=False,
            strict=True,
        )

    with tempfile.TemporaryDirectory(prefix="openscenesense-integration-") as directory:
        result = analyzer.analyze_video_structured(str(_fixture(directory)))
    validate(result.to_dict(), analysis_result_schema())
    if len(result.frame_analyses) != 2 or not result.summary.brief:
        raise SystemExit("Provider integration returned an incomplete result.")
    print(
        f"{args.provider} integration OK: frames={len(result.frame_analyses)}, "
        f"tokens={result.metadata.usage.total_tokens}"
    )


if __name__ == "__main__":
    main()
