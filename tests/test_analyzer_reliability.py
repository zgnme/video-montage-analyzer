import base64
import io

import numpy as np
import pytest
from PIL import Image

from openscenesense import UniformFrameSelector, VideoAnalyzer
from openscenesense.exceptions import ConfigurationError, ResponseValidationError
from openscenesense.models import FrameAnalysis, SummaryResult, UsageMetadata


class FailingFrameProvider:
    provider_name = "failing"

    def analyze_frame(self, *_args):
        raise RuntimeError("frame provider failed")

    def generate_summary(self, _prompt, timeline, transcript):
        return SummaryResult("fallback input", "brief", timeline, transcript), [], UsageMetadata()


class SummaryFailProvider:
    provider_name = "summary-failing"

    def analyze_frame(self, frame, *_args):
        return (
            FrameAnalysis(
                timestamp=frame.timestamp,
                description="frame",
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
            ),
            UsageMetadata(),
        )

    def generate_summary(self, *_args):
        raise RuntimeError("summary provider failed")


class BadTranscriber:
    def transcribe(self, _video_path):
        raise RuntimeError("audio failed")


def _analyzer(provider, **kwargs):
    analyzer = VideoAnalyzer(
        api_key="test",
        enable_audio=False,
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=2,
        **kwargs,
    )
    analyzer.provider = provider
    return analyzer


def test_partial_frame_failures_warn_when_allowed(synthetic_video):
    result = _analyzer(
        FailingFrameProvider(), max_frame_failure_ratio=1.0
    ).analyze_video_structured(str(synthetic_video))

    assert result.metadata.failed_frame_analyses == 2
    assert result.errors == ["frame provider failed", "frame provider failed"]
    assert any("frame analyses failed" in warning for warning in result.warnings)


def test_failure_ratio_and_strict_mode_abort(synthetic_video):
    with pytest.raises(ResponseValidationError, match="failure ratio"):
        _analyzer(FailingFrameProvider()).analyze_video_structured(str(synthetic_video))

    with pytest.raises(RuntimeError, match="frame provider failed"):
        _analyzer(FailingFrameProvider(), strict=True).analyze_video_structured(
            str(synthetic_video)
        )


def test_summary_failure_uses_deterministic_fallback(synthetic_video):
    result = _analyzer(SummaryFailProvider()).analyze_video_structured(str(synthetic_video))

    assert result.summary.detailed == "frame frame"
    assert len(result.timeline) == 2
    assert any("deterministic fallback" in warning for warning in result.warnings)


def test_audio_failure_warns_or_raises_in_strict_mode(synthetic_video):
    analyzer = VideoAnalyzer(
        api_key="test",
        audio_transcriber=BadTranscriber(),
        frame_selector=UniformFrameSelector(),
        min_frames=2,
        max_frames=2,
        frames_per_minute=2,
    )
    analyzer.provider = SummaryFailProvider()
    result = analyzer.analyze_video_structured(str(synthetic_video))
    assert any("Audio transcription failed" in warning for warning in result.warnings)

    analyzer.strict = True
    with pytest.raises(RuntimeError, match="audio failed"):
        analyzer.analyze_video_structured(str(synthetic_video))


def test_progress_stages_are_ordered(synthetic_video):
    events = []
    _analyzer(SummaryFailProvider(), on_progress=events.append).analyze_video_structured(
        str(synthetic_video)
    )
    stages = [event.stage for event in events]

    assert stages[0:2] == ["probing", "selecting_frames"]
    assert stages.index("analyzing_frames") < stages.index("summarizing")
    assert stages[-1] == "complete"


def test_image_preprocessing_resizes_and_uses_jpeg():
    analyzer = _analyzer(SummaryFailProvider(), max_image_dimension=320, jpeg_quality=70)
    encoded = analyzer._frame_to_base64(np.zeros((2000, 1000, 3), dtype=np.uint8))
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))

    assert image.format == "JPEG"
    assert max(image.size) == 320
    assert image.size == (160, 320)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_frames": 0},
        {"min_frames": 2, "max_frames": 1},
        {"frames_per_minute": 0},
        {"max_workers": 0},
        {"max_frame_failure_ratio": 1.1},
        {"max_image_dimension": 63},
        {"jpeg_quality": 101},
        {"timeout": 0},
    ],
)
def test_invalid_analyzer_configuration_is_rejected(kwargs):
    base = {"api_key": "test", "enable_audio": False}
    with pytest.raises(ConfigurationError):
        VideoAnalyzer(**base, **kwargs)
