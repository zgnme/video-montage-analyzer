from __future__ import annotations

from openscenesense import UniformFrameSelector, VideoAnalyzer
from openscenesense.models import (
    FrameAnalysis,
    SummaryResult,
    TimelineEvent,
    UsageMetadata,
)


class FakeProvider:
    provider_name = "fake"

    def __init__(self):
        self.frame_calls = 0
        self.summary_calls = 0

    def analyze_frame(self, frame, _base64_image, _prompt):
        self.frame_calls += 1
        return (
            FrameAnalysis(
                timestamp=frame.timestamp,
                description=f"frame {frame.timestamp:.2f}",
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
                difference_score=frame.difference_score,
            ),
            UsageMetadata(1, 2, 3),
        )

    def generate_summary(self, _prompt, timeline, transcript):
        self.summary_calls += 1
        return (
            SummaryResult("detailed", "brief", timeline, transcript),
            [TimelineEvent(0, 1, "event", [0])],
            UsageMetadata(2, 3, 5),
        )


def _analyzer(provider, **kwargs):
    analyzer = VideoAnalyzer(
        api_key="test",
        enable_audio=False,
        frame_selector=UniformFrameSelector(),
        min_frames=3,
        max_frames=3,
        frames_per_minute=3,
        max_workers=2,
        **kwargs,
    )
    analyzer.provider = provider
    return analyzer


def test_analyzer_returns_structured_and_legacy_results(synthetic_video):
    progress = []
    provider = FakeProvider()
    analyzer = _analyzer(provider, on_progress=progress.append)

    result = analyzer.analyze_video_structured(str(synthetic_video))
    legacy = result.to_legacy_dict()

    assert provider.frame_calls == 3
    assert provider.summary_calls == 1
    assert result.metadata.usage.total_tokens == 14
    assert legacy["summary"] == "detailed"
    assert legacy["metadata"]["num_frames_analyzed"] == 3
    assert progress[-1].stage == "complete"


def test_resume_reuses_completed_frame_analyses(synthetic_video, tmp_path):
    first_provider = FakeProvider()
    _analyzer(first_provider, cache_dir=str(tmp_path), resume=True).analyze_video_structured(
        str(synthetic_video)
    )
    second_provider = FakeProvider()
    _analyzer(second_provider, cache_dir=str(tmp_path), resume=True).analyze_video_structured(
        str(synthetic_video)
    )

    assert first_provider.frame_calls == 3
    assert second_provider.frame_calls == 0
    assert second_provider.summary_calls == 1
