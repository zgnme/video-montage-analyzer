from jsonschema import validate

from openscenesense.models import (
    AnalysisMetadata,
    AnalysisResult,
    FrameAnalysis,
    ModelsUsed,
    SummaryResult,
    analysis_result_schema,
)


def test_structured_and_legacy_contracts_are_both_available():
    result = AnalysisResult(
        summary=SummaryResult("detailed", "brief", "timeline", "transcript"),
        frame_analyses=[FrameAnalysis(0.0, "opening", "static", selection_reason="opening")],
        audio_segments=[],
        metadata=AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=0,
            video_duration=1.0,
            scene_distribution={"static": 1},
            models_used=ModelsUsed("vision", "summary", None, "test"),
            successful_frame_analyses=1,
        ),
    )
    structured = result.to_dict()
    legacy = result.to_legacy_dict()

    validate(structured, analysis_result_schema())
    assert structured["schema_version"] == "1.2"
    assert structured["metadata"]["models_used"]["provider"] == "test"
    assert legacy["summary"] == "detailed"
    assert legacy["timeline"] == "timeline"
