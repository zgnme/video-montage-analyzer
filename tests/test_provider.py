import json
from types import SimpleNamespace

from openscenesense.models import Frame, SceneType
from openscenesense.providers.openai_provider import OpenAIProvider


class Recorder:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.outputs.pop(0),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        )


def test_responses_provider_uses_image_schema_and_usage():
    recorder = Recorder(
        [
            json.dumps(
                {
                    "description": "A test frame",
                    "objects": ["square"],
                    "actions": [],
                    "visible_text": [],
                    "tags": ["test"],
                }
            )
        ]
    )
    client = SimpleNamespace(responses=recorder)
    provider = OpenAIProvider(client, "vision", "summary")
    frame = Frame(None, 1.0, SceneType.STATIC, selection_reason="opening")

    analysis, usage = provider.analyze_frame(frame, "encoded", "describe")

    assert analysis.description == "A test frame"
    assert usage.total_tokens == 15
    call = recorder.calls[0]
    assert call["store"] is False
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["input"][0]["content"][1]["type"] == "input_image"


def test_summary_is_generated_in_one_structured_call():
    recorder = Recorder([json.dumps({"detailed": "long", "brief": "short", "events": []})])
    provider = OpenAIProvider(SimpleNamespace(responses=recorder), "vision", "summary")

    summary, events, usage = provider.generate_summary("prompt", "timeline", "transcript")

    assert summary.detailed == "long"
    assert summary.brief == "short"
    assert events == []
    assert usage.total_tokens == 15
    assert len(recorder.calls) == 1
