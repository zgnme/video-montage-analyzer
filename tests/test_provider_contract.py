import json
from types import SimpleNamespace

import pytest

from openscenesense.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderConnectionError,
    RateLimitError,
    ResponseValidationError,
)
from openscenesense.models import Frame, SceneType
from openscenesense.providers.base import parse_json_object
from openscenesense.providers.openai_provider import OpenAIProvider
from openscenesense.providers.openrouter_provider import OpenRouterProvider


class StatusError(Exception):
    def __init__(self, status_code, code=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.code = code


class Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            output_text=value,
            usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5),
        )


class Chat:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value))],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


def _frame_payload(**overrides):
    payload = {
        "description": "valid",
        "objects": [],
        "actions": [],
        "visible_text": [],
        "tags": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _summary_payload(**overrides):
    payload = {"detailed": "long", "brief": "short", "events": []}
    payload.update(overrides)
    return json.dumps(payload)


def test_json_parser_accepts_fence_and_rejects_non_object():
    assert parse_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(ValueError, match="object"):
        parse_json_object("[]")


def test_auto_falls_back_only_for_unsupported_endpoint():
    responses = Responses([StatusError(405)])
    chat = Chat([_frame_payload()])
    provider = OpenAIProvider(
        SimpleNamespace(responses=responses, chat=SimpleNamespace(completions=chat)),
        "vision",
        "summary",
        api_mode="auto",
    )

    analysis, _usage = provider.analyze_frame(Frame(None, 0, SceneType.STATIC), "image", "prompt")

    assert analysis.description == "valid"
    assert len(responses.calls) == 1
    assert len(chat.calls) == 1


def test_default_mode_does_not_fallback_and_auth_is_classified():
    responses = Responses([StatusError(401)])
    chat = Chat([_frame_payload()])
    provider = OpenAIProvider(
        SimpleNamespace(responses=responses, chat=SimpleNamespace(completions=chat)),
        "vision",
        "summary",
    )

    with pytest.raises(AuthenticationError):
        provider.analyze_frame(Frame(None, 0, SceneType.STATIC), "image", "prompt")
    assert chat.calls == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [(404, ModelNotFoundError), (429, RateLimitError), (500, ProviderConnectionError)],
)
def test_provider_http_errors_are_classified(status, expected):
    provider = OpenAIProvider(
        SimpleNamespace(responses=Responses([StatusError(status)])),
        "vision",
        "summary",
    )

    with pytest.raises(expected):
        provider.analyze_frame(Frame(None, 0, SceneType.STATIC), "image", "prompt")


def test_summary_repair_is_single_retry_and_usage_is_aggregated():
    responses = Responses(["not json", _summary_payload()])
    provider = OpenAIProvider(SimpleNamespace(responses=responses), "vision", "summary")

    summary, events, usage = provider.generate_summary("prompt", "timeline", "transcript")

    assert summary.brief == "short"
    assert events == []
    assert usage.total_tokens == 10
    assert len(responses.calls) == 2


def test_repair_and_array_types_are_validated():
    responses = Responses([_summary_payload(detailed=""), _summary_payload(brief="")])
    provider = OpenAIProvider(SimpleNamespace(responses=responses), "vision", "summary")
    with pytest.raises(ResponseValidationError, match="after repair"):
        provider.generate_summary("prompt", "timeline", "transcript")

    malformed = Responses([_frame_payload(objects="not-an-array")])
    provider = OpenAIProvider(SimpleNamespace(responses=malformed), "vision", "summary")
    with pytest.raises(ResponseValidationError, match="array of strings"):
        provider.analyze_frame(Frame(None, 0, SceneType.STATIC), "image", "prompt")


def test_openrouter_uses_chat_schema_and_classifies_connection_error():
    chat = Chat([_frame_payload()])
    provider = OpenRouterProvider(
        SimpleNamespace(chat=SimpleNamespace(completions=chat)), "vision", "summary"
    )
    analysis, usage = provider.analyze_frame(Frame(None, 0, SceneType.STATIC), "image", "prompt")
    assert analysis.description == "valid"
    assert usage.total_tokens == 5
    assert chat.calls[0]["response_format"]["type"] == "json_schema"

    failing = Chat([StatusError(500)])
    provider = OpenRouterProvider(
        SimpleNamespace(chat=SimpleNamespace(completions=failing)), "vision", "summary"
    )
    with pytest.raises(ProviderConnectionError):
        provider.generate_summary("prompt", "timeline", "transcript")
