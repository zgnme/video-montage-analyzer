from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from ..models import Frame, FrameAnalysis, SummaryResult, TimelineEvent, UsageMetadata

FRAME_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["description", "objects", "actions", "visible_text", "tags"],
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "objects": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}


SUMMARY_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["detailed", "brief", "events"],
    "properties": {
        "detailed": {"type": "string", "minLength": 1},
        "brief": {"type": "string", "minLength": 1},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "start_time",
                    "end_time",
                    "description",
                    "source_frame_timestamps",
                    "objects",
                    "actions",
                    "visible_text",
                ],
                "properties": {
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "description": {"type": "string", "minLength": 1},
                    "source_frame_timestamps": {"type": "array", "items": {"type": "number"}},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "actions": {"type": "array", "items": {"type": "string"}},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Provider response must be a JSON object.")
    return parsed


def usage_from_response(response: Any) -> UsageMetadata:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageMetadata()
    input_tokens = int(
        getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None) or 0
    )
    total_tokens = int(getattr(usage, "total_tokens", None) or input_tokens + output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def merge_usage(target: UsageMetadata, incoming: UsageMetadata) -> None:
    target.input_tokens += incoming.input_tokens
    target.output_tokens += incoming.output_tokens
    target.total_tokens += incoming.total_tokens
    target.provider_details.update(incoming.provider_details)


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


def frame_analysis_from_payload(frame: Frame, payload: dict[str, Any]) -> FrameAnalysis:
    return FrameAnalysis(
        timestamp=frame.timestamp,
        description=_required_text(payload, "description"),
        scene_type=frame.scene_type.value,
        selection_reason=frame.selection_reason,
        difference_score=frame.difference_score,
        objects=_required_string_list(payload, "objects"),
        actions=_required_string_list(payload, "actions"),
        visible_text=_required_string_list(payload, "visible_text"),
        tags=_required_string_list(payload, "tags"),
    )


def summary_from_payload(
    payload: dict[str, Any], timeline_text: str, transcript: str
) -> tuple[SummaryResult, list[TimelineEvent]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("events must be an array")
    events = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise ValueError("each event must be an object")
        if not isinstance(item.get("start_time"), (int, float)) or not isinstance(
            item.get("end_time"), (int, float)
        ):
            raise ValueError("event start_time and end_time must be numbers")
        source_timestamps = item.get("source_frame_timestamps")
        if not isinstance(source_timestamps, list) or any(
            not isinstance(value, (int, float)) for value in source_timestamps
        ):
            raise ValueError("source_frame_timestamps must be an array of numbers")
        events.append(
            TimelineEvent(
                start_time=max(0.0, float(item["start_time"])),
                end_time=max(0.0, float(item["end_time"])),
                description=_required_text(item, "description"),
                source_frame_timestamps=[max(0.0, float(value)) for value in source_timestamps],
                objects=_required_string_list(item, "objects"),
                actions=_required_string_list(item, "actions"),
                visible_text=_required_string_list(item, "visible_text"),
            )
        )
    return (
        SummaryResult(
            detailed=_required_text(payload, "detailed"),
            brief=_required_text(payload, "brief"),
            timeline=timeline_text,
            transcript=transcript,
        ),
        events,
    )


class ProviderAdapter(ABC):
    provider_name: str

    @abstractmethod
    def analyze_frame(
        self, frame: Frame, base64_image: str, prompt: str
    ) -> tuple[FrameAnalysis, UsageMetadata]:
        raise NotImplementedError

    @abstractmethod
    def generate_summary(
        self,
        prompt: str,
        timeline_text: str,
        transcript: str,
    ) -> tuple[SummaryResult, list[TimelineEvent], UsageMetadata]:
        raise NotImplementedError
