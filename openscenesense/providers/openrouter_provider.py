from __future__ import annotations

from typing import Any

from openai import OpenAI

from ..exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderConnectionError,
    RateLimitError,
    ResponseValidationError,
)
from ..models import Frame, FrameAnalysis, SummaryResult, TimelineEvent, UsageMetadata
from .base import (
    FRAME_OUTPUT_SCHEMA,
    SUMMARY_OUTPUT_SCHEMA,
    ProviderAdapter,
    frame_analysis_from_payload,
    parse_json_object,
    summary_from_payload,
    usage_from_response,
)


class OpenRouterProvider(ProviderAdapter):
    provider_name = "openrouter"

    def __init__(self, client: OpenAI, vision_model: str, summary_model: str) -> None:
        self.client = client
        self.vision_model = vision_model
        self.summary_model = summary_model

    @staticmethod
    def _raise_classified(exc: Exception) -> None:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            raise AuthenticationError(str(exc)) from exc
        if status == 429:
            raise RateLimitError(str(exc)) from exc
        if status == 404:
            raise ModelNotFoundError(str(exc)) from exc
        raise ProviderConnectionError(str(exc)) from exc

    def _chat(
        self,
        model: str,
        content: Any,
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            )
        except Exception as exc:
            self._raise_classified(exc)
            raise AssertionError("unreachable") from exc
        if not response.choices or not response.choices[0].message.content:
            raise ResponseValidationError("OpenRouter returned no completion content.")
        return response.choices[0].message.content, response

    def analyze_frame(
        self, frame: Frame, base64_image: str, prompt: str
    ) -> tuple[FrameAnalysis, UsageMetadata]:
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
        output, response = self._chat(
            self.vision_model, content, "frame_analysis", FRAME_OUTPUT_SCHEMA
        )
        try:
            analysis = frame_analysis_from_payload(frame, parse_json_object(output))
            if not analysis.description:
                raise ValueError("description is empty")
            return analysis, usage_from_response(response)
        except (TypeError, ValueError) as exc:
            raise ResponseValidationError(f"Invalid frame-analysis response: {exc}") from exc

    def generate_summary(
        self, prompt: str, timeline_text: str, transcript: str
    ) -> tuple[SummaryResult, list[TimelineEvent], UsageMetadata]:
        output, response = self._chat(
            self.summary_model, prompt, "video_summary", SUMMARY_OUTPUT_SCHEMA
        )
        try:
            payload = parse_json_object(output)
            summary, events = summary_from_payload(payload, timeline_text, transcript)
            if not summary.detailed or not summary.brief:
                raise ValueError("summary fields must not be empty")
            return summary, events, usage_from_response(response)
        except (TypeError, ValueError) as exc:
            repair_prompt = (
                "Repair this invalid video-summary output as JSON matching the required schema:\n"
                + output
            )
            repaired, repair_response = self._chat(
                self.summary_model, repair_prompt, "video_summary", SUMMARY_OUTPUT_SCHEMA
            )
            try:
                summary, events = summary_from_payload(
                    parse_json_object(repaired), timeline_text, transcript
                )
                usage = usage_from_response(response)
                extra = usage_from_response(repair_response)
                usage.input_tokens += extra.input_tokens
                usage.output_tokens += extra.output_tokens
                usage.total_tokens += extra.total_tokens
                return summary, events, usage
            except (TypeError, ValueError) as repair_exc:
                raise ResponseValidationError(
                    f"Invalid summary response after repair: {repair_exc}"
                ) from exc
