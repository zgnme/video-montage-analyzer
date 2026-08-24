from __future__ import annotations

from typing import Any, Literal

from openai import OpenAI

from ..exceptions import (
    AuthenticationError,
    ConfigurationError,
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


class OpenAIProvider(ProviderAdapter):
    provider_name = "openai"

    def __init__(
        self,
        client: OpenAI,
        vision_model: str,
        summary_model: str,
        api_mode: Literal["responses", "chat_completions", "auto"] = "responses",
    ) -> None:
        if api_mode not in {"responses", "chat_completions", "auto"}:
            raise ConfigurationError(f"Unsupported api_mode: {api_mode}")
        self.client = client
        self.vision_model = vision_model
        self.summary_model = summary_model
        self.api_mode = api_mode

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

    @staticmethod
    def _endpoint_unsupported(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        return status in {405, 501} or code in {"unsupported_endpoint", "not_implemented"}

    @staticmethod
    def _text_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        }

    def _responses(
        self,
        model: str,
        content: str | list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[str, Any]:
        response_input: Any = content
        if isinstance(content, list):
            response_input = [{"role": "user", "content": content}]
        response = self.client.responses.create(
            model=model,
            input=response_input,
            text=self._text_format(schema_name, schema),
            store=False,
        )
        output = getattr(response, "output_text", None)
        if not output:
            raise ResponseValidationError("OpenAI returned no output text.")
        return output, response

    def _chat(
        self,
        model: str,
        content: str | list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[str, Any]:
        message_content: Any = content
        if isinstance(content, list):
            message_content = []
            for item in content:
                if item.get("type") == "input_text":
                    message_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "input_image":
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": item["image_url"]}}
                    )
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        )
        if not response.choices or not response.choices[0].message.content:
            raise ResponseValidationError("OpenAI returned no chat completion content.")
        return response.choices[0].message.content, response

    def _request(
        self,
        model: str,
        content: str | list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
    ) -> tuple[str, Any]:
        try:
            if self.api_mode == "chat_completions":
                return self._chat(model, content, schema_name, schema)
            try:
                return self._responses(model, content, schema_name, schema)
            except Exception as exc:
                if self.api_mode == "auto" and self._endpoint_unsupported(exc):
                    return self._chat(model, content, schema_name, schema)
                raise
        except ResponseValidationError:
            raise
        except Exception as exc:
            self._raise_classified(exc)
            raise AssertionError("unreachable") from exc

    def analyze_frame(
        self, frame: Frame, base64_image: str, prompt: str
    ) -> tuple[FrameAnalysis, UsageMetadata]:
        content = [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high",
            },
        ]
        output, response = self._request(
            self.vision_model, content, "frame_analysis", FRAME_OUTPUT_SCHEMA
        )
        try:
            payload = parse_json_object(output)
            analysis = frame_analysis_from_payload(frame, payload)
            if not analysis.description:
                raise ValueError("description is empty")
            return analysis, usage_from_response(response)
        except (TypeError, ValueError) as exc:
            raise ResponseValidationError(f"Invalid frame-analysis response: {exc}") from exc

    def generate_summary(
        self, prompt: str, timeline_text: str, transcript: str
    ) -> tuple[SummaryResult, list[TimelineEvent], UsageMetadata]:
        output, response = self._request(
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
                "Repair the following invalid video-summary output. Return only data matching "
                f"the supplied JSON schema.\n\nInvalid output:\n{output}"
            )
            repaired, repair_response = self._request(
                self.summary_model, repair_prompt, "video_summary", SUMMARY_OUTPUT_SCHEMA
            )
            try:
                payload = parse_json_object(repaired)
                summary, events = summary_from_payload(payload, timeline_text, transcript)
                usage = usage_from_response(response)
                repaired_usage = usage_from_response(repair_response)
                usage.input_tokens += repaired_usage.input_tokens
                usage.output_tokens += repaired_usage.output_tokens
                usage.total_tokens += repaired_usage.total_tokens
                return summary, events, usage
            except (TypeError, ValueError) as repair_exc:
                raise ResponseValidationError(
                    f"Invalid summary response after repair: {repair_exc}"
                ) from exc
