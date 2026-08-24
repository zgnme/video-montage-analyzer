from __future__ import annotations

import logging

from openai import OpenAI

from .analyzer import ProgressCallback, VideoAnalyzer
from .frame_selectors import FrameSelector
from .models import AnalysisPrompts, ModelConfig
from .providers import OpenRouterProvider
from .transcriber import AudioTranscriber, NoAudioTranscriber, OpenAITranscriber


class OpenRouterAnalyzer(VideoAnalyzer):
    """Use OpenRouter for vision/summary and an independent transcriber for audio."""

    def __init__(
        self,
        openrouter_key: str,
        openai_key: str | None,
        model_config: ModelConfig | None = None,
        frame_selector: FrameSelector | None = None,
        min_frames: int = 8,
        max_frames: int = 32,
        frames_per_minute: float = 4.0,
        prompts: AnalysisPrompts | None = None,
        log_level: int = logging.INFO,
        http_referer: str | None = None,
        app_title: str | None = None,
        max_workers: int = 5,
        *,
        audio_transcriber: AudioTranscriber | None = None,
        enable_audio: bool = True,
        strict: bool = False,
        max_frame_failure_ratio: float = 0.25,
        on_progress: ProgressCallback | None = None,
        max_image_dimension: int = 1280,
        jpeg_quality: int = 85,
        timeout: float = 120.0,
        cache_dir: str | None = None,
        resume: bool = False,
    ) -> None:
        headers = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_title:
            headers["X-Title"] = app_title
        super().__init__(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            model_config=model_config,
            frame_selector=frame_selector,
            min_frames=min_frames,
            max_frames=max_frames,
            frames_per_minute=frames_per_minute,
            prompts=prompts,
            log_level=log_level,
            max_workers=max_workers,
            enable_audio=False,
            api_mode="chat_completions",
            strict=strict,
            max_frame_failure_ratio=max_frame_failure_ratio,
            on_progress=on_progress,
            max_image_dimension=max_image_dimension,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
            default_headers=headers or None,
            cache_dir=cache_dir,
            resume=resume,
        )
        self.provider = OpenRouterProvider(
            self.client,
            self.model_config.vision_model,
            self.model_config.text_model,
        )
        self.enable_audio = bool(enable_audio)
        self.audio_client: OpenAI | None = None
        if audio_transcriber is not None:
            self.audio_transcriber = audio_transcriber
        elif self.enable_audio:
            kwargs = {"timeout": timeout}
            if openai_key:
                kwargs["api_key"] = openai_key
            self.audio_client = OpenAI(**kwargs)
            self.audio_transcriber = OpenAITranscriber(
                self.audio_client, model=self.model_config.audio_model
            )
        else:
            self.audio_transcriber = NoAudioTranscriber()

    def _transcribe_audio(self, video_path: str):
        """Compatibility hook retained from v1.1."""
        return self.audio_transcriber.transcribe(video_path) if self.enable_audio else []
