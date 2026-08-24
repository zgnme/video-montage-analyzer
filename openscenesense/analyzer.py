from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Literal

from openai import OpenAI
from PIL import Image

from .cache import AnalysisCache, analysis_cache_key
from .exceptions import (
    AudioTranscriptionError,
    AuthenticationError,
    ConfigurationError,
    FrameAnalysisError,
    ModelNotFoundError,
    OpenSceneSenseError,
    RateLimitError,
    ResponseValidationError,
    VideoAnalysisError,
    VideoAnalyzerError,
    VideoLoadError,
)
from .frame_selectors import DynamicFrameSelector, FrameSelector
from .models import (
    AnalysisMetadata,
    AnalysisPrompts,
    AnalysisResult,
    AudioSegment,
    Frame,
    FrameAnalysis,
    ModelConfig,
    ModelsUsed,
    PerformanceMetadata,
    SceneType,
    SelectionMetadata,
    SummaryResult,
    TimelineEvent,
    UsageMetadata,
)
from .progress import ProgressEvent
from .providers import OpenAIProvider, ProviderAdapter
from .providers.base import merge_usage
from .transcriber import AudioTranscriber, NoAudioTranscriber, OpenAITranscriber
from .video_utils import probe_video

ProgressCallback = Callable[[ProgressEvent], None]


class VideoAnalyzer:
    def __init__(
        self,
        api_key: str | None = None,
        model_config: ModelConfig | None = None,
        frame_selector: FrameSelector | None = None,
        min_frames: int = 8,
        max_frames: int = 32,
        frames_per_minute: float = 4.0,
        prompts: AnalysisPrompts | None = None,
        log_level: int = logging.INFO,
        base_url: str | None = None,
        organization: str | None = None,
        max_workers: int = 5,
        *,
        audio_transcriber: AudioTranscriber | None = None,
        enable_audio: bool = True,
        api_mode: Literal["responses", "chat_completions", "auto"] = "responses",
        strict: bool = False,
        max_frame_failure_ratio: float = 0.25,
        on_progress: ProgressCallback | None = None,
        max_image_dimension: int = 1280,
        jpeg_quality: int = 85,
        timeout: float = 120.0,
        default_headers: dict[str, str] | None = None,
        cache_dir: str | None = None,
        resume: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._validate_initialization_parameters(
            min_frames,
            max_frames,
            frames_per_minute,
            max_workers,
            max_frame_failure_ratio,
            max_image_dimension,
            jpeg_quality,
            timeout,
        )
        self.model_config = model_config or ModelConfig()
        self.frame_selector = frame_selector or DynamicFrameSelector(logger=self.logger)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.prompts = prompts or AnalysisPrompts()
        self.max_workers = max_workers
        self.strict = strict
        self.max_frame_failure_ratio = max_frame_failure_ratio
        self.on_progress = on_progress
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        self.cache_dir = cache_dir
        self.resume = resume
        self.client = self._initialize_openai_client(
            api_key, base_url, organization, timeout, default_headers
        )
        self.provider: ProviderAdapter = OpenAIProvider(
            self.client,
            self.model_config.vision_model,
            self.model_config.text_model,
            api_mode=api_mode,
        )
        self.enable_audio = bool(enable_audio)
        if audio_transcriber is not None:
            self.audio_transcriber: AudioTranscriber = audio_transcriber
        elif self.enable_audio:
            self.audio_transcriber = OpenAITranscriber(
                self.client, model=self.model_config.audio_model
            )
        else:
            self.audio_transcriber = NoAudioTranscriber()
        self._usage = UsageMetadata()
        self._usage_lock = threading.Lock()

    @staticmethod
    def _validate_initialization_parameters(
        min_frames: int,
        max_frames: int,
        frames_per_minute: float,
        max_workers: int,
        max_frame_failure_ratio: float,
        max_image_dimension: int,
        jpeg_quality: int,
        timeout: float,
    ) -> None:
        if min_frames < 1:
            raise ConfigurationError("min_frames must be at least 1.")
        if max_frames < min_frames:
            raise ConfigurationError("max_frames must be greater than or equal to min_frames.")
        if frames_per_minute <= 0:
            raise ConfigurationError("frames_per_minute must be positive.")
        if max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1.")
        if not 0 <= max_frame_failure_ratio <= 1:
            raise ConfigurationError("max_frame_failure_ratio must be between 0 and 1.")
        if max_image_dimension < 64:
            raise ConfigurationError("max_image_dimension must be at least 64.")
        if not 1 <= jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be between 1 and 100.")
        if timeout <= 0:
            raise ConfigurationError("timeout must be positive.")

    def _initialize_openai_client(
        self,
        api_key: str | None,
        base_url: str | None,
        organization: str | None,
        timeout: float,
        default_headers: dict[str, str] | None,
    ) -> OpenAI:
        kwargs: dict[str, object] = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if default_headers:
            kwargs["default_headers"] = default_headers
        try:
            return OpenAI(**kwargs)
        except Exception as exc:
            raise ConfigurationError(f"OpenAI client initialization failed: {exc}") from exc

    def _emit(
        self,
        stage: str,
        current: int = 0,
        total: int = 0,
        message: str = "",
        **details: object,
    ) -> None:
        if not self.on_progress:
            return
        try:
            self.on_progress(ProgressEvent(stage, current, total, message, dict(details)))
        except Exception as exc:
            self.logger.warning("Progress callback failed: %s", exc)

    def _record_usage(self, usage: UsageMetadata) -> None:
        with self._usage_lock:
            merge_usage(self._usage, usage)

    def _frame_to_base64(self, frame: object) -> str:
        try:
            image = Image.fromarray(frame)
            image.thumbnail(
                (self.max_image_dimension, self.max_image_dimension),
                Image.Resampling.LANCZOS,
            )
            output = io.BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=self.jpeg_quality,
                optimize=True,
            )
            return base64.b64encode(output.getvalue()).decode("ascii")
        except Exception as exc:
            raise ResponseValidationError(f"Frame preprocessing failed: {exc}") from exc

    def _analyze_frame(self, frame: Frame) -> FrameAnalysis:
        if frame.image is None:
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description="Error analyzing frame",
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
                difference_score=frame.difference_score,
                error="Frame image data is missing.",
            )
        try:
            analysis, usage = self.provider.analyze_frame(
                frame,
                self._frame_to_base64(frame.image),
                self.prompts.frame_analysis,
            )
            self._record_usage(usage)
            return analysis
        except (AuthenticationError, ModelNotFoundError, RateLimitError):
            raise
        except Exception as exc:
            if self.strict:
                raise
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description="Error analyzing frame",
                scene_type=frame.scene_type.value,
                selection_reason=frame.selection_reason,
                difference_score=frame.difference_score,
                error=str(exc),
            )

    def _concurrently_analyze_frames(
        self, frames: list[Frame], cache: AnalysisCache | None = None
    ) -> list[FrameAnalysis]:
        results: list[FrameAnalysis] = []
        pending: list[tuple[int, Frame]] = []
        for index, frame in enumerate(frames):
            cached = (
                cache.read_json(f"frame_analyses/{index:04d}.json")
                if cache and self.resume
                else None
            )
            if isinstance(cached, dict):
                try:
                    results.append(FrameAnalysis(**cached))
                    self._emit(
                        "analyzing_frames",
                        len(results),
                        len(frames),
                        f"Reused cached frame at {frame.timestamp:.2f}s",
                    )
                    continue
                except (TypeError, ValueError):
                    pass
            pending.append((index, frame))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_frame, frame): (index, frame)
                for index, frame in pending
            }
            completed = len(results)
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                index, _frame = futures[future]
                if cache:
                    cache.write_json(f"frame_analyses/{index:04d}.json", asdict(result))
                completed += 1
                self._emit(
                    "analyzing_frames",
                    completed,
                    len(frames),
                    f"Analyzed frame at {result.timestamp:.2f}s",
                )
        return sorted(results, key=lambda item: item.timestamp)

    def _prepare_cache(self, video_path: str) -> AnalysisCache | None:
        if not self.cache_dir:
            return None
        configuration = {
            "schema_version": "1.2",
            "provider": self.provider.provider_name,
            "provider_adapter": self.provider.__class__.__name__,
            "models": asdict(self.model_config),
            "prompts": asdict(self.prompts),
            "frame_selector": self.frame_selector.__class__.__name__,
            "selection": {
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
                "frames_per_minute": self.frames_per_minute,
                **self.frame_selector.selection_parameters(),
            },
            "image": {
                "max_image_dimension": self.max_image_dimension,
                "jpeg_quality": self.jpeg_quality,
            },
            "audio": {
                "enabled": self.enable_audio,
                "transcriber": self.audio_transcriber.__class__.__name__,
            },
        }
        key = analysis_cache_key(video_path, configuration)
        cache = AnalysisCache(self.cache_dir, key)
        cache.write_json("manifest.json", {"key": key, "configuration": configuration})
        return cache

    @staticmethod
    def _format_timeline(frame_descriptions: list[FrameAnalysis]) -> str:
        return "\n".join(
            f"Time {item.timestamp:.2f}s ({item.scene_type}; {item.selection_reason}): "
            f"{item.description}"
            for item in frame_descriptions
        )

    @staticmethod
    def _format_transcript(audio_segments: list[AudioSegment]) -> str:
        if not audio_segments:
            return "No audio transcript available."
        return "\n".join(
            f"[{item.start_time:.1f}s - {item.end_time:.1f}s]: {item.text}"
            for item in audio_segments
        )

    def _summary_prompt(self, timeline: str, transcript: str, duration: float) -> str:
        detailed = self.prompts.detailed_summary.format(
            duration=duration, timeline=timeline, transcript=transcript
        )
        brief = self.prompts.brief_summary.format(
            duration=duration, timeline=timeline, transcript=transcript
        )
        return (
            "Return one structured result containing a detailed summary, a brief summary, and "
            "chronological events grounded in the supplied frame timestamps. Do not invent "
            "events that are not supported by the inputs.\n\n"
            f"Detailed-summary requirements:\n{detailed}\n\n"
            f"Brief-summary requirements:\n{brief}"
        )

    @staticmethod
    def _fallback_summary(
        frame_descriptions: list[FrameAnalysis], timeline: str, transcript: str
    ) -> tuple[SummaryResult, list[TimelineEvent]]:
        successful = [item for item in frame_descriptions if not item.error]
        descriptions = [item.description for item in successful]
        detailed = " ".join(descriptions) or "No frame analyses were completed successfully."
        brief = " ".join(descriptions[:2]) or detailed
        events = [
            TimelineEvent(
                start_time=item.timestamp,
                end_time=item.timestamp,
                description=item.description,
                source_frame_timestamps=[item.timestamp],
                objects=list(item.objects),
                actions=list(item.actions),
                visible_text=list(item.visible_text),
            )
            for item in successful
        ]
        return SummaryResult(detailed, brief, timeline, transcript), events

    def _generate_summary(
        self,
        frame_descriptions: list[FrameAnalysis],
        audio_segments: list[AudioSegment],
        video_duration: float,
    ) -> tuple[SummaryResult, list[TimelineEvent], str | None]:
        timeline = self._format_timeline(frame_descriptions)
        transcript = self._format_transcript(audio_segments)
        try:
            summary, events, usage = self.provider.generate_summary(
                self._summary_prompt(timeline, transcript, video_duration),
                timeline,
                transcript,
            )
            self._record_usage(usage)
            return summary, events, None
        except (AuthenticationError, ModelNotFoundError, RateLimitError):
            raise
        except Exception as exc:
            if self.strict:
                raise
            summary, events = self._fallback_summary(frame_descriptions, timeline, transcript)
            return summary, events, f"Summary generation failed; deterministic fallback used: {exc}"

    def _select_and_log_frames(self, video_path: str) -> list[Frame]:
        frames = self.frame_selector.select_frames(
            video_path,
            self.min_frames,
            self.max_frames,
            self.frames_per_minute,
        )
        if not frames:
            raise VideoLoadError("No decodable frames were selected.")
        return frames

    def analyze_video_structured(self, video_path: str) -> AnalysisResult:
        if not os.path.isfile(video_path):
            raise VideoLoadError(f"Video file does not exist: {video_path}")
        self._usage = UsageMetadata()
        started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        warnings: list[str] = []
        cache = self._prepare_cache(video_path)

        self._emit("probing", message="Probing video metadata")
        stage_started = time.perf_counter()
        metadata = probe_video(video_path)
        stage_seconds["probing"] = time.perf_counter() - stage_started

        self._emit("selecting_frames", message="Selecting key frames")
        stage_started = time.perf_counter()
        frames = self._select_and_log_frames(video_path)
        stage_seconds["selection"] = time.perf_counter() - stage_started
        selector_metadata = getattr(self.frame_selector, "last_metadata", metadata)
        if selector_metadata.duration > 0:
            metadata = selector_metadata
        if cache:
            cache.write_json("metadata.json", asdict(metadata))
            cache.write_json(
                "selected_frames.json",
                [
                    {
                        "timestamp": frame.timestamp,
                        "scene_type": frame.scene_type.value,
                        "selection_reason": frame.selection_reason,
                        "difference_score": frame.difference_score,
                    }
                    for frame in frames
                ],
            )

        audio_segments: list[AudioSegment] = []
        if self.enable_audio:
            self._emit("extracting_audio", message="Extracting and transcribing audio")
            stage_started = time.perf_counter()
            try:
                cached_transcript = (
                    cache.read_json("transcript.json") if cache and self.resume else None
                )
                if isinstance(cached_transcript, list):
                    audio_segments = [AudioSegment(**item) for item in cached_transcript]
                else:
                    self._emit("transcribing", message="Transcribing audio")
                    audio_segments = self.audio_transcriber.transcribe(video_path)
                    if cache:
                        cache.write_json(
                            "transcript.json", [asdict(item) for item in audio_segments]
                        )
            except Exception as exc:
                if self.strict:
                    raise
                warnings.append(f"Audio transcription failed: {exc}")
            stage_seconds["transcription"] = time.perf_counter() - stage_started

        self._emit("analyzing_frames", 0, len(frames), "Analyzing selected frames")
        stage_started = time.perf_counter()
        frame_descriptions = self._concurrently_analyze_frames(frames, cache=cache)
        stage_seconds["frame_analysis"] = time.perf_counter() - stage_started
        failed = sum(bool(item.error) for item in frame_descriptions)
        if failed:
            warnings.append(f"{failed} frame analyses failed.")
        if frame_descriptions and failed / len(frame_descriptions) > self.max_frame_failure_ratio:
            raise ResponseValidationError(
                f"Frame failure ratio {failed / len(frame_descriptions):.1%} exceeded "
                f"the configured {self.max_frame_failure_ratio:.1%}."
            )

        self._emit("summarizing", message="Generating structured summary")
        stage_started = time.perf_counter()
        summary, timeline, summary_warning = self._generate_summary(
            frame_descriptions, audio_segments, metadata.duration
        )
        stage_seconds["summary"] = time.perf_counter() - stage_started
        if summary_warning:
            warnings.append(summary_warning)

        scene_distribution = {
            scene_type.value: sum(frame.scene_type == scene_type for frame in frames)
            for scene_type in SceneType
        }
        selection = getattr(
            self.frame_selector,
            "last_selection_metadata",
            SelectionMetadata(
                strategy=self.frame_selector.__class__.__name__,
                selected_frame_count=len(frames),
            ),
        )
        result = AnalysisResult(
            summary=summary,
            timeline=timeline,
            frame_analyses=frame_descriptions,
            audio_segments=audio_segments,
            metadata=AnalysisMetadata(
                num_frames_analyzed=len(frames),
                num_audio_segments=len(audio_segments),
                video_duration=metadata.duration,
                scene_distribution=scene_distribution,
                models_used=ModelsUsed(
                    frame_analysis=self.model_config.vision_model,
                    summary=self.model_config.text_model,
                    audio=(self.model_config.audio_model if self.enable_audio else None),
                    provider=self.provider.provider_name,
                ),
                video=metadata,
                selection=selection,
                performance=PerformanceMetadata(
                    total_seconds=time.perf_counter() - started,
                    stage_seconds=stage_seconds,
                ),
                usage=self._usage,
                successful_frame_analyses=len(frame_descriptions) - failed,
                failed_frame_analyses=failed,
            ),
            warnings=warnings,
            errors=[item.error for item in frame_descriptions if item.error],
        )
        self._emit("complete", len(frames), len(frames), "Video analysis complete")
        if cache:
            cache.write_json("result.json", result.to_dict())
        return result

    def analyze_video(self, video_path: str) -> dict[str, object]:
        """Analyze a video and return the backward-compatible dictionary shape."""
        return self.analyze_video_structured(video_path).to_legacy_dict()

    def _compile_results(
        self,
        frames: list[Frame],
        frame_descriptions: list[dict[str, object] | FrameAnalysis],
        audio_segments: list[AudioSegment],
        summaries: dict[str, object],
        video_duration: float,
    ) -> dict[str, object]:
        """Compatibility helper retained for callers that subclassed v1.1."""
        normalized = [
            item
            if isinstance(item, FrameAnalysis)
            else FrameAnalysis(
                timestamp=float(item.get("timestamp", 0)),
                description=str(item.get("description", "")),
                scene_type=str(item.get("scene_type", SceneType.STATIC.value)),
                selection_reason=str(item.get("selection_reason", "uniform_fill")),
                error=item.get("error") if isinstance(item.get("error"), str) else None,
            )
            for item in frame_descriptions
        ]
        timeline_text = str(summaries.get("timeline", self._format_timeline(normalized)))
        transcript = str(summaries.get("transcript", self._format_transcript(audio_segments)))
        result = AnalysisResult(
            summary=SummaryResult(
                detailed=str(summaries.get("detailed", "")),
                brief=str(summaries.get("brief", "")),
                timeline=timeline_text,
                transcript=transcript,
            ),
            frame_analyses=normalized,
            audio_segments=audio_segments,
            metadata=AnalysisMetadata(
                num_frames_analyzed=len(frames),
                num_audio_segments=len(audio_segments),
                video_duration=video_duration,
                scene_distribution={
                    scene_type.value: sum(frame.scene_type == scene_type for frame in frames)
                    for scene_type in SceneType
                },
                models_used=ModelsUsed(
                    self.model_config.vision_model,
                    self.model_config.text_model,
                    self.model_config.audio_model,
                    self.provider.provider_name,
                ),
            ),
        )
        return result.to_legacy_dict()


# These imports historically came from openscenesense.analyzer.
__all__ = [
    "AudioTranscriptionError",
    "FrameAnalysisError",
    "VideoAnalyzer",
    "VideoAnalyzerError",
    "VideoAnalysisError",
    "OpenSceneSenseError",
]
