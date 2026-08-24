from __future__ import annotations

import logging
import os
import tempfile
from typing import Protocol

import ffmpeg
from openai import OpenAI

from .exceptions import AudioExtractionError, TranscriptionError
from .models import AudioSegment
from .video_utils import has_audio_stream

logger = logging.getLogger(__name__)


class AudioTranscriber(Protocol):
    def transcribe(self, video_path: str) -> list[AudioSegment]: ...


class NoAudioTranscriber:
    def transcribe(self, video_path: str) -> list[AudioSegment]:
        return []


class OpenAITranscriber:
    def __init__(self, client: OpenAI, model: str = "whisper-1") -> None:
        self.client = client
        self.model = model

    @staticmethod
    def _extract_wav(video_path: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            temporary_path = temporary.name
        try:
            (
                ffmpeg.input(video_path)
                .output(temporary_path, ac=1, ar=16000, format="wav", vn=None)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
                raise AudioExtractionError("FFmpeg produced an empty audio file.")
            return temporary_path
        except Exception as exc:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            if isinstance(exc, AudioExtractionError):
                raise
            stderr = getattr(exc, "stderr", b"")
            detail = stderr.decode(errors="replace").strip() if stderr else str(exc)
            raise AudioExtractionError(f"FFmpeg audio extraction failed: {detail}") from exc

    def transcribe(self, video_path: str) -> list[AudioSegment]:
        if not has_audio_stream(video_path):
            return []
        temporary_path = self._extract_wav(video_path)
        try:
            response_format = "verbose_json" if "whisper" in self.model.lower() else "json"
            with open(temporary_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format=response_format,
                )
            raw_segments = getattr(response, "segments", None) or []
            if raw_segments:
                return [
                    AudioSegment(
                        text=str(getattr(item, "text", "")),
                        start_time=float(getattr(item, "start", 0) or 0),
                        end_time=float(getattr(item, "end", 0) or 0),
                        confidence=float(getattr(item, "confidence", 1) or 1),
                    )
                    for item in raw_segments
                ]
            text = getattr(response, "text", None) or getattr(response, "output_text", None)
            return [AudioSegment(str(text or response), 0.0, 0.0, 1.0)]
        except Exception as exc:
            raise TranscriptionError(f"Audio transcription failed: {exc}") from exc
        finally:
            try:
                os.unlink(temporary_path)
            except OSError as exc:
                logger.warning("Failed to remove temporary audio file %s: %s", temporary_path, exc)
