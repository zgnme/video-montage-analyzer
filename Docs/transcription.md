# Transcription

Cloud transcription is optional and replaceable.

- `enable_audio=True` uses `OpenAITranscriber` by default.
- `enable_audio=False` performs no extraction or transcription.
- `audio_transcriber=<object>` accepts any object implementing
  `transcribe(video_path) -> list[AudioSegment]`.

The built-in transcriber asks FFmpeg for a temporary mono 16 kHz WAV and uploads it to the configured
OpenAI audio model. The library does not require librosa or soundfile and does not retain the
temporary file.

In normal mode, missing audio or extraction/transcription failures produce warnings and visual
analysis continues. In strict mode they raise. `OpenRouterAnalyzer` uses OpenRouter only for visual
and summary calls; built-in audio uses a separate OpenAI client, while a custom transcriber needs no
OpenAI audio credential.

Treat transcripts and caches as potentially sensitive application data.
