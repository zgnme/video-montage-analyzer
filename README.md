# OpenSceneSense

Turn video into structured, timestamped scene intelligence without installing a local ML stack.

OpenSceneSense samples the most useful frames, transcribes optional audio, analyzes visual content with OpenAI or OpenRouter, and returns summaries, events, metadata, timing, and usage information. It is designed for applications, batch pipelines, dataset tooling, and anything that needs a dependable JSON result rather than a demo-only paragraph.

> Looking for private, local inference? Use [OpenSceneSense Ollama](https://github.com/ymrohit/openscenesense-ollama). The two packages share the v1.2 result contract but intentionally do not share a dependency graph.

## Why v1.2

- Modern OpenAI Responses API support with image input and strict JSON Schema output.
- OpenRouter support through an explicit Chat Completions adapter.
- One structured summary request produces the detailed summary, brief summary, and events.
- Budget-bounded scene selection: scene density changes which frames win, never the configured cost ceiling.
- Reduced-rate scene scanning on downscaled frames instead of analyzing every decoded frame.
- JPEG preprocessing with a configurable size and quality ceiling.
- Optional, replaceable audio transcription with no local audio-decoding stack.
- Typed results, a checked-in JSON Schema, legacy dictionary compatibility, usage telemetry, progress events, strict mode, and resumable stage caches.
- No import-time FFmpeg process, console output, or global logging configuration.

## Install

OpenSceneSense supports Python 3.10+ and requires the FFmpeg and FFprobe executables.

```bash
pip install openscenesense
```

Install FFmpeg with your platform package manager:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

Verify the local requirement:

```bash
openscenesense check
```

The default package remains API-focused. It does not install Torch, Transformers, Ollama, librosa, or soundfile.

## Python quick start

Set `OPENAI_API_KEY`, then choose model IDs explicitly so deployments do not depend on changing aliases or README defaults:

```python
from openscenesense import ModelConfig, VideoAnalyzer

analyzer = VideoAnalyzer(
    model_config=ModelConfig(
        vision_model="gpt-5.6-luna",
        text_model="gpt-5.6-luna",
        audio_model="whisper-1",
    ),
    min_frames=8,
    max_frames=32,
    frames_per_minute=4,
)

result = analyzer.analyze_video_structured("video.mp4")

print(result.summary.brief)
for event in result.timeline:
    print(event.start_time, event.description)
```

`analyze_video_structured()` is the preferred v1.2 API. Existing code can continue using `analyze_video()`, which returns the v1.1 dictionary shape with additive metadata:

```python
legacy = analyzer.analyze_video("video.mp4")
print(legacy["brief_summary"])
print(legacy["frame_analyses"])
```

## CLI quick start

API keys are read from environment variables and are never accepted as CLI arguments.

```bash
export OPENAI_API_KEY="..."

openscenesense analyze video.mp4 \
  --provider openai \
  --vision-model gpt-5.6-luna \
  --summary-model gpt-5.6-luna \
  --structured-output \
  --output result.json
```

Useful controls:

```text
--provider openai|openrouter
--frame-selector dynamic|uniform
--min-frames / --max-frames / --frames-per-minute
--scene-change-threshold / --scene-scan-fps / --min-scene-gap
--max-image-dimension / --jpeg-quality
--no-audio
--api-mode responses|chat_completions|auto
--timeout / --max-workers
--strict / --max-frame-failure-ratio
--cache-dir / --resume / --force
--structured-output
```

Print the result schema with:

```bash
openscenesense schema
```

## OpenRouter

OpenRouter handles frame and summary inference; OpenAI audio transcription remains independent and can be disabled or replaced.

```python
import os

from openscenesense import ModelConfig, OpenRouterAnalyzer

analyzer = OpenRouterAnalyzer(
    openrouter_key=os.environ["OPENROUTER_API_KEY"],
    openai_key=os.environ.get("OPENAI_API_KEY"),
    model_config=ModelConfig(
        vision_model="your-vision-capable-openrouter-model",
        text_model="your-structured-output-model",
        audio_model="whisper-1",
    ),
    enable_audio=False,
)

result = analyzer.analyze_video_structured("video.mp4")
```

OpenAI defaults to the Responses API. OpenRouter defaults to Chat Completions. `api_mode="auto"` only falls back when an endpoint is genuinely unsupported; authentication, model, rate-limit, and malformed-response failures are never retried through a second endpoint.

## Data boundary

Metadata probing and frame selection happen locally. The package sends only the selected,
size-bounded JPEG frames to the chosen vision provider; it does not upload the original video as
one file. When built-in audio is enabled, a temporary mono WAV is sent to OpenAI transcription.
The resulting frame descriptions and transcript are sent to the configured summary provider.

Disable audio or supply your own transcriber when that boundary is too broad. Provider retention
and training policies remain the provider's responsibility, so review them for sensitive
workloads. Opt-in caches stay on the machine running OpenSceneSense.

## Structured result contract

Both OpenSceneSense distributions emit schema version `1.2`:

```text
AnalysisResult
├── schema_version
├── summary
│   ├── detailed
│   └── brief
├── timeline[]
├── frame_analyses[]
├── audio_segments[]
├── metadata
│   ├── video
│   ├── selection
│   ├── models
│   ├── performance
│   └── usage
├── warnings[]
└── errors[]
```

The source schema is [Docs/analysis_result.schema.json](Docs/analysis_result.schema.json). Regenerate it after contract changes:

```bash
python scripts/export_schema.py
```

## Frame selection

`DynamicFrameSelector` builds one predictable frame budget:

```python
from openscenesense import DynamicFrameSelector, VideoAnalyzer

selector = DynamicFrameSelector(
    scene_change_threshold=0.18,
    scene_scan_fps=2.0,
    min_scene_gap=0.75,
)

analyzer = VideoAnalyzer(
    frame_selector=selector,
    min_frames=8,
    max_frames=32,
    frames_per_minute=4,
)
```

The opening and closing frames are retained, strong scene-change peaks receive up to 60% of the remaining budget, and unused positions fill the largest temporal gaps. Selected frames record `selection_reason` and a normalized `difference_score`.

Use `UniformFrameSelector` when deterministic spacing is more important than scene changes.

## Audio is replaceable

Disable audio without changing the visual pipeline:

```python
analyzer = VideoAnalyzer(enable_audio=False)
```

Or provide an object with `transcribe(video_path) -> list[AudioSegment]`:

```python
import os

from openscenesense import AudioSegment, VideoAnalyzer


class ExistingTranscript:
    def transcribe(self, video_path):
        return [AudioSegment("Already transcribed", 0.0, 2.0, 1.0)]


analyzer = VideoAnalyzer(
    audio_transcriber=ExistingTranscript(),
    api_key=os.environ["OPENAI_API_KEY"],
)
```

The built-in OpenAI transcriber extracts a temporary mono 16 kHz WAV through FFmpeg. Extraction failures become warnings in normal mode and exceptions in strict mode.

## Reliability controls

```python
def progress(event):
    print(event.stage, event.current, event.total, event.message)


analyzer = VideoAnalyzer(
    strict=False,
    max_frame_failure_ratio=0.25,
    on_progress=progress,
    timeout=120,
    max_workers=5,
)
```

Authentication, missing-model, and rate-limit failures stop immediately. In normal mode, isolated frame, transcription, or summary-validation failures are returned as warnings with deterministic fallbacks. `strict=True` converts partial failures into exceptions.

## Cache and resume

Caching is opt-in because results can contain sensitive descriptions and transcripts.

```python
analyzer = VideoAnalyzer(
    cache_dir=".openscenesense-cache",
    resume=True,
)
```

The key incorporates the video edge hash, size, modification nanoseconds, provider, models, prompts, selection settings, preprocessing settings, and transcription configuration. Manifests, metadata, transcripts, frame analyses, and final results are written atomically in separate stages.

## Development

```bash
git clone https://github.com/ymrohit/openscenesense.git
cd openscenesense
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check openscenesense tests scripts benchmarks Examples
python -m build
```

CI tests minimum-supported and latest-compatible dependencies across supported Python versions.
The bounded live integration under `scripts/` runs only through manual dispatch with provider
credentials.

More detail:

- [Result schema](Docs/result-schema.md)
- [Frame selection](Docs/frame-selection.md)
- [Transcription](Docs/transcription.md)
- [Performance](Docs/performance.md)
- [Troubleshooting](Docs/troubleshooting.md)
- [v1.2 migration](Docs/v1.2-migration.md)

## License and support

OpenSceneSense is released under the MIT License. Report bugs and request features through [GitHub Issues](https://github.com/ymrohit/openscenesense/issues).
