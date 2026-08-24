# Examples

Run the scripts from the repository root after installing the development environment.

OpenAI:

```bash
export OPENAI_API_KEY=...
python Examples/OpenAIDemo.py path/to/video.mp4 --no-audio
```

OpenRouter model availability changes, so select the two model IDs explicitly:

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_VISION_MODEL=...
export OPENROUTER_SUMMARY_MODEL=...
python Examples/OpenrouterDemo.py path/to/video.mp4
```

Both examples use `analyze_video_structured()` and a bounded 4–12 frame budget. API keys are read
only from environment variables.
