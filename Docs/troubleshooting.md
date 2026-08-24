# Troubleshooting

Run `openscenesense check` first. It reports whether FFmpeg and FFprobe are available without
running anything during package import.

## Common failures

- **Missing video or undecodable frames:** verify the path, codec, and `ffprobe <video>`.
- **Authentication:** set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`; keys are intentionally not CLI
  flags.
- **Unsupported Responses endpoint:** select `api_mode="chat_completions"` explicitly for a
  compatible non-OpenAI endpoint, or use `auto`. Authentication, model, and rate-limit failures do
  not trigger endpoint fallback.
- **Partial frame failures:** inspect `warnings`, `errors`, and frame-level `error`; lower
  concurrency or increase timeout. The configured failure-ratio ceiling still aborts bad runs.
- **Schema failures:** keep structured output enabled and inspect the warning if deterministic
  fallback was used.
- **Cache surprises:** caches are keyed by video fingerprint and analysis configuration. Use
  `--force` to ignore resume state, or select a new cache directory.

Use `strict=True` while validating a deployment so recoverable pipeline failures surface
immediately.
