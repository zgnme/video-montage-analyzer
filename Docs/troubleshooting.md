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
- **OpenRouter shared-pool throttling:** use `max_workers=1` for a rate-limited model and enable a
  cache with `--resume` so completed frames survive retries. A provider-side HTTP 429 is surfaced as
  `RateLimitError`; the library does not silently switch models.
- **Schema failures:** OpenRouter requests include both `response_format` and an explicit schema
  instruction. One text-only repair is attempted when a routed vision provider ignores the format;
  invalid repaired output still fails validation.
- **Cache surprises:** caches are keyed by video fingerprint and analysis configuration. Use
  `--force` to ignore resume state, or select a new cache directory.

Use `strict=True` while validating a deployment so recoverable pipeline failures surface
immediately.
