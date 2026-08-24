# Changelog

## 1.2.0 - Unreleased

### Added

- Shared schema-versioned `AnalysisResult` contract and `analyze_video_structured()`.
- OpenAI and OpenRouter provider adapters with explicit endpoint policies.
- OpenAI Responses image input, JSON Schema output, and usage extraction.
- Structured frame fields and a single-call summary/event result.
- Reduced-rate, budget-bounded dynamic frame selection.
- Progress callbacks, strict mode, failure-ratio limits, diagnostics, CLI, and resumable caches.
- Minimum/latest dependency CI across supported Python versions.

### Changed

- Updated the OpenAI SDK feature baseline to 3.3.1.
- Frames are resized and encoded as JPEG before submission.
- `analyze_video()` retains the legacy dictionary and now includes additive telemetry.
- Endpoint fallback is opt-in and limited to unsupported endpoints.
- OpenRouter now includes an explicit schema instruction and one validated text-only repair for
  routed vision models that ignore `response_format`.

### Removed

- Mandatory `librosa` and `soundfile` dependencies.
- Import-time FFmpeg checks, console warnings, and library-level `logging.basicConfig()`.
