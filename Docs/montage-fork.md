# Video montage analysis fork

This fork adds `video-montage` to MIT-licensed OpenSceneSense. Upstream source is retained;
the new montage path replaces independent image captions with ordered multi-image requests.
Upstream base: `ef38012341fef88d40c4b8a596b4c1a5d361f8dc`.
No ShotParser code is included: no explicit license was found in that repository.

For detailed montage work, use the new [precision mode](precision-mode.md): all input frames,
up to ten parallel requests, independent boundary reviews and a durable usage ledger.
The sampled `prepare/analyze` commands described below remain available for compatibility.

## Selection and qualification

OpenSceneSense is the reusable package/test/provider foundation, **not a demonstrated best
montage analyzer**. ShotParser has a more relevant shot-oriented design but sends native Gemini
video, has no discovered license, and its offset re-scan loses a known cut in our synthetic test.
VideoReverse's image extraction assigns I-frame timestamps using the extracted image index
divided by source FPS; this does not preserve presentation time. byjlw/video-analyzer processes
individual images with text history. These observations motivated a separate shot-aware path.

The new path uses PySceneDetect 0.6.7.1 Content + Adaptive detectors at full decoded frame rate
with minimum scene length one frame. It maps detected frame indices to FFprobe presentation
timestamps, retaining variable frame rates and micro-inserts. Detections are candidates, not
guaranteed exact editing decisions. Every shot has visual coverage. Longer shots are split into
bounded windows; transitions get separate dense before/after frame series. This is a fixed
evidence strategy, not an autonomous model-directed rewatch loop.

## Install and run

Python 3.12, FFmpeg/FFprobe, CPU. No local VLM or CUDA is needed.

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
video-montage prepare /path/video.mp4 -o /path/run
video-montage analyze /path/video.mp4 -o /path/run
```

Create `~/.config/video-montage/config.env` with mode 0600:

```dotenv
MONTAGE_BASE_URL=https://codex.sale/v1
MONTAGE_MODEL=deepseek-flash
MONTAGE_API_MODE=responses
MONTAGE_API_KEY=YOUR_OWN_KEY
MONTAGE_MAX_OUTPUT_TOKENS=8192
MONTAGE_REASONING_EFFORT=low
MONTAGE_ALLOWED_MODELS=deepseek-flash
```

Both `responses` and `chat_completions` accept multi-image payloads. No provider-specific strict
JSON Schema feature is required: results are validated locally, including evidence frame IDs.
The output allowance includes provider reasoning. A 3,000-token allowance truncated a real
DeepSeek qualification request (the same detailed task also returned HTTP 503 through Responses); 8,192 is
the default. This is a cap, not a claim that every call consumes 8,192 tokens.
`--model` and `--reasoning` are explicit per-run overrides. They never rewrite the dedicated
configuration. Default reasoning is `low`; `high`/`max` remain available. Actual provider,
model and effort are recorded in results and separated in the cache identity.
The deployed allowed-model list permits only `deepseek-flash`. A per-run override to any
other model is rejected before a request. The user subsequently authorized Luna for qualification; a process-local
`MONTAGE_ALLOWED_MODELS=gpt-5.6-luna` override with `--model gpt-5.6-luna` implements that choice.
DeepSeek remains the default. Anthropic is not authorized for this tool.
The application uses only its dedicated configuration and MONTAGE_* environment variables.
There is no fallback to Anthropic, another model, or a different host. Redirects are not followed.

Output: `manifest.json` (evidence), `result.json` (coverage/status/provider/usage), `report.md`,
`report.html`, `frames/`, resumable `analyses/`. Cache identity includes full source hash, sampling,
endpoint, model, prompt and transcript. Frame hashes are checked on resume. Output directories
are exclusively locked. Failed or truncated responses never count as completed analysis.

AV1 (or a source rejected by OpenCV's decoder) is normalized by system FFmpeg to a local H.264
decode proxy. Frame count and presentation timing must match before any evidence is accepted.
For one-packet-per-frame MP4 with a matching declared frame count, sorted packet PTS avoid a
full decode just to read the clock; other files fall back to decoded frame timestamps.

`--max-calls` bounds attempts per run; default 300 includes transient retry allowance. A partial
run exits 2 and can be resumed. `prepare` makes no API calls. Costs depend on images, output,
provider rates and the number of windows; prior estimates for a single full-video Gemini request
do not apply to this multi-request workflow. Detailed runs can be substantially more expensive.

## Audio

Audio energy accents are measured locally at 10 ms resolution. They are **not verified musical
beats, sound-effect labels or a music transcript**. Image-only models cannot listen to audio.
Optional `--transcribe` calls a separately installed local Whisper environment via two explicit
configuration fields: `MONTAGE_ASR_PYTHON` and `MONTAGE_ASR_SCRIPT`. The supplied
`scripts/montage_asr.py` uses a pre-cached model, three CPU threads, and preserves timed Russian
text. Alternatively `--transcript` accepts timed JSON or `[seconds --> seconds] text`.

## Tests

```sh
.venv/bin/pytest -q
.venv/bin/ruff check openscenesense/montage tests/test_montage.py scripts/montage_asr.py
```

Tests include two-frame insert retention, VFR timestamp mapping, full-duration window coverage,
multi-image API payloads with timestamps, interrupted responses, wrong evidence references,
credential isolation, Cyrillic transcripts, frame corruption, process locking and failed-run
resume. Decoder proxy timing is verified, packet timestamps are compared with decoded B-frames,
and terminating the wrapper is tested to stop its owned media subprocess. Mock API tests verify
behavior, not the perceptual quality of a model. Real provider
qualification initially used synthetic images/video, then the user-selected public reference;
no user media is included in this repository.

## Live provider qualification status (2026-09-19)

DeepSeek through the supplied codex.sale endpoint correctly identified ordered synthetic
images and completed some multi-image synthetic windows, but repeatedly returned HTTP 503
on detailed tasks and on a single 320px real video frame. This is an unresolved live-provider
failure, not a passing end-to-end DeepSeek qualification. Short GPT control requests succeeded
before the user prohibited Luna; no fallback is enabled. The server tool is installed and
locally tested, but its default provider's real-video analysis is currently blocked by these
errors. Do not present the control-model output as a DeepSeek result.
Subsequent bounded diagnostics returned the same 503 for text-only, synthetic JPEG, real JPEG
and real PNG through both Responses and Chat Completions. Even a bare text-only Chat Completions
request without optional parameters failed in under one second while `/models` returned 200.
The current failure is therefore not specific to video/image size. The gateway does not reveal
whether its upstream channel, quota, billing or model routing is responsible; that requires
provider logs. Identifiers and private diagnostics are retained only in the server output folder.

The user-selected reference was YouTube `ElxeH-eXC88` (1595 seconds). The original AV1 stream
exposed the bundled OpenCV decoder incompatibility; the proxy path fixes the general case,
and the H.264 source variant is used for the full reference scan. Visual/API spot checks are
limited to selected clips, not a paid semantic analysis of all 26 minutes. Downloaded media,
captions, keys and full model reports remain outside this public repository.
The H.264 reference scan completed with 909 candidate shots, 1,863 windows and 15,878 unique
evidence frames. Coverage is continuous across 1,594.541 seconds. These are detector candidates,
not human-validated cut counts. The final local and server test suites both passed 74 tests.
