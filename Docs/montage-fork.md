# Video montage analysis fork

This fork adds `video-montage` to MIT-licensed OpenSceneSense. Upstream source is retained;
the new montage path replaces independent image captions with ordered multi-image requests.
Upstream base: `ef38012341fef88d40c4b8a596b4c1a5d361f8dc`.
No ShotParser code is included: no explicit license was found in that repository.

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
```

Both `responses` and `chat_completions` accept multi-image payloads. No provider-specific strict
JSON Schema feature is required: results are validated locally, including evidence frame IDs.
The application uses only its dedicated configuration and MONTAGE_* environment variables.
There is no fallback to Anthropic, another model, or a different host. Redirects are not followed.

Output: `manifest.json` (evidence), `result.json` (coverage/status/provider/usage), `report.md`,
`report.html`, `frames/`, resumable `analyses/`. Cache identity includes full source hash, sampling,
endpoint, model, prompt and transcript. Frame hashes are checked on resume. Output directories
are exclusively locked. Failed or truncated responses never count as completed analysis.

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
resume. Mock API tests verify behavior, not the perceptual quality of a model. Real provider
qualification uses synthetic images/video only; no user media is included in this repository.
