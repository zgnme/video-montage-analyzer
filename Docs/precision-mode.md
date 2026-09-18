# Precision montage analysis

`video-montage precision` is the production path for recreating a short video reference.
It supplies **every input presentation frame**, including variable-frame-rate video, to a
vision API. It runs on a CPU server; model inference is remote. The legacy `prepare/analyze`
commands remain a separate sampled, sequential path.

```sh
video-montage precision /absolute/video.mp4 -o /absolute/run --transcribe
```

Omit `--transcribe` for silent material; use `--transcript` for existing timed JSON/text.
Input may be a downloaded file or a separately extracted clip. `--source-offset 20` maps a
clip beginning at source second 20 onto the original timeline. External transcript times
must use that resulting timeline. Local ASR output is offset automatically.

## Evidence and analysis

- FFprobe presentation times own the clock; no timestamps are invented by the model.
  Frame indices in reports belong to the input file, not necessarily its parent video.
- All decoded input frames are stored as JPEG, resized to at most 960 pixels on the long
  side by default. This preserves temporal coverage, not original pixel fidelity.
  Raise `--frame-size` up to 2048 for small text. Native source video remains authoritative.
- Primary windows own 48 frames each and receive up to 8 context frames on each side.
  Every input frame has exactly one primary owner. Model payloads do not depend on worker
  count or on the order in which other model answers finish.
- Full-rate Content + Adaptive detectors produce boundary candidates. For every candidate,
  the model must describe the immediately adjacent before/after frames and cite both.
  It may reject a candidate; it cannot silently classify an omitted candidate as absent.
- A separate request reviews every candidate. Omitted primary verdicts and model-proposed
  extra boundaries receive two reviews. Disagreements receive another independent review,
  with at most three valid votes per boundary. Review prompts do not contain earlier answers.
- Two non-low-confidence votes must agree on the broad class. Hard cuts and crop jumps
  share the abrupt-change class; camera motion and no-change share the no-edit class.
  Exact alternatives and all votes remain in the JSON. Unresolved candidates are explicit.
- Within-shot animations, inserts and camera motion have frame-anchored intervals. These
  are primary-pass model observations, not independently verified facts. Overlapping
  observations retain window provenance rather than being silently merged into invented events.
- Reports are assembled deterministically from evidence-owned results. No additional model
  rewrites the timeline or invents timings during final aggregation.

Two agreeing model requests can share the same mistake. Full frame delivery does not prove
that the model attended to every frame. This is a tested engineering workflow, not a claim
of universal perceptual accuracy or superiority over every video model.

## Concurrency, accounting and recovery

Default `--workers 10`, valid range 1–10. The first primary window is a provider pilot; a
failure there stops the run before opening the bulk parallel workload. At most three attempts
per task handle malformed responses and transient failures, with bounded delay. A terminal
failure stops new dispatch and drains already running requests. No automatic model fallback.

`precision.sqlite` uses a durable transactional ledger. Each attempt is recorded **before**
network dispatch. Usage is retained even if the returned JSON fails local validation. A
crashed/incomplete request with no usage is recorded as unknown, never free. Successful task
results are reused on resume, including reviews. A nonblocking directory lock prevents two
processes from owning the same run. Frame hashes are verified before reuse.

`--max-calls 300` and `--max-tokens 20000000` are cumulative across resumes in the same folder.
The token gate includes conservative reservations for in-flight/unknown requests. It is a
local planning limit, **not an upstream monetary cap**. `result.json` reports known input,
output, cached input, and unknown-usage attempts; billing remains the provider's authority.
A user-authorized larger task can resume by explicitly raising the cumulative limits.
Changing source, preparation settings, model, prompt version or transcript requires a fresh
output directory. Changing only worker count or increasing limits does not discard receipts.

SIGINT/SIGTERM stops dispatch; running calls can take up to the provider timeout to settle.
A hard kill leaves unknown attempts in the ledger; restart retains successful cached work.

## Results and exit codes

- `precision-plan.json`: input hash, exact frame times, frame hashes, window ownership,
  local detector candidates and preparation settings.
- `result.json`: actual model, coverage, candidate votes, unresolved IDs, observed effects,
  reproduction suggestions, cumulative usage and failures.
- `report.md`, `report.html`: timeline and instructions with before/after images.
- `montage.csv`: evidence-backed boundary sheet; `shots.csv`: in/out points and durations
  from agreed abrupt boundaries for downstream editing workflows.
- `frames/`: all evidence JPEGs. Keep beside HTML when sharing an offline report.

`status=complete` means all primary windows completed without an operational failure.
`quality_status=reviewed` means all candidates reached classification consensus; it does not
mean every possible effect was found. Exit 0: complete with consensus; exit 2: complete with
unresolved candidates; exit 1: partial/operational failure. Never present partial coverage as
an analysis of the entire input.

## Provider and audio

The dedicated deployment default is `deepseek-flash` through codex.sale Responses.
The key is read only from the private montage configuration or `MONTAGE_*` variables.
No Hermes, Anthropic or Codex credentials are reused. No automatic provider/model fallback.
For an explicitly selected Luna run, the deployment supports a process-scoped override:

```sh
MONTAGE_ALLOWED_MODELS=gpt-5.6-luna video-montage precision /absolute/video.mp4 \
  -o /absolute/luna-run --model gpt-5.6-luna
```

This does not alter DeepSeek as the default. User-authorized qualification used Luna;
Anthropic was not used. Provider availability remains outside this tool's control.

Local Whisper can supply speech timestamps. Image API requests do not receive audio.
The precision mode does not claim to recognize music, beats or sound effects. It cannot
recover an original editor project, exact plugin settings, lens or hidden layers.

## Qualification: 2026-09-19

One minute, original timeline 00:20–01:20 of user-selected YouTube `ElxeH-eXC88`:
1,500 source frames at 25 fps, 640×360 benchmark evidence, Luna with low reasoning,
10 workers, 32 primary windows and 40 detector candidates.

The run took 131.22 seconds of API workflow after frame preparation. It made 74 attempts
(72 valid task results plus two rejected responses retried), with known usage on every attempt:
1,169,218 input tokens, 27,979 output tokens, including 284,160 cached input tokens.
All 38 manually annotated abrupt boundaries matched their frame PTS exactly, including
crop jumps at 40.24 and 51.80 seconds missed by the earlier unprompted benchmark.
21.48 seconds was classified as graphics overlay and 79.32 as a gradual transition.
The manual annotation predates this run and was not provided to the model.

This is 38/38 recall against that boundary annotation, **not a universal 100% accuracy claim**.
Effect timing/description still has model uncertainty; primary-pass animation observations
are not covered by the boundary-consensus claim. The earlier [sampling benchmark](frame-sampling-benchmark.md)
measured a different prompt/workflow and must not be conflated with this qualification.

A replay of the same minute kept the attempt count at 74: no new API requests. A separate
122-frame synthetic reference preserved a two-frame insert, classifying its boundaries at
2.000000 and 2.066667 seconds correctly (five successful requests). Regression coverage
includes changed worker counts, omitted candidates, third-vote adjudication, model-proposed
boundaries, interruption/resume, cumulative budgets, invalid usage, real decoded-frame
integrity and escaped report HTML.

The deployed Hermes skill was discovered through `skills_list` and loaded through `skill_view`
without restarting the gateway. A final one-image probe of the unchanged default
`deepseek-flash` endpoint still returned HTTP 503; the successful live qualification above
used explicitly selected Luna. Model availability is not hidden by automatic fallback.
