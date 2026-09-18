# Controlled sampling comparison

User-selected reference: YouTube `ElxeH-eXC88`, source interval 20–80 seconds.
YouTube video streams through 2160p and both downloaded source variants report 25 fps.
The 60-second test has 1,500 frames. No 60-fps stream was listed.

Three conditions used the same 640x360 JPEG evidence, model `gpt-5.6-luna`, low reasoning,
identical prompts, and 30 identical two-second scoring intervals with up to 0.2 seconds
of context overlap. Model calls were interleaved across conditions. FPS labels and local
cut detections were not supplied to the model. The user's later explicit authorization
allowed Luna for this comparison; the persistent deployment remains DeepSeek-only.

| FPS | Unique frames | Input tokens | Output tokens | Cached input | Estimated RUB |
|---|---:|---:|---:|---:|---:|
| 4 | 240 | 223020 | 11155 | 115200 | 0.534 |
| 8 | 480 | 313016 | 11707 | 115200 | 0.808 |
| 25 | 1500 | 682362 | 12229 | 115200 | 1.926 |

The estimate applies the user's displayed rate: 4.80 RUB per million internal units,
model factor 0.63, and 50% cached-input charging. It is not an invoice reconciliation.
There were 90 valid answers and one malformed JSON response that required a retry;
the unknown charge for that malformed answer is excluded. Cached valid trials were reused.

Mean successful request latency was 11.9 / 13.1 / 17.6 seconds for 4 / 8 / 25 fps.
Summed per-condition request durations divided by three workers yield projected minute-video
times of 1.99 / 2.18 / 2.94 minutes, before tail imbalance and retries. These conditions were
interleaved, not independently wall-clock timed. The union of successful API call intervals
was 7.45 minutes; first-to-last elapsed time including pauses was 11.71 minutes.
The production `video-montage analyze` command remains sequential; the benchmark runner
provides the tested three-worker concurrency. Do not present the projected parallel time as
a measured full-length production CLI runtime.

Manual inspection checked before/at/after frames at 41 detector candidates. Thirty-eight
were annotated as abrupt shot/framing boundaries. A graphic overlay, an ambiguous image
jump and a light transition were excluded from hard-cut ground truth. This is not an
exhaustive semantic annotation of every effect in the minute.

When deduplicating cut predictions within 80 ms and matching one-to-one within 350 ms,
4/8/25 fps respectively localized 36/35/36 of these 38 boundaries. Median timestamp
errors among matches were 120/40/0 ms. For predicted intervals the midpoint was used.
These figures measure this run's localization, not correctness of every description.
At full FPS, two abrupt framing changes were described as movement rather than listed
as cuts. Full FPS also did not prevent an abrupt color swap being described as gradual.

For the 79.32–79.52 second light/blur transition, 4 fps could not establish the type;
8 fps recognized a bright blurred transition; 25 fps provided a more precise start and
identified the subsequent translucent duplicate graphic. At 22.12 seconds the full-rate
frames show a phone graphic scaling in; 4 fps could not establish smooth vs instantaneous
appearance while 8 and 25 fps described the scaling.

Recommendation for short detailed references: preserve the source frame rate, use bounded
ordered windows and local boundary evidence. For this video that means 25 fps. 8 fps is
a reasonable overview compromise; 4 fps should not be the sole basis for reconstructing
fast transitions. One clip and one run per condition do not establish a universal optimum.

`scripts/benchmark_montage_fps.py` runs the controlled comparison on the prepared minute
and explicit per-run model authorization. Private media, individual model answers, audit
contact sheets and detailed annotations are retained in the server's output directory,
not this repository.
