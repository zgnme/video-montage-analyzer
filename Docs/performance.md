# Performance

The v1.2 selector reduces scene scanning work and API payload size before changing model-side
latency.

On the bundled 14.04-second, 351-frame development fixture, the selector inspected 31 frames
(91.17% fewer than a full-frame scan). A final local comparison measured about 2.23 seconds for v1.2
versus 2.72 seconds for the legacy scan, a 1.22× speedup. These numbers describe one machine and
codec, not a universal guarantee.

Reproduce the comparison:

```bash
python benchmarks/benchmark_frame_selector.py Examples/pizza.mp4
```

For cloud workloads, the main cost controls are `max_frames`, `frames_per_minute`,
`max_image_dimension`, JPEG quality, audio enablement, and model selection. Frame analysis is
concurrent up to `max_workers`; increase it only within provider rate limits. Metadata contains
per-stage wall time and provider usage counts so applications can measure their own workloads.
