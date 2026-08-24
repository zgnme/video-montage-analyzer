# Frame selection

The v1.2 dynamic selector separates scene discovery from the final frame budget.

1. Probe metadata once with FFprobe, falling back to OpenCV.
2. Inspect frames at `scene_scan_fps` and resize scans to `scan_width`.
3. Compute normalized grayscale differences.
4. Keep local peaks above `scene_change_threshold`.
5. Apply temporal suppression using `min_scene_gap`.
6. Preserve opening and closing frames, allocate at most `scene_budget_ratio` of remaining slots to
   cuts, then fill the largest temporal gaps.

`frames_per_minute` determines the target:

```text
clamp(round(duration_seconds / 60 * frames_per_minute), min_frames, max_frames)
```

Scene density changes which frames win, never the maximum number of provider calls. Every frame is
marked as `opening`, `closing`, `scene_change`, or `uniform_fill`.

Use `UniformFrameSelector` for reproducible temporal spacing. Dynamic scanning defaults to 2 FPS at
approximately 320 pixels wide; benchmark and tune the threshold for unusual footage.
