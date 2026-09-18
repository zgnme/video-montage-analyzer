"""Full presentation-frame coverage, independent of CFR/VFR and worker count."""

from __future__ import annotations

import bisect
import json

from .cli import fingerprint
from .media import (
    decode_source,
    detect_shots,
    digest,
    extract_frames,
    frame_times,
    probe,
    write_json,
)


def candidate(index, frames, origin="detector"):
    if not 0 < index < len(frames):
        raise ValueError("Boundary must have an adjacent frame on both sides")
    return {
        "id": f"c{index:08d}",
        "index": index,
        "time": frames[index]["time"],
        "origin": origin,
        "left": frames[index - 1],
        "right": frames[index],
    }


def window(frames, start, stop, candidates, duration, offset, context=8):
    lo, hi = max(0, start - context), min(len(frames), stop + context)
    selected = frames[lo:hi]
    local = [
        {
            "id": c["id"],
            "time": c["time"],
            "left_ref": c["index"] - lo,
            "right_ref": c["index"] - lo + 1,
        }
        for c in candidates
    ]
    return {
        "frames": selected,
        "core_start_index": start,
        "core_stop_index": stop,
        "core_start": frames[start]["time"],
        "core_end": frames[stop]["time"] if stop < len(frames) else offset + duration,
        "candidates": local,
    }


def make_windows(frames, candidates, duration, offset, core_frames=48, context=8):
    return [
        {
            "id": f"w{start:08d}",
            **window(
                frames,
                start,
                min(len(frames), start + core_frames),
                [c for c in candidates if start <= c["index"] < start + core_frames],
                duration,
                offset,
                context,
            ),
        }
        for start in range(0, len(frames), core_frames)
    ]


def review_window(plan, boundary, radius=6):
    i = boundary["index"]
    return window(
        plan["frames"],
        i,
        i + 1,
        [boundary],
        plan["metadata"]["duration"],
        plan["settings"]["source_offset"],
        radius,
    )


def verify(plan, output):
    for index, frame in enumerate(plan["frames"]):
        path = (output / frame["path"]).resolve()
        if frame["index"] != index or not path.is_relative_to(output.resolve()):
            raise ValueError("Invalid evidence catalog")
        if digest(path) != frame["sha256"]:
            raise ValueError("Evidence changed; choose a fresh output directory")
    owned = [i for w in plan["windows"] for i in range(w["core_start_index"], w["core_stop_index"])]
    if owned != list(range(len(plan["frames"]))):
        raise ValueError("Primary windows must cover every input frame exactly once")


def prepare(args):
    source = args.video.expanduser().resolve(strict=True)
    metadata = probe(source)
    if metadata["duration"] > args.max_duration:
        raise ValueError("Input exceeds --max-duration; choose a clip or explicitly raise limit")
    settings = {
        "version": "precision-plan-1",
        "source_sha256": digest(source),
        "source_offset": args.source_offset,
        "frame_size": args.frame_size,
        "core_frames": args.core_frames,
        "context_frames": args.context_frames,
        "threshold": args.threshold,
    }
    key = fingerprint(settings)
    destination = args.output / "precision-plan.json"
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan["plan_key"] != key:
            raise ValueError("Different input/preparation parameters: use a new output directory")
        verify(plan, args.output)
        return plan
    times = frame_times(source)
    if len(times) > args.max_frames_total:
        raise ValueError("Frame count exceeds --max-frames-total; no API requests made")
    if any(a >= b for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Frame PTS must be strictly increasing")
    decoded = decode_source(source, metadata, times, args.output, args.frame_size)
    shots = detect_shots(decoded, times, metadata["duration"], args.threshold)
    paths = extract_frames(
        decoded, times, list(range(len(times))), args.output / "frames", args.frame_size
    )
    frames = [
        {
            "index": i,
            "time": t + args.source_offset,
            "path": paths[i],
            "sha256": digest(args.output / paths[i]),
        }
        for i, t in enumerate(times)
    ]
    candidates = [candidate(bisect.bisect_left(times, s["start"]), frames) for s in shots[1:]]
    windows = make_windows(
        frames,
        candidates,
        metadata["duration"],
        args.source_offset,
        args.core_frames,
        args.context_frames,
    )
    plan = {
        "plan_key": key,
        "settings": settings,
        "metadata": metadata,
        "source": str(source),
        "frames": frames,
        "candidates": candidates,
        "windows": windows,
        "clock": (
            "Presentation timestamps relative to first input frame plus explicit source offset"
        ),
        "decode_proxy_used": decoded != source,
    }
    verify(plan, args.output)
    write_json(destination, plan)
    return plan
