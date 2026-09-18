from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector


def run(argv: list[str], timeout: int = 3600) -> bytes:
    process = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Media operation interrupted")

        signal.signal(signal.SIGTERM, interrupted)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
        raise
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    if process.returncode:
        # Do not include arbitrary subprocess output (media metadata may be untrusted).
        raise RuntimeError(f"{Path(argv[0]).name} failed (exit {process.returncode})")
    return stdout


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def probe(path: Path) -> dict:
    data = json.loads(
        run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    )
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("Input has no video stream")
    duration = float(data["format"].get("duration", video.get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration is invalid")
    return {
        "duration": duration,
        "codec": video.get("codec_name", "unknown"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def frame_times(path: Path) -> list[float]:
    # Preserve presentation timestamps: extracted I-frame numbering / FPS is NOT time.
    # MP4 exposes per-packet PTS and an authoritative frame count. Avoid decoding AV1
    # just to read its clock; fall back when packets cannot map one-to-one to frames.
    packet_data = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_streams",
                "-show_entries",
                "packet=pts_time:stream=nb_frames",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    packets = packet_data.get("packets", [])
    declared = packet_data.get("streams", [{}])[0].get("nb_frames", "0")
    try:
        packet_times = sorted(float(p["pts_time"]) for p in packets)
        if (
            int(declared) == len(packet_times) > 0
            and len(set(packet_times)) == len(packet_times)
            and all(math.isfinite(t) for t in packet_times)
        ):
            return [t - packet_times[0] for t in packet_times]
    except (KeyError, ValueError, TypeError):
        pass
    raw = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "2",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    times = []
    for line in raw.decode().splitlines():
        try:
            value = float(line.split(",")[0])
        except ValueError:
            continue
        if math.isfinite(value):
            times.append(value)
    if not times or any(b < a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("Could not obtain monotonic video timestamps")
    origin = times[0]
    return [t - origin for t in times]


def decode_source(path: Path, metadata: dict, times: list[float], workdir: Path, size: int) -> Path:
    """Use system FFmpeg when wheel-bundled OpenCV cannot decode the source codec."""
    import cv2

    supported = False
    if metadata.get("codec") != "av1":
        cap = cv2.VideoCapture(str(path))
        try:
            supported = cap.isOpened() and cap.read()[0]
        finally:
            cap.release()
    if supported:
        return path
    print("Creating an H.264 decode proxy with system FFmpeg; preserving frame timing…", flush=True)
    proxy = workdir / "decode-proxy.mp4"
    temporary = workdir / "decode-proxy.tmp.mp4"
    # Keep source frame order and variable timing. Audio is handled from the original file.
    scale = f"scale=w='min({size},iw)':h=-2:flags=lanczos,setpts=PTS-STARTPTS"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-threads",
            "2",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            scale,
            "-an",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "14",
            "-threads",
            "2",
            str(temporary),
        ]
    )
    proxy_times = frame_times(temporary)
    if len(proxy_times) != len(times) or any(
        abs(a - b) > 0.003 for a, b in zip(times, proxy_times, strict=False)
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError("Decode proxy changed frame count/timing; refusing shifted evidence")
    temporary.replace(proxy)
    return proxy


def detect_shots(path: Path, times: list[float], duration: float, threshold: float) -> list[dict]:
    video = open_video(str(path))
    manager = SceneManager()
    # Full-frame scan, including single-frame inserts. Union retains candidates for review.
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=1))
    manager.add_detector(AdaptiveDetector(min_scene_len=1))
    try:
        processed = manager.detect_scenes(video, show_progress=False)
        if processed != len(times):
            raise ValueError("Scene detector did not decode every source frame")
        scenes = manager.get_scene_list(start_in_scene=True)
    finally:
        # VideoStreamCv2 has no public close method in the pinned release.
        capture = getattr(video, "_cap", None)
        if capture is not None:
            capture.release()
    indices = sorted({0, *(s[0].get_frames() for s in scenes)})
    if indices and indices[-1] >= len(times):
        raise ValueError("Detector frame indices do not match decoded timestamps")
    boundaries = sorted({0.0, *(times[i] for i in indices), duration})
    return [
        {"id": i + 1, "start": a, "end": b, "duration": b - a}
        for i, (a, b) in enumerate(zip(boundaries, boundaries[1:], strict=False))
        if b > a
    ]


def nearest_indices(times: list[float], start: float, end: float, count: int) -> list[int]:
    lo = int(np.searchsorted(times, start, side="left"))
    hi = int(np.searchsorted(times, end, side="left")) - 1
    lo = min(lo, len(times) - 1)
    hi = max(lo, min(hi, len(times) - 1))
    return sorted(set(np.linspace(lo, hi, min(count, hi - lo + 1)).round().astype(int).tolist()))


def build_windows(
    shots: list[dict], times: list[float], duration: float, fps: float = 4, max_frames: int = 24
) -> list[dict]:
    windows = []
    chunk_seconds = (max_frames - 1) / fps
    for shot in shots:
        n = max(1, math.ceil(shot["duration"] / chunk_seconds))
        edges = np.linspace(shot["start"], shot["end"], n + 1).tolist()
        for a, b in zip(edges, edges[1:], strict=False):
            count = min(max_frames, max(3, math.ceil((b - a) * fps) + 1))
            windows.append(
                {
                    "kind": "shot",
                    "shot_id": shot["id"],
                    "start": a,
                    "end": b,
                    "indices": nearest_indices(times, a, b, count),
                }
            )
    for shot in shots[1:]:
        cut = shot["start"]
        a, b = max(0, cut - 0.35), min(duration, cut + 0.35)
        # Dense boundary evidence, including both adjacent sides of a cut.
        windows.append(
            {
                "kind": "transition",
                "shot_id": shot["id"],
                "cut": cut,
                "start": a,
                "end": b,
                "indices": nearest_indices(times, a, b, max_frames),
            }
        )
    windows.sort(key=lambda w: (w["start"], w["kind"]))
    for i, window in enumerate(windows):
        window["id"] = f"w{i + 1:05d}"
    return windows


def extract_frames(
    path: Path, times: list[float], indices: list[int], dest: Path, size: int = 960
) -> dict[int, str]:
    import cv2

    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(indices)
    result = {}
    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while wanted:
            ok = cap.grab()
            if not ok:
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Failed to decode selected frame {i}")
                h, w = frame.shape[:2]
                scale = min(1.0, size / max(h, w))
                if scale < 1:
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
                filename = f"f{i:08d}.jpg"
                if not cv2.imwrite(str(dest / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    raise ValueError("Could not write evidence frame")
                result[i] = "frames/" + filename
                wanted.remove(i)
            i += 1
    finally:
        cap.release()
    if wanted:
        raise ValueError("Some selected frames were not decoded; refusing incomplete evidence")
    return result


def audio_energy(path: Path, workdir: Path) -> dict:
    pcm = workdir / "audio-energy.wav"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(pcm),
        ]
    )
    try:
        with wave.open(str(pcm)) as w:
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
        step = 160  # 10 ms envelope, not a beat or music classifier.
        n = len(samples) // step
        if n < 3:
            return {"method": "rms_onset_10ms", "accents": []}
        rms = np.sqrt(np.mean(samples[: n * step].reshape(n, step) ** 2, axis=1)) / 32768
        changes = np.maximum(0, np.diff(rms, prepend=0))
        threshold = max(0.025, float(np.quantile(changes, 0.98)))
        selected = []
        for i in np.argsort(changes)[::-1]:
            if changes[i] < threshold:
                break
            t = float(i) / 100
            if all(abs(t - x["time"]) >= 0.15 for x in selected):
                selected.append({"time": t, "strength": round(float(changes[i]), 4)})
            if len(selected) >= 2000:
                break
        return {
            "method": "rms_onset_10ms",
            "accents": sorted(selected, key=lambda x: x["time"]),
            "limitation": "Energy changes only; not verified musical beats or sound identities.",
        }
    finally:
        pcm.unlink(missing_ok=True)
