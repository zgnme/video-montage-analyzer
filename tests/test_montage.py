import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest

from openscenesense.montage import cli, media, provider


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "insert.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x90:r=30:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=160x90:r=30:d=0.0666667",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=30:d=1",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-g",
            "30",
            str(path),
        ],
        check=True,
    )
    return path


def args_for(clip, tmp_path):
    return SimpleNamespace(
        video=clip,
        output=tmp_path / "out",
        max_duration=7200,
        fps=4,
        max_frames=24,
        threshold=27,
        frame_size=320,
        no_audio=True,
        max_windows=100,
        transcribe=False,
        transcript=None,
        config=tmp_path / "config.env",
        max_calls=300,
    )


def test_short_insert_and_cut_frames_are_preserved(clip, tmp_path):
    times = media.frame_times(clip)
    shots = media.detect_shots(clip, times, media.probe(clip)["duration"], 27)
    assert [s["start"] for s in shots] == pytest.approx([0, 1, 32 / 30], abs=0.002)
    windows = media.build_windows(shots, times, times[-1] + 1 / 30)
    insert = next(w for w in windows if w["kind"] == "shot" and w["shot_id"] == 2)
    assert insert["indices"] == [30, 31]
    assert sum(w["kind"] == "transition" for w in windows) == 2
    first_cut = next(w for w in windows if w.get("cut") == 1)
    assert any(times[i] < 1 for i in first_cut["indices"])
    assert any(times[i] >= 1 for i in first_cut["indices"])
    paths = media.extract_frames(clip, times, [30, 31], tmp_path / "frames")
    from PIL import Image

    r, g, b = Image.open(tmp_path / paths[30]).getpixel((80, 45))
    assert g > r and g > b


def test_packet_pts_fast_path_matches_decoded_b_frames(clip):
    expected_raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(clip),
        ]
    )
    decoded = [float(x["best_effort_timestamp_time"]) for x in json.loads(expected_raw)["frames"]]
    assert media.frame_times(clip) == pytest.approx([x - decoded[0] for x in decoded], abs=1e-6)


def test_variable_frame_rate_uses_presentation_times(clip, tmp_path):
    vfr = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(clip),
            "-vf",
            "setpts='if(lt(N,30),N/(30*TB),(1+(N-30)/15)/TB)'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(vfr),
        ],
        check=True,
    )
    times = media.frame_times(vfr)
    shots = media.detect_shots(vfr, times, media.probe(vfr)["duration"], 27)
    assert shots[1]["start"] == pytest.approx(1, abs=0.035)
    assert shots[2]["start"] == pytest.approx(1 + 2 / 15, abs=0.035)
    assert times[-1] > 2.9


def test_ffmpeg_proxy_preserves_short_insert_and_timing(clip, tmp_path):
    times = media.frame_times(clip)
    metadata = media.probe(clip)
    metadata["codec"] = "av1"  # Exercise the mandatory FFmpeg decoder fallback.
    decoded = media.decode_source(clip, metadata, times, tmp_path, 320)
    assert decoded != clip
    assert media.frame_times(decoded) == pytest.approx(times, abs=0.003)
    shots = media.detect_shots(decoded, times, metadata["duration"], 27)
    assert [s["start"] for s in shots] == pytest.approx([0, 1, 32 / 30], abs=0.002)


def test_long_continuous_shot_has_no_window_gaps():
    times = [i / 30 for i in range(1800 * 30)]
    windows = media.build_windows(
        [{"id": 1, "start": 0, "end": 1800, "duration": 1800}], times, 1800
    )
    assert windows[0]["start"] == 0 and windows[-1]["end"] == 1800
    assert all(a["end"] == b["start"] for a, b in zip(windows, windows[1:], strict=False))
    assert all(3 <= len(w["indices"]) <= 24 for w in windows)
    assert sum(len(w["indices"]) for w in windows) > 7000


def analysis():
    return {
        "summary": "Красный сменяется синим",
        "observations": [{"description": "Смена цвета", "frames": [1, 2], "confidence": "high"}],
        "recreation_steps": ["Поставить прямую склейку"],
        "uncertainties": [],
    }


def test_provider_sends_multiple_images_with_absolute_timestamps(tmp_path, monkeypatch):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"synthetic")
    captured = []

    def handle(request):
        data = json.loads(request.content)
        captured.append(data)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(analysis())}]}],
            },
        )

    client_class = httpx.Client
    monkeypatch.setattr(
        provider.httpx,
        "Client",
        lambda **kw: client_class(transport=httpx.MockTransport(handle), **kw),
    )
    api = provider.VisionAPI(
        {
            "MONTAGE_API_KEY": "test-only",
            "MONTAGE_MODEL": "deepseek-flash",
            "MONTAGE_BASE_URL": "https://example.invalid/v1",
            "MONTAGE_API_MODE": "responses",
        }
    )
    assert api.request("compare", [(image, 59.5), (image, 60.0)]) == analysis()
    content = captured[0]["input"][0]["content"]
    assert sum(p["type"] == "input_image" for p in content) == 2
    assert "59.500000" in content[1]["text"] and "60.000000" in content[3]["text"]
    assert captured[0]["model"] == "deepseek-flash"


@pytest.mark.parametrize("finish", ["response.failed", "response.incomplete", "missing"])
def test_truncated_stream_is_not_success(finish):
    events = 'data: {"type":"response.output_text.delta","delta":"{}"}\n\n'
    if finish != "missing":
        events += "data: " + json.dumps({"type": finish}) + "\n\n"
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=events.encode()
    )
    with pytest.raises(ValueError):
        provider.VisionAPI._read(response)


def test_invalid_evidence_reference_rejected():
    data = analysis()
    data["observations"][0]["frames"] = [99]
    with pytest.raises(ValueError, match="missing evidence"):
        provider.validate_analysis(data, 3)


def test_foreign_credentials_are_not_used(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("MONTAGE_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-use")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-use")
    with pytest.raises(ValueError, match="MONTAGE_API_KEY"):
        provider.load_config(tmp_path / "absent.env")


def test_disallowed_model_is_blocked_before_any_request():
    with pytest.raises(ValueError, match="no API request"):
        provider.VisionAPI(
            {"MONTAGE_ALLOWED_MODELS": "deepseek-flash", "MONTAGE_MODEL": "gpt-5.6-luna"}
        )


def test_cyrillic_transcript_preserved(tmp_path):
    f = tmp_path / "speech.txt"
    f.write_text("[00001.000 --> 00002.000] Камера приближается.\n")
    assert cli.load_transcript(f)[0]["text"] == "Камера приближается."


def test_prepare_resume_and_tampered_evidence(clip, tmp_path):
    args = args_for(clip, tmp_path)
    args.output.mkdir()
    manifest = cli.prepare(args)
    assert cli.prepare(args) == manifest
    frame = args.output / manifest["windows"][0]["frames"][0]["path"]
    frame.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Evidence frame changed"):
        cli.prepare(args)


def test_api_failure_resume_does_not_repeat_completed_windows(clip, tmp_path, monkeypatch):
    args = args_for(clip, tmp_path)
    args.output.mkdir()
    manifest = cli.prepare(args)
    monkeypatch.setattr(cli, "load_config", lambda _: {})
    state = {"requests": 0, "fail": True}

    class API:
        identity = {"model": "test"}

        def __init__(self, config):
            self.calls = 0
            self.usage = []

        def request(self, prompt, frames):
            self.calls += 1
            state["requests"] += 1
            if state["fail"] and state["requests"] == 2:
                raise RuntimeError("controlled failure")
            return analysis()

    monkeypatch.setattr(cli, "VisionAPI", API)
    assert cli.analyze(args, manifest) == 2
    first = json.loads((args.output / "result.json").read_text())
    assert first["completed_windows"] == 1 and first["status"] == "partial"
    state["fail"] = False
    assert cli.analyze(args, manifest) == 0
    assert state["requests"] == len(manifest["windows"]) + 1
    final = json.loads((args.output / "result.json").read_text())
    assert final["status"] == "complete"
    assert "test-only" not in (args.output / "report.html").read_text()


def test_output_lock_prevents_duplicate_processing(tmp_path):
    with cli.lock(tmp_path):
        with pytest.raises(ValueError, match="Another run"):
            with cli.lock(tmp_path):
                pytest.fail("lock must exclude second run")


def test_termination_stops_owned_media_child(tmp_path):
    marker = tmp_path / "child.pid"
    child_code = (
        "import os,time; from pathlib import Path; "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    parent_code = (
        "from openscenesense.montage.media import run; "
        f"run([{sys.executable!r}, '-c', {child_code!r}])"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 8
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "Media child never started"
        child_pid = int(marker.read_text())
        parent.send_signal(signal.SIGTERM)
        parent.communicate(timeout=8)
        state = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True
        )
        assert not state.stdout.strip() or state.stdout.strip().startswith("Z")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate()
        if child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
