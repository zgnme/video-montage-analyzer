from types import SimpleNamespace

import cv2
import pytest

from openscenesense.exceptions import VideoLoadError
from openscenesense.video_utils import _rate, has_audio_stream, probe_video


def test_rate_handles_fraction_and_invalid_values():
    assert _rate("30000/1001") == pytest.approx(29.97, rel=1e-3)
    assert _rate("0/0") == 0
    assert _rate("invalid") == 0


def test_probe_uses_ffprobe_and_derives_missing_frame_count(tmp_path, monkeypatch):
    video = tmp_path / "video.bin"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "openscenesense.video_utils.ffmpeg.probe",
        lambda _path: {
            "streams": [
                {
                    "codec_type": "video",
                    "avg_frame_rate": "10/1",
                    "duration": "2.5",
                    "width": 320,
                    "height": 180,
                }
            ],
            "format": {},
        },
    )

    metadata = probe_video(str(video))

    assert metadata.frame_count == 25
    assert metadata.fps == 10
    assert metadata.duration == 2.5


def test_probe_falls_back_to_opencv_when_ffprobe_metadata_is_bad(tmp_path, monkeypatch):
    video = tmp_path / "video.bin"
    video.write_bytes(b"video")
    monkeypatch.setattr("openscenesense.video_utils.ffmpeg.probe", lambda _path: {})

    class Capture:
        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FPS: 0,
                cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_POS_MSEC: 1500,
            }
            return values.get(prop, 0)

        def set(self, *_args):
            return True

        def read(self):
            return True, SimpleNamespace()

        def release(self):
            return None

    monkeypatch.setattr("openscenesense.video_utils.cv2.VideoCapture", lambda _path: Capture())

    metadata = probe_video(str(video))

    assert metadata.fps == 0
    assert metadata.frame_count == 5
    assert metadata.duration == 1.5


def test_probe_rejects_missing_and_corrupt_files(tmp_path):
    with pytest.raises(VideoLoadError, match="does not exist"):
        probe_video(str(tmp_path / "missing.mp4"))

    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not video")
    with pytest.raises(VideoLoadError, match="open"):
        probe_video(str(corrupt))


def test_audio_probe_is_conservative_on_probe_failure(monkeypatch):
    monkeypatch.setattr(
        "openscenesense.video_utils.ffmpeg.probe",
        lambda _path: (_ for _ in ()).throw(OSError("missing")),
    )
    assert has_audio_stream("video.mp4") is True
