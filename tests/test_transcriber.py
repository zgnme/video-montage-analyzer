from types import SimpleNamespace

import pytest

from openscenesense.exceptions import TranscriptionError
from openscenesense.transcriber import NoAudioTranscriber, OpenAITranscriber


def test_no_audio_transcriber_is_a_noop():
    assert NoAudioTranscriber().transcribe("anything") == []


def test_transcriber_skips_video_without_audio(monkeypatch):
    client = SimpleNamespace()
    transcriber = OpenAITranscriber(client)
    monkeypatch.setattr("openscenesense.transcriber.has_audio_stream", lambda _path: False)

    assert transcriber.transcribe("silent.mp4") == []


def test_transcriber_maps_segments_and_removes_temporary_file(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")
    response = SimpleNamespace(
        segments=[
            SimpleNamespace(text="hello", start=1, end=2, confidence=0.8),
        ]
    )

    def create(**_kwargs):
        return response

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    transcriber = OpenAITranscriber(client)
    monkeypatch.setattr("openscenesense.transcriber.has_audio_stream", lambda _path: True)
    monkeypatch.setattr(transcriber, "_extract_wav", lambda _path: str(wav))

    segments = transcriber.transcribe("video.mp4")

    assert segments[0].text == "hello"
    assert segments[0].confidence == 0.8
    assert not wav.exists()


def test_transcriber_wraps_provider_failure(tmp_path, monkeypatch):
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF")

    def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=fail)))
    transcriber = OpenAITranscriber(client)
    monkeypatch.setattr("openscenesense.transcriber.has_audio_stream", lambda _path: True)
    monkeypatch.setattr(transcriber, "_extract_wav", lambda _path: str(wav))

    with pytest.raises(TranscriptionError, match="provider unavailable"):
        transcriber.transcribe("video.mp4")
    assert not wav.exists()
