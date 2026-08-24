import subprocess
import sys


def test_import_has_no_output_or_environment_probe():
    completed = subprocess.run(
        [sys.executable, "-c", "import openscenesense"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_v11_analyzer_exception_imports_remain_available():
    from openscenesense.analyzer import AudioTranscriptionError, FrameAnalysisError

    assert issubclass(AudioTranscriptionError, Exception)
    assert issubclass(FrameAnalysisError, Exception)
