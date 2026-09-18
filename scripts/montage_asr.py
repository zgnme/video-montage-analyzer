"""Run using the server's existing faster-whisper environment, not the video API."""

import argparse
import os
from pathlib import Path

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--language")
    args = parser.parse_args()
    model = WhisperModel(
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        device="cpu",
        compute_type="int8",
        cpu_threads=3,
        num_workers=1,
        local_files_only=True,
    )
    segments, info = model.transcribe(
        args.input,
        language=args.language,
        beam_size=5,
        vad_filter=True,
    )
    temporary = args.output.with_suffix(".tmp")
    os.umask(0o077)
    with temporary.open("w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"[{segment.start:09.3f} --> {segment.end:09.3f}] {segment.text.strip()}\n")
    temporary.replace(args.output)
    print(f"Local transcript complete; language={info.language}")


if __name__ == "__main__":
    main()
