"""Generate a synthetic reference: movement, two-frame insert, hard cut, text, audio accent."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    silent = args.output.with_suffix(".silent.mp4")
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    font = next(
        (ImageFont.truetype(p, 28) for p in font_paths if Path(p).exists()),
        ImageFont.load_default(),
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "320x180",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(silent),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(122):
        if i < 60:
            image = Image.new("RGB", (320, 180), "white")
            draw = ImageDraw.Draw(image)
            x = 20 + round(i * 3)
            draw.rectangle((x, 80, x + 45, 125), fill="red")
            draw.text((12, 10), "MOVE RIGHT", font=font, fill="black")
        elif i < 62:
            image = Image.new("RGB", (320, 180), "green")
            ImageDraw.Draw(image).text((65, 70), "INSERT", font=font, fill="white")
        else:
            image = Image.new("RGB", (320, 180), "blue")
            draw = ImageDraw.Draw(image)
            draw.ellipse((130, 70, 190, 130), fill="yellow")
            draw.text((80, 10), "FINAL", font=font, fill="white")
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Fixture encoding failed")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=4.066667:sample_rate=16000",
            "-af",
            "volume='if(between(t,1.98,2.12),0.8,0)':eval=frame",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(args.output),
        ],
        check=True,
    )
    silent.unlink()
    args.output.with_suffix(".expected.json").write_text(
        json.dumps(
            {
                "cuts": [2, 62 / 30],
                "insert_duration": 2 / 30,
                "motion": "red square moves right; camera itself is static",
                "overlays": ["MOVE RIGHT", "INSERT", "FINAL"],
                "audio": "synthetic sine burst near 2 seconds, no speech",
            },
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
