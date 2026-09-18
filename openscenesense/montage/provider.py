from __future__ import annotations

import base64
import json
import os
import stat
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

DEFAULT_CONFIG = Path.home() / ".config/video-montage/config.env"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = {}
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("API configuration must have permissions 0600")
        config.update({k: v for k, v in dotenv_values(path).items() if v})
    # Deliberately do not read ANTHROPIC_*, OPENAI_* or Hermes/Codex auth files.
    config.update({k: v for k, v in os.environ.items() if k.startswith("MONTAGE_")})
    for field in ("MONTAGE_API_KEY", "MONTAGE_BASE_URL", "MONTAGE_MODEL"):
        if not config.get(field):
            raise ValueError(f"Missing {field} in dedicated video-montage configuration")
    config.setdefault("MONTAGE_API_MODE", "responses")
    if config["MONTAGE_API_MODE"] not in ("responses", "chat_completions"):
        raise ValueError("API mode must be responses or chat_completions")
    url = urlsplit(config["MONTAGE_BASE_URL"])
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("API base URL must not contain credentials, query or fragment")
    if url.scheme != "https" and not (
        url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
    ):
        raise ValueError("Remote API endpoints must use HTTPS")
    return config


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Model returned a non-object JSON response")
    return data


def validate_analysis(data: dict, frame_count: int) -> dict:
    if not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("Missing analysis summary")
    observations = data.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Missing evidence-backed observations")
    for obs in observations:
        if not isinstance(obs, dict) or not isinstance(obs.get("description"), str):
            raise ValueError("Invalid observation")
        ids = obs.get("frames")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i < 1 or i > frame_count for i in ids)
        ):
            raise ValueError("Observation cites a missing evidence frame")
        if obs.get("confidence") not in ("high", "medium", "low"):
            raise ValueError("Invalid confidence")
    for field in ("recreation_steps", "uncertainties"):
        if not isinstance(data.get(field), list) or any(
            not isinstance(x, str) for x in data[field]
        ):
            raise ValueError(f"Missing or invalid {field}")
    return data


class VisionAPI:
    def __init__(self, config: dict):
        self.config = config
        self.calls = 0
        self.usage = []

    @property
    def identity(self) -> dict:
        return {
            k: self.config[k] for k in ("MONTAGE_BASE_URL", "MONTAGE_MODEL", "MONTAGE_API_MODE")
        }

    def request(self, prompt: str, frames: list[tuple[Path, float]]) -> dict:
        mode = self.config["MONTAGE_API_MODE"]
        content = [{"type": "input_text" if mode == "responses" else "text", "text": prompt}]
        for i, (path, timestamp) in enumerate(frames, 1):
            content.append(
                {
                    "type": "input_text" if mode == "responses" else "text",
                    "text": f"Frame {i}; absolute source time {timestamp:.6f} seconds",
                }
            )
            url = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
            if mode == "responses":
                content.append({"type": "input_image", "image_url": url, "detail": "high"})
            else:
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        payload = {"model": self.config["MONTAGE_MODEL"], "stream": True}
        if mode == "responses":
            payload.update(
                {
                    "input": [{"role": "user", "content": content}],
                    "max_output_tokens": 3000,
                    "store": False,
                }
            )
            endpoint = "/responses"
        else:
            payload.update({"messages": [{"role": "user", "content": content}], "max_tokens": 3000})
            endpoint = "/chat/completions"
        headers = {"Authorization": "Bearer " + self.config["MONTAGE_API_KEY"]}
        for attempt in range(2):
            self.calls += 1
            with httpx.Client(
                timeout=httpx.Timeout(180, connect=20), follow_redirects=False
            ) as client:
                with client.stream(
                    "POST",
                    self.config["MONTAGE_BASE_URL"].rstrip("/") + endpoint,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                        time.sleep(2)
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Vision API HTTP {response.status_code}; no model fallback"
                        )
                    text, usage = self._read(response)
            self.usage.append(usage)
            return parse_json(text)
        raise RuntimeError("Vision API retry exhausted")

    @staticmethod
    def _read(response: httpx.Response) -> tuple[str, dict]:
        if "text/event-stream" not in response.headers.get("content-type", ""):
            raw = b""
            for chunk in response.iter_bytes():
                raw += chunk
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError("API response exceeds limit")
            data = json.loads(raw)
            if data.get("choices"):
                choice = data["choices"][0]
                if choice.get("finish_reason") not in (None, "stop"):
                    raise ValueError("API output incomplete")
                return choice["message"]["content"], data.get("usage", {})
            if data.get("status") not in (None, "completed"):
                raise ValueError("API output incomplete")
            return "".join(
                p.get("text", "")
                for o in data.get("output", [])
                for p in o.get("content", [])
                if p.get("type") == "output_text"
            ), data.get("usage", {})
        chunks, usage, complete, size = [], {}, False, 0
        for line in response.iter_lines():
            size += len(line)
            if size > 4 * 1024 * 1024:
                raise ValueError("API stream exceeds limit")
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                # Chat completion still requires an explicit finish_reason=stop.
                continue
            if not raw:
                continue
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "response.output_text.delta":
                chunks.append(event.get("delta", ""))
            elif kind == "response.completed":
                final = event.get("response", {})
                complete = final.get("status") == "completed"
                usage = final.get("usage", {})
                if not chunks:
                    chunks = [
                        p.get("text", "")
                        for o in final.get("output", [])
                        for p in o.get("content", [])
                        if p.get("type") == "output_text"
                    ]
            elif kind in ("error", "response.failed", "response.incomplete"):
                raise ValueError("API generation failed or was truncated")
            for choice in event.get("choices", []):
                chunks.append(choice.get("delta", {}).get("content") or "")
                if choice.get("finish_reason") == "stop":
                    complete = True
                elif choice.get("finish_reason"):
                    raise ValueError("API generation was truncated")
            if event.get("usage"):
                usage = event["usage"]
        if not complete:
            raise ValueError("API stream ended without confirmed completion")
        return "".join(chunks), usage
