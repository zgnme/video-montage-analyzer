"""Bounded parallel map, independent review, explicit unresolved outcomes."""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .cli import fingerprint
from .provider import VisionAPI
from .quality_plan import candidate, review_window
from .quality_schema import (
    PROMPT_VERSION,
    reconcile,
    review_prompt,
    validate_review,
    validate_window,
    window_prompt,
)
from .quality_store import Store


class Runner:
    def __init__(self, args, plan, config, transcript, api_factory=VisionAPI):
        self.args, self.plan, self.config = args, plan, config
        self.transcript = transcript
        self.api_factory = api_factory
        self.identity = api_factory(config).identity
        self.run_key = fingerprint(
            {
                "plan": plan["plan_key"],
                "provider": self.identity,
                "prompt": PROMPT_VERSION,
                "transcript": transcript,
                "output_tokens": config.get("MONTAGE_MAX_OUTPUT_TOKENS", "8192"),
            }
        )
        self.store = Store(
            args.output / "precision.sqlite", self.run_key, args.max_calls, args.max_tokens
        )
        self.stop = threading.Event()
        self.failures = []
        self.started = time.monotonic()
        self.context = {
            "duration": plan["metadata"]["duration"],
            "source_offset": plan["settings"]["source_offset"],
            "purpose": "повторить монтаж референса",
            "all_input_frames": True,
        }

    def task(self, name, window, prompt, validator):
        key = fingerprint({"name": name, "window": window, "prompt": prompt, "run": self.run_key})
        return {
            "name": name,
            "key": key,
            "window": window,
            "prompt": prompt,
            "validator": validator,
        }

    def execute(self, task):
        cached = self.store.cached(task["key"])
        if cached is not None:
            return task["validator"](cached, task["window"])
        last_error = None
        for retry in range(3):
            if self.stop.is_set():
                raise RuntimeError("Run stopped; completed requests retained")
            # A conservative planning reservation, not a provider-enforced billing cap.
            reserve = (
                len(task["window"]["frames"]) * 4096
                + int(self.config.get("MONTAGE_MAX_OUTPUT_TOKENS", "8192"))
                + len(task["prompt"]) * 4
            )
            attempt = self.store.begin(task["key"], reserve)
            api = self.api_factory(self.config, max_attempts=1)
            value, error = None, None
            try:
                raw = api.request(
                    task["prompt"],
                    [(self.args.output / f["path"], f["time"]) for f in task["window"]["frames"]],
                )
                value = task["validator"](raw, task["window"])
            except Exception as exc:
                # Provider exceptions contain only sanitized errors, never request bodies/keys.
                error = f"{type(exc).__name__}: {exc}"[:300]
                last_error = exc
            finally:
                self.store.finish(
                    attempt, task["key"], value, api.usage[-1] if api.usage else {}, error
                )
            if error is None:
                return value
            if self.stop.wait(min(2**retry, 4)):
                break
        raise RuntimeError(f"{task['name']}: {last_error}")

    def stage(self, tasks, label):
        results = {}
        pending = iter(tasks)
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            active = {}

            def submit():
                if self.stop.is_set():
                    return False
                task = next(pending, None)
                if task is None:
                    return False
                active[pool.submit(self.execute, task)] = task
                return True

            for _ in range(self.args.workers):
                submit()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    try:
                        results[task["name"]] = future.result()
                        print(f"{label}: {len(results)}/{len(tasks)} ({task['name']})", flush=True)
                    except Exception as exc:
                        self.failures.append({"task": task["name"], "reason": str(exc)[:300]})
                        self.stop.set()
                    submit()
        return results

    def analyze(self):
        primary = []
        for w in self.plan["windows"]:
            speech = [
                s
                for s in self.transcript
                if s["end"] > w["frames"][0]["time"] and s["start"] <= w["frames"][-1]["time"]
            ]
            primary.append(
                self.task(w["id"], w, window_prompt(w, speech, {}, self.context), validate_window)
            )
        # Fail fast on an unavailable model before opening ten paid requests.
        first = self.stage(primary[:1], "Pilot")
        first.update(self.stage(primary[1:], "Windows"))
        candidates = {c["id"]: c for c in self.plan["candidates"]}
        votes = {c: [] for c in candidates}
        events, windows = [], []
        for task in primary:
            value = first.get(task["name"])
            if value is None:
                continue
            w = task["window"]
            windows.append(
                {"id": task["name"], "start": w["core_start"], "end": w["core_end"], **value}
            )
            for verdict in value["boundaries"]:
                votes[verdict["id"]].append(self.attach(verdict, w, "primary"))
            for proposed in value["extra_boundaries"]:
                i = w["frames"][proposed["after_ref"] - 1]["index"]
                if not w["core_start_index"] <= i < w["core_stop_index"]:
                    continue
                c = candidate(i, self.plan["frames"], "model_proposal")
                candidates.setdefault(c["id"], c)
                votes.setdefault(c["id"], [])
            for e in value["events"]:
                a, b = w["frames"][e["start_ref"] - 1], w["frames"][e["end_ref"] - 1]
                if b["index"] < w["core_start_index"] or a["index"] >= w["core_stop_index"]:
                    continue
                events.append(
                    {
                        **e,
                        "start": a["time"],
                        "end": b["time"],
                        "window": w["id"],
                        "first_frame": a,
                        "last_frame": b,
                        "status": "model_observation",
                    }
                )
        for round_number in (1, 2, 3):
            tasks = []
            for cid, c in sorted(candidates.items()):
                # At most three votes; omitted primary verdicts still receive two reviews.
                if len(votes[cid]) >= 3 or reconcile(c, votes[cid])["status"] == "reviewed":
                    continue
                w = review_window(self.plan, c, radius=6 if round_number == 1 else 10)
                tasks.append(
                    self.task(
                        f"review{round_number}:{cid}",
                        w,
                        review_prompt(w, third=round_number > 1),
                        validate_review,
                    )
                )
            output = self.stage(tasks, f"Review {round_number}")
            for task in tasks:
                if task["name"] in output:
                    v = output[task["name"]]["boundary"]
                    votes[v["id"]].append(self.attach(v, task["window"], f"review{round_number}"))
        boundaries = [reconcile(c, votes[cid]) for cid, c in sorted(candidates.items())]
        indices = [0] + [c["index"] for c in boundaries if c["kind"] in ("hard_cut", "crop_jump")]
        indices.append(len(self.plan["frames"]))
        shots = []
        for number, (start, stop) in enumerate(zip(indices, indices[1:], strict=False), 1):
            first_frame = self.plan["frames"][start]
            end = (
                self.plan["frames"][stop]["time"]
                if stop < len(self.plan["frames"])
                else self.plan["settings"]["source_offset"] + self.plan["metadata"]["duration"]
            )
            shots.append(
                {
                    "id": f"s{number:05d}",
                    "start": first_frame["time"],
                    "end": end,
                    "duration": end - first_frame["time"],
                    "first_frame_index": start,
                    "last_frame_index": stop - 1,
                }
            )
        complete = len(windows) == len(primary) and not self.stop.is_set()
        unresolved = [c["id"] for c in boundaries if c["status"] != "reviewed"]
        return {
            "version": PROMPT_VERSION,
            "run_key": self.run_key,
            "provider": self.identity,
            "status": "complete" if complete else "partial",
            "quality_status": "needs_review"
            if unresolved
            else ("reviewed" if complete else "incomplete"),
            "completed_windows": len(windows),
            "total_windows": len(primary),
            "source_frames": len(self.plan["frames"]),
            "analyzed_frames": sum(
                t["window"]["core_stop_index"] - t["window"]["core_start_index"]
                for t in primary
                if t["name"] in first
            ),
            "boundaries": boundaries,
            "shots": shots,
            "events": events,
            "windows": windows,
            "unresolved_candidates": unresolved,
            "failures": self.failures,
            "usage": self.store.summary(),
            "elapsed_seconds_this_run": time.monotonic() - self.started,
            "limitations": [
                "Согласие двух проверок не доказывает правильность классификации.",
                "Все кадры переданы модели; внимание модели к каждому кадру не гарантируется.",
                "Звук не классифицируется. Транскрипт содержит только речь.",
                "События внутри планов — наблюдения одного прохода; "
                "границы проверяются независимо.",
                "Разрешение JPEG ограничено frame-size; "
                "исходные пиксели доступны в исходном видео.",
            ],
        }

    @staticmethod
    def attach(verdict, window, stage):
        return {
            **verdict,
            "stage": stage,
            "evidence": [window["frames"][i - 1] for i in verdict["evidence_refs"]],
        }
