from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

from openscenesense.montage.quality_plan import candidate, make_windows, review_window
from openscenesense.montage.quality_report import render
from openscenesense.montage.quality_runner import Runner
from openscenesense.montage.quality_schema import reconcile, validate_window
from openscenesense.montage.quality_store import BudgetReached, Store


def plan_fixture(tmp_path):
    frames = [
        {
            "index": i,
            "time": 20 + i * 0.04 + (0.013 if i > 12 else 0),
            "path": f"frames/{i}.jpg",
            "sha256": "fixture",
        }
        for i in range(40)
    ]
    candidates = [candidate(i, frames) for i in (1, 8, 9, 17, 39)]
    windows = make_windows(frames, candidates, 1.613, 20, core_frames=8, context=2)
    return {
        "plan_key": "fixture",
        "frames": frames,
        "candidates": candidates,
        "windows": windows,
        "metadata": {"duration": 1.613},
        "settings": {"source_offset": 20},
    }


def args_fixture(tmp_path, workers=10, max_calls=300):
    return SimpleNamespace(
        output=tmp_path, workers=workers, max_calls=max_calls, max_tokens=20_000_000
    )


def verdict(c, kind="hard_cut"):
    return {
        "id": c["id"],
        "kind": kind,
        "confidence": "high",
        "before": "Красный кадр",
        "after": "Синий кадр",
        "description": "Резкая смена",
        "evidence_refs": [c["left_ref"], c["right_ref"]],
    }


class FakeAPI:
    guard = threading.Lock()
    requests = []
    active = peak = 0
    omit = False
    malformed = False

    def __init__(self, config, max_attempts=2):
        self.usage = []

    @property
    def identity(self):
        return {"MONTAGE_MODEL": "fixture"}

    def request(self, prompt, frames):
        with self.guard:
            type(self).active += 1
            type(self).peak = max(self.active, self.peak)
            self.requests.append(
                hashlib.sha256(
                    (prompt + repr([(str(p.name), t) for p, t in frames])).encode()
                ).hexdigest()
            )
        time.sleep(0.01)
        with self.guard:
            type(self).active -= 1
        self.usage.append(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 40},
            }
        )
        if self.malformed:
            raise ValueError("Invalid fixture JSON")
        if "\nДанные: " in prompt:
            data = json.loads(prompt.split("\nДанные: ")[-1])
            return {
                "summary": "Тест",
                "boundaries": [] if self.omit else [verdict(c) for c in data["candidates"]],
                "events": [],
                "extra_boundaries": [],
                "uncertainties": [],
                "recreation_steps": ["Прямая склейка"],
            }
        c = json.loads(prompt.split("\nКандидат: ")[-1])
        return {"boundary": verdict(c)}


@pytest.fixture(autouse=True)
def reset_fake():
    FakeAPI.requests = []
    FakeAPI.active = FakeAPI.peak = 0
    FakeAPI.omit = FakeAPI.malformed = False


def test_vfr_coverage_single_frame_insert_and_seam_anchors(tmp_path):
    plan = plan_fixture(tmp_path)
    owned = [i for w in plan["windows"] for i in range(w["core_start_index"], w["core_stop_index"])]
    assert owned == list(range(40))
    assert len({c["id"] for w in plan["windows"] for c in w["candidates"]}) == 5
    for w in plan["windows"]:
        for c in w["candidates"]:
            a, b = [w["frames"][c[k] - 1] for k in ("left_ref", "right_ref")]
            assert b["index"] == a["index"] + 1
            assert c["time"] == b["time"]
    assert plan["frames"][13]["time"] == 20.533


def test_missing_candidate_is_recoverable_but_invalid_anchor_is_rejected(tmp_path):
    w = plan_fixture(tmp_path)["windows"][0]
    missing = validate_window({"summary": "x"}, w)
    assert missing["missing_candidates"] == [w["candidates"][0]["id"]]
    v = verdict(w["candidates"][0])
    v["evidence_refs"] = [1]
    with pytest.raises(ValueError, match="both adjacent"):
        validate_window({"summary": "x", "boundaries": [v]}, w)


def test_parallelism_preserves_payloads_and_resume_costs_zero(tmp_path):
    results, hashes = [], []
    for workers in (1, 10):
        output = tmp_path / str(workers)
        output.mkdir()
        FakeAPI.requests = []
        runner = Runner(args_fixture(output, workers), plan_fixture(output), {}, [], FakeAPI)
        result = runner.analyze()
        hashes.append(sorted(FakeAPI.requests))
        results.append(result)
        before = len(FakeAPI.requests)
        again = runner.analyze()
        assert len(FakeAPI.requests) == before
        assert again["usage"] == result["usage"]
        runner.store.close()
    assert hashes[0] == hashes[1]
    assert results[0]["boundaries"] == results[1]["boundaries"]
    assert results[1]["quality_status"] == "reviewed"
    assert results[1]["analyzed_frames"] == 40
    assert FakeAPI.peak > 1


def test_omitted_primary_verdicts_get_two_independent_reviews(tmp_path):
    FakeAPI.omit = True
    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], FakeAPI)
    result = runner.analyze()
    assert result["quality_status"] == "reviewed"
    assert all(
        [v["stage"] for v in c["votes"]] == ["review1", "review2"] for c in result["boundaries"]
    )
    assert result["usage"]["attempts"] == 15
    runner.store.close()


def test_disagreement_is_never_silently_promoted(tmp_path):
    plan = plan_fixture(tmp_path)
    c = plan["candidates"][0]
    local = review_window(plan, c)["candidates"][0]
    votes = [verdict(local, kind) for kind in ("hard_cut", "camera_motion", "uncertain")]
    assert reconcile(c, votes)["status"] == "needs_review"
    votes[1] = verdict(local, "crop_jump")
    assert reconcile(c, votes)["status"] == "reviewed"


def test_cumulative_budget_and_interrupted_usage_survive_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite"
    store = Store(path, "run", 1, 1000)
    store.begin("one", 500)
    store.close()
    store = Store(path, "run", 1, 1000)
    assert store.summary()["unknown_usage_attempts"] == 1
    assert store.summary()["accounted_tokens"] == 500
    with pytest.raises(BudgetReached):
        store.begin("two", 100)
    store.close()
    with pytest.raises(ValueError, match="Different model"):
        Store(path, "changed", 99, 10000)


def test_malformed_paid_answers_are_counted_and_stop_at_pilot(tmp_path, monkeypatch):
    FakeAPI.malformed = True
    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], FakeAPI)
    monkeypatch.setattr(runner.stop, "wait", lambda delay: False)
    result = runner.analyze()
    assert result["status"] == "partial"
    assert result["usage"]["attempts"] == 3
    assert result["usage"]["input_tokens"] == 300
    assert result["analyzed_frames"] == 0
    runner.store.close()


def test_budget_stop_resume_and_report_escape(tmp_path):
    plan = plan_fixture(tmp_path)
    runner = Runner(args_fixture(tmp_path, max_calls=1), plan, {}, [], FakeAPI)
    first = runner.analyze()
    assert first["status"] == "partial"
    assert first["usage"]["attempts"] == 1
    runner.store.close()
    runner = Runner(args_fixture(tmp_path), plan, {}, [], FakeAPI)
    result = runner.analyze()
    assert result["status"] == "complete"
    assert result["usage"]["attempts"] == 10
    result["boundaries"][0]["description"] = "<script>alert('bad')</script>"
    render(plan, result, tmp_path)
    assert "<script>" not in (tmp_path / "report.html").read_text()
    assert "&lt;script&gt;" in (tmp_path / "report.html").read_text()
    assert (tmp_path / "montage.csv").exists()
    runner.store.close()


def test_disagreement_triggers_third_vote(tmp_path):
    class DisagreeingAPI(FakeAPI):
        def request(self, prompt, frames):
            result = super().request(prompt, frames)
            if "boundaries" in result:
                for row in result["boundaries"]:
                    row["kind"] = "camera_motion"
            return result

    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], DisagreeingAPI)
    result = runner.analyze()
    assert result["quality_status"] == "reviewed"
    assert result["usage"]["attempts"] == 15
    assert all(len(c["votes"]) == 3 for c in result["boundaries"])
    assert all(c["kind"] == "hard_cut" for c in result["boundaries"])
    runner.store.close()


def test_model_proposal_requires_two_focused_checks(tmp_path):
    class ProposingAPI(FakeAPI):
        def request(self, prompt, frames):
            result = super().request(prompt, frames)
            if "boundaries" in result and frames[0][1] == 20:
                result["extra_boundaries"] = [{"after_ref": 4, "reason": "Additional cut"}]
            return result

    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], ProposingAPI)
    result = runner.analyze()
    proposed = next(c for c in result["boundaries"] if c["index"] == 3)
    assert proposed["origin"] == "model_proposal"
    assert [v["stage"] for v in proposed["votes"]] == ["review1", "review2"]
    assert proposed["status"] == "reviewed"
    assert sum(s["duration"] for s in result["shots"]) == pytest.approx(1.613)
    runner.store.close()


def test_cancelled_dispatch_retains_completed_request_for_resume(tmp_path):
    runner = None

    class StoppingAPI(FakeAPI):
        def request(self, prompt, frames):
            result = super().request(prompt, frames)
            runner.stop.set()
            return result

    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], StoppingAPI)
    result = runner.analyze()
    assert result["status"] == "partial"
    assert result["usage"]["attempts"] == 1
    runner.store.close()
    runner = Runner(args_fixture(tmp_path), plan_fixture(tmp_path), {}, [], FakeAPI)
    result = runner.analyze()
    assert result["status"] == "complete"
    assert result["usage"]["attempts"] == 10
    runner.store.close()


@pytest.mark.parametrize(
    "usage",
    [
        {"output_tokens": 50},
        {"input_tokens": -1, "output_tokens": 2},
        {"input_tokens": None, "output_tokens": 2},
    ],
)
def test_incomplete_or_invalid_usage_retains_unknown_reservation(tmp_path, usage):
    store = Store(tmp_path / "usage.sqlite", "run", 10, 1000)
    attempt = store.begin("task", 400)
    store.finish(attempt, "task", {"result": "completed"}, usage)
    summary = store.summary()
    assert summary["unknown_usage_attempts"] == 1
    assert summary["accounted_tokens"] == 400
    assert store.cached("task") == {"result": "completed"}
    store.close()
