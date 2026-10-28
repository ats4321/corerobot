"""UI loaders (no browser): demos, run logs, and on-page errors instead of crashes."""

import json

import pytest

from core_robot.eval.report import FAKE_BANNER
from core_robot.ui import app


def test_demos_listed_and_loaded():
    demos = app.list_demos()
    assert "t4_stack_three/seed_0" in demos and len(demos) == 10
    d = app.load_demo("t4_stack_three/seed_0")
    assert d["error"] is None and d["video"].endswith("video.mp4") and d["plan"]["steps"]


@pytest.mark.parametrize("demo_id", ["", "t9_nope/seed_0", "../../etc/seed_0"])
def test_bad_demo_is_an_error_not_a_crash(demo_id):
    d = app.load_demo(demo_id)
    assert d["error"] and d["video"] is None


@pytest.fixture
def run_set(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    d = tmp_path / "runs" / "fake" / "x" / "t1_red_in_red_bin" / "B" / "seed_100"
    d.mkdir(parents=True)
    log = {
        "task_id": "t1_red_in_red_bin", "condition": "B", "exec_seed": 100, "success": True, "attempts_used": 2,
        "run_source": "fake", "plan_source": "ground_truth", "plan": {"steps": []},
        "attempts": [
            {"attempt": 1, "code": "def run(robot, scene):\n    pass\n", "report": "Summary: FAILED ...",
             "video": str(d / "missing.mp4"),
             "outcome": {"success": False, "per_goal": [{"subject": "red_block", "relation": "in",
                                                         "target": "red_bin", "ok": False, "error_cm": 12.5}]}},
            {"attempt": 2, "code": "def run(robot, scene):\n    robot.pick('red_block')\n", "report": "Summary: success",
             "video": None, "outcome": {"success": True, "per_goal": []}},
        ],
    }
    (d / "run.json").write_text(json.dumps(log))
    return "runs/fake/x", d


def test_load_run_shows_attempts_and_fake_banner(run_set):
    rs, _ = run_set
    r = app.load_run(rs, "t1_red_in_red_bin", "B", 100)
    assert r["error"] is None and r["banner"] == FAKE_BANNER
    assert "SUCCESS" in r["summary"] and len(r["attempts"]) == 2
    a1, a2 = r["attempts"]
    assert "red_block in red_bin: NO (12.5 cm off)" in a1["outcome"]
    assert a1["video"] is None  # missing file is dropped, not passed to the player
    assert a2["outcome"].startswith("Attempt 2: success")


@pytest.mark.parametrize("kwargs,needle", [
    ({"seed": 7}, "not an evaluation seed"),
    ({"seed": "abc"}, "must be an integer"),
    ({"condition": "A"}, "No run log"),
    ({"run_set": ""}, "No evaluation runs found"),
    ({"run_set": "runs/../../etc"}, "Invalid run set"),
    ({"run_set": "/etc"}, "Invalid run set"),
    ({"task_id": "../x"}, "Unknown task or condition"),
    ({"condition": "Z"}, "Unknown task or condition"),
])
def test_load_run_errors(run_set, kwargs, needle):
    rs, _ = run_set
    args = {"run_set": rs, "task_id": "t1_red_in_red_bin", "condition": "B", "seed": 100, "replay": True, **kwargs}
    r = app.load_run(**args)
    assert needle in r["error"] and r["attempts"] == []


def test_live_mode_without_key_is_an_error_not_a_call(run_set, monkeypatch):
    from core_robot.config import Settings

    monkeypatch.setattr(app, "load_settings", lambda: Settings(api_key=None, replay=False))
    r = app.load_run("", "t1_red_in_red_bin", "B", 100, replay=False)
    assert "Live mode" in r["error"] and "MODEL_API_KEY" in r["error"] and r["attempts"] == []


def test_corrupt_log_is_an_error(run_set):
    rs, d = run_set
    (d / "run.json").write_text("{not json")
    assert "Could not read" in app.load_run(rs, "t1_red_in_red_bin", "B", 100)["error"]


def test_app_builds():
    assert app.build_app() is not None
