"""Checker, failure report, and repair loop with a fake (offline) code generator."""

import json
from pathlib import Path

import pytest

from core_robot.config import exec_seeds
from core_robot.demos.tasks import TASKS_BY_ID
from core_robot.planning.schema import GoalAtom
from core_robot.repair.loop import run_with_repair
from core_robot.verify.checker import check, goal_error_cm
from core_robot.verify.report import build_failure_report

T1 = TASKS_BY_ID["t1_red_in_red_bin"]
SEED = exec_seeds()[0]

WRONG = """
def run(robot, scene):
    robot.pick("red_block")
    robot.place("red_block", "blue_bin")
"""
RIGHT = """
def run(robot, scene):
    robot.pick("red_block")
    robot.place("red_block", "red_bin")
"""


class FakeGenerator:
    """Wrong bin on attempt 1, correct code from attempt 2 on. Records what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, task, plan, attempt, feedback=None, image_path=None):
        self.calls.append({"attempt": attempt, "feedback": feedback, "image_path": image_path})
        return WRONG if attempt == 1 else RIGHT


@pytest.fixture(scope="module")
def plan():
    from core_robot.sim.env import TabletopEnv

    env = TabletopEnv()
    scene = T1.scene_builder(env, SEED)
    env.close()
    return T1.ground_truth_plan(scene)


@pytest.fixture(scope="module")
def runs(plan, tmp_path_factory):
    out = {}
    for cond in ("A", "B", "C"):
        gen = FakeGenerator()
        log = run_with_repair(T1, plan, SEED, cond, gen, out_dir=tmp_path_factory.mktemp(cond), record=False)
        out[cond] = (log, gen)
    return out


def test_condition_a_one_attempt_fails(runs):
    log, gen = runs["A"]
    assert not log["success"] and log["attempts_used"] == 1
    assert [c["attempt"] for c in gen.calls] == [1]


@pytest.mark.parametrize("cond", ["B", "C"])
def test_conditions_b_c_fail_then_pass(runs, cond):
    log, gen = runs[cond]
    outcomes = [a["outcome"]["success"] for a in log["attempts"]]
    assert outcomes == [False, True], outcomes
    assert log["success"] and log["attempts_used"] == 2
    assert gen.calls[0]["feedback"] is None
    fb = gen.calls[1]["feedback"]
    assert "red_block in red_bin" in fb and "cm from goal" in fb and fb.startswith("Summary: FAILED")


def test_condition_b_gets_no_image_and_c_gets_frame(runs):
    assert runs["B"][1].calls[1]["image_path"] is None
    img = runs["C"][1].calls[1]["image_path"]
    assert img and img.endswith("final_angled.png") and Path(img).exists()


def test_every_attempt_starts_from_fresh_reset(runs):
    a1, a2 = runs["B"][0]["attempts"]
    assert a1["initial_state"] == a2["initial_state"]
    # attempt 1 left red_block in the blue bin; attempt 2 did not start from there
    moved_to = a1["final_state"]["red_block"]["position"]
    assert a2["initial_state"]["red_block"]["position"] != moved_to
    assert abs(a2["initial_state"]["red_block"]["position"][0] - moved_to[0]) > 0.02 or \
        abs(a2["initial_state"]["red_block"]["position"][1] - moved_to[1]) > 0.02


def test_run_log_written_with_code_report_outcome(runs):
    log, _ = runs["C"]
    saved = json.loads(Path(log["log_path"]).read_text())
    assert saved["condition"] == "C" and saved["success"]
    assert [a["code"] for a in saved["attempts"]] == [WRONG, RIGHT]
    assert saved["attempts"][0]["report"].startswith("Summary: FAILED")
    assert saved["attempts"][1]["image_given"].endswith("final_angled.png")


def test_bad_condition_rejected(plan):
    with pytest.raises(ValueError):
        run_with_repair(T1, plan, SEED, "D", FakeGenerator())


# ---- checker and report units ----

def st(**pos):
    kinds = {"red_block": "block", "green_block": "block", "blue_block": "block", "red_bin": "bin", "blue_bin": "bin"}
    return {n: {"position": list(p), "orientation": [0, 0, 0, 1], "color": n.split("_")[0], "kind": kinds[n]}
            for n, p in pos.items()}


def test_checker_error_cm():
    init = st(red_block=(0.4, 0.1, 0.02), green_block=(0.5, -0.2, 0.02), blue_block=(0.6, 0.2, 0.02),
              red_bin=(0.5, 0.25, 0.0))
    final = st(red_block=(0.5, 0.0, 0.02), green_block=(0.5, -0.2, 0.02), blue_block=(0.6, 0.2, 0.02),
               red_bin=(0.5, 0.25, 0.0))
    g = lambda s, r, t: GoalAtom(subject=s, relation=r, target=t)
    assert goal_error_cm(g("red_block", "in", "red_bin"), final, init) == 25.0
    assert goal_error_cm(g("green_block", "on", "red_block"), final, init) == pytest.approx(20.4, abs=0.1)
    assert goal_error_cm(g("red_block", "left_of", "blue_block"), final, init) == pytest.approx(23.0, abs=0.1)
    assert goal_error_cm(g("blue_block", "left_of", "red_block"), final, init) == 0.0


def test_report_names_first_failed_primitive_and_disturbed_block():
    plan = T1.ground_truth_plan(st(red_block=(0.4, 0.1, 0.02), red_bin=(0.5, 0.25, 0.0), blue_bin=(0.5, -0.25, 0.0)))
    init = st(red_block=(0.4, 0.1, 0.02), red_bin=(0.5, 0.25, 0.0), blue_bin=(0.5, -0.25, 0.0))
    result = {
        "ok": True, "exception": None, "final_state": init, "initial_state": init, "frames": {"angled": "/x.png"},
        "call_log": [
            {"primitive": "pick", "args": {"obj": "red_block"}, "ok": True, "reason": ""},
            {"primitive": "place", "args": {"obj": "red_block", "target": "green_bin"}, "ok": False,
             "reason": "unknown target 'green_bin'"},
        ],
    }
    chk = check(T1, init, init, plan)
    text = build_failure_report(result, chk)
    assert text.splitlines()[0].startswith("Summary: FAILED (place failed)")
    assert "call #2 place('red_block', 'green_bin') -> unknown target 'green_bin'" in text
    assert "red_block in red_bin:" in text and "cm from goal" in text
    assert "angled: /x.png" in text


def test_report_on_timeout_has_no_state():
    plan = T1.ground_truth_plan(st(red_block=(0.4, 0.1, 0.02), red_bin=(0.5, 0.25, 0.0), blue_bin=(0.5, -0.25, 0.0)))
    result = {"ok": False, "timed_out": True, "exception": "timeout: run(robot, scene) exceeded 60 s",
              "final_state": None, "initial_state": None, "call_log": [], "frames": {}}
    chk = check(T1, None, None, plan)
    assert not chk["success"] and chk["per_goal"][0]["error_cm"] is None
    text = build_failure_report(result, chk)
    assert "run timed out" in text and "Exception: timeout" in text and "not evaluated" in text
