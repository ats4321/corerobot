"""Every task's scripted demonstrator passes its own goal check; the demo set exists on disk."""

import json

import numpy as np
import pytest

from core_robot.demos.demonstrator import demo_dir, run_demo
from core_robot.demos.tasks import TASKS, TASKS_BY_ID, blocks_by_mug_distance
from core_robot.planning.schema import Plan
from core_robot.sim.env import TabletopEnv

DEMO_SET = [(t.id, 0) for t in TASKS] + [("t2_sort_by_color", 1), ("t5_line_up", 1)]


@pytest.fixture(scope="module")
def env():
    e = TabletopEnv()
    yield e
    e.close()


def test_eight_tasks():
    assert len(TASKS) == 8 and len(TASKS_BY_ID) == 8


@pytest.mark.parametrize("task_id,seed", DEMO_SET)
def test_demonstrator_passes_goal_check(env, task_id, seed):
    r = run_demo(TASKS_BY_ID[task_id], seed, env)
    assert r["success"], r["check"]
    assert all(c["ok"] for c in r["call_log"]), r["call_log"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_initial_scene_fails_goal_and_plan_is_valid(env, task):
    scene = task.scene_builder(env, 0)
    assert not task.check(scene, scene)["success"]
    plan = task.ground_truth_plan(scene)
    Plan.model_validate(plan.model_dump())  # round-trips through the validator
    plan.check_scene(scene)


def test_scene_builder_deterministic(env):
    t = TASKS_BY_ID["t8_nearest_to_mug"]
    a, b = t.scene_builder(env, 2), t.scene_builder(env, 2)
    assert {k: v["position"] for k, v in a.items()} == {k: v["position"] for k, v in b.items()}


@pytest.mark.parametrize("seed", range(5))
def test_nearest_to_mug_margin(env, seed):
    scene = TASKS_BY_ID["t8_nearest_to_mug"].scene_builder(env, seed)
    d = blocks_by_mug_distance(scene)
    assert d[1][0] - d[0][0] >= 0.05


def test_overhead_left_is_plus_y(env):
    env.reset(0, objects=[], fixed={"red_block": (0.52, 0.25), "blue_block": (0.52, -0.25)})
    img = env.render("overhead").astype(int)
    red = (img[:, :, 0] > 150) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)
    blue = (img[:, :, 2] > 150) & (img[:, :, 0] < 80)
    w = img.shape[1]
    assert np.nonzero(red)[1].mean() < w / 2 < np.nonzero(blue)[1].mean()


@pytest.mark.parametrize("task_id,seed", DEMO_SET)
def test_demo_files_exist(task_id, seed):
    d = demo_dir(task_id, seed)
    for f in ("video.mp4", "ground_truth_plan.json", "initial_state.json", "final_state.json", "result.json"):
        assert (d / f).exists(), d / f
    assert json.loads((d / "result.json").read_text())["success"]
    Plan.model_validate_json((d / "ground_truth_plan.json").read_text())
