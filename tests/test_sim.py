"""Simulator determinism and primitive return-value contracts."""

import math

import pytest

from core_robot.sim.env import OBJECTS, TabletopEnv, in_bin
from core_robot.sim.primitives import Robot


@pytest.fixture(scope="module")
def env():
    e = TabletopEnv()
    yield e
    e.close()


def positions(state):
    return {k: v["position"] for k, v in state.items()}


def test_reset_same_seed_same_layout(env):
    a = env.reset(3)
    b = env.reset(3)
    assert positions(a) == positions(b)
    assert [v["orientation"] for v in a.values()] == [v["orientation"] for v in b.values()]


def test_reset_different_seed_different_layout(env):
    assert positions(env.reset(1)) != positions(env.reset(2))


def test_scene_state_schema(env):
    s = env.reset(0)
    assert set(s) == {o.name for o in OBJECTS}
    for name, v in s.items():
        assert set(v) == {"position", "orientation", "color", "kind"}
        assert len(v["position"]) == 3 and len(v["orientation"]) == 4
        assert name.startswith(v["color"]) or v["kind"] == "mug"


@pytest.mark.parametrize("seed", range(5))
def test_no_overlap(env, seed):
    s = env.reset(seed)
    specs = {o.name: o for o in OBJECTS}
    names = list(s)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = s[a]["position"], s[b]["position"]
            assert math.hypot(pa[0] - pb[0], pa[1] - pb[1]) >= specs[a].radius + specs[b].radius


def assert_result(r):
    assert set(r) == {"ok", "reason", "details"}
    assert isinstance(r["ok"], bool) and isinstance(r["reason"], str) and isinstance(r["details"], dict)


def test_primitives_success_path(env):
    env.reset(0)
    robot = Robot(env)
    for r in (robot.move_above("red_block"), robot.pick("red_block"), robot.place("red_block", "red_bin")):
        assert_result(r)
        assert r["ok"], r
    assert in_bin(env.scene_state(), "red_block", "red_bin")
    assert [c["primitive"] for c in robot.call_log] == ["move_above", "pick", "place"]


def test_move_above_point(env):
    env.reset(0)
    r = Robot(env).move_above((0.5, 0.1))
    assert r["ok"], r
    x, y, _ = r["details"]["gripper_position"]
    assert abs(x - 0.5) < 0.01 and abs(y - 0.1) < 0.01


def test_primitives_fail_without_raising(env):
    env.reset(0)
    robot = Robot(env)
    cases = [
        robot.pick("purple_block"),
        robot.pick("red_bin"),
        robot.place("red_block", "red_bin"),  # nothing held
        robot.move_above("nowhere"),
        robot.move_above((5.0, 5.0)),  # out of reach
        robot.move_above("not a point"),
    ]
    for r in cases:
        assert_result(r)
        assert r["ok"] is False and r["reason"]
    assert len(robot.call_log) == len(cases)


def test_pick_twice_and_wrong_place(env):
    env.reset(0)
    robot = Robot(env)
    assert robot.pick("red_block")["ok"]
    again = robot.pick("blue_block")
    assert not again["ok"] and "already holding" in again["reason"]
    wrong = robot.place("blue_block", "red_bin")
    assert not wrong["ok"]
    released = robot.open_gripper()
    assert released["ok"] and released["details"]["released"] == "red_block"
    assert robot.held is None


def test_place_on_point(env):
    env.reset(0)
    robot = Robot(env)
    robot.pick("green_block")
    r = robot.place("green_block", (0.36, -0.12))  # free table spot for seed 0
    assert_result(r)
    assert r["ok"], r


def test_arm_home_and_view_clear_after_place(env):
    import pybullet as p

    from core_robot.sim.env import CAMERAS, WORKSPACE_X

    env.reset(0)
    robot = Robot(env)
    robot.pick("red_block")
    assert robot.place("red_block", "red_bin")["ok"]
    # Every arm link is behind the near edge of the object workspace.
    for link in range(-1, 12):
        assert p.getAABB(env.robot, link, physicsClientId=env.cid)[1][0] < WORKSPACE_X[0]
    # A ray from the overhead camera to each object's top hits that object first, not the arm.
    eye = CAMERAS["overhead"]["eye"]
    for name, body in env.bodies.items():
        lo, hi = env.aabb(name)
        top = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] - 0.002]
        hit = p.rayTest(eye, top, physicsClientId=env.cid)[0][0]
        assert hit != env.robot, f"arm blocks overhead view of {name}"
