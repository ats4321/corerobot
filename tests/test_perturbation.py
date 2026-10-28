"""calibration_offset perturbation: determinism, off by default, hidden from code and prompts,
true state for the checker, and paired attempt 1 with the clean environment."""

import json
import math

import pytest
from _fakes import ScriptedTransport, code_reply, user_text

from core_robot.codegen.api_doc import primitives_reference
from core_robot.codegen.muse_generator import MuseGenerator
from core_robot.codegen.prompt import build_codegen_prompt
from core_robot.codegen.sandbox import run_code
from core_robot.codegen.scene import exec_initial_state, read_only_scene
from core_robot.config import PILOT_SEEDS, Settings, exec_seeds
from core_robot.demos.tasks import MAX_LAYOUT_TRIES, TASKS_BY_ID
from core_robot.meta_client import MetaClient
from core_robot.repair.loop import run_with_repair
from core_robot.sim.env import TabletopEnv
from core_robot.sim.perturbation import calibration_offset
from core_robot.sim.primitives import Robot
from core_robot.verify.checker import check

T3 = TASKS_BY_ID["t3_stack_green_on_red"]
STACK = 'def run(robot, scene):\n    robot.pick("green_block")\n    r = robot.place("green_block", "red_block")\n' \
        '    print("place", r["ok"], r["details"]["object_position"])\n'


def test_offset_is_deterministic_per_seed_with_fixed_magnitude():
    a, b = calibration_offset(100, 3), calibration_offset(100, 3)
    assert a == b and math.hypot(*a) == pytest.approx(0.03)
    assert calibration_offset(101, 3) != a  # direction depends on the seed
    dirs = [math.atan2(*reversed(calibration_offset(s, 4))) for s in range(100, 105)]
    assert len({round(d, 6) for d in dirs}) == 5
    da, db = calibration_offset(100, 2), calibration_offset(100, 5)  # same direction, scaled
    assert math.atan2(da[1], da[0]) == pytest.approx(math.atan2(db[1], db[0]))


def _place_green_on_red(offset):
    env = TabletopEnv()
    env.reset(0, objects=["red_block", "green_block", "blue_block", "yellow_block"])
    robot = Robot(env, calibration_offset=offset)
    robot.pick("green_block")
    r = robot.place("green_block", "red_block")
    state = env.scene_state()
    env.close()
    return r, state, robot.call_log


def test_off_by_default_and_zero_offset_equals_clean():
    clean = _place_green_on_red(None)
    zero = _place_green_on_red((0.0, 0.0))
    assert clean[0] == zero[0] and clean[1] == zero[1]  # identical result and identical final state
    assert all("true" not in c for c in clean[2] + zero[2])


def test_perturbed_place_reports_belief_while_true_state_differs():
    dx, dy = calibration_offset(100, 5)
    r, state, log = _place_green_on_red((dx, dy))
    true_pos = state["green_block"]["position"]
    believed = r["details"]["object_position"]
    assert r["ok"] is True and r["details"]["on_target"] is True  # the robot thinks it worked
    red = state["red_block"]["position"]
    assert believed[:2] == pytest.approx(red[:2], abs=1e-3)  # belief: the commanded point (on top of red)
    assert math.hypot(believed[0] - true_pos[0], believed[1] - true_pos[1]) > 0.03  # truth: off by the offset
    assert log[-1]["true"]["object_position"] == pytest.approx(true_pos, abs=1e-4)  # logs keep the truth
    assert "true" not in r  # but the caller never sees it


def test_sandbox_returns_belief_to_code_and_checker_uses_true_state(tmp_path):
    seed = PILOT_SEEDS[0]
    r = run_code(STACK, T3.id, seed, tmp_path, record=False, calibration_offset_cm=5)
    assert r["ok"], r["exception"]
    printed = r["stdout"].split()
    assert printed[:2] == ["place", "True"]
    believed = json.loads(" ".join(printed[2:]))
    true_green = r["final_state"]["green_block"]["position"]
    red = r["final_state"]["red_block"]["position"]
    assert math.hypot(believed[0] - red[0], believed[1] - red[1]) < 0.01  # belief: right on top of red
    assert math.hypot(true_green[0] - red[0], true_green[1] - red[1]) > 0.03  # truth: 5 cm off
    plan = T3.ground_truth_plan(r["initial_state"])
    c = check(T3, r["final_state"], r["initial_state"], plan)
    assert not c["success"] and not c["per_goal"][0]["ok"]
    place = [x for x in r["call_log"] if x["primitive"] == "place"][0]
    assert place["ok"] and place["true"]["object_position"] == pytest.approx(true_green, abs=1e-3)


def test_failure_report_gives_error_direction(tmp_path):
    from core_robot.verify.report import build_failure_report

    seed = PILOT_SEEDS[0]
    r = run_code(STACK, T3.id, seed, tmp_path, record=False, calibration_offset_cm=5)
    c = check(T3, r["final_state"], r["initial_state"], T3.ground_truth_plan(r["initial_state"]))
    vec = c["per_goal"][0]["error_vec_cm"]
    dx, dy = calibration_offset(seed, 5)
    if math.hypot(*vec) > 1.0:  # the block may slide off the stack; the error points along the offset
        assert (vec[0] * dx + vec[1] * dy) > 0
    text = build_failure_report(r, c)
    assert f"dx {vec[0]:+.1f} cm, dy {vec[1]:+.1f} cm" in text


def test_perturbation_never_mentioned_to_the_model():
    scene = read_only_scene(exec_initial_state(T3, PILOT_SEEDS[0]))
    plan = T3.ground_truth_plan(exec_initial_state(T3, PILOT_SEEDS[0]))
    _, text = build_codegen_prompt(plan, scene, primitives_reference())
    for doc in (primitives_reference(), text):
        low = doc.lower()
        for word in ("calibration", "perturb", "offset", "miscalibrat"):
            assert word not in low, f"'{word}' leaks into what the model sees"


def test_perturbed_attempt_one_shares_request_key_with_clean(tmp_path):
    """Fails loudly if the environment ever changes the first-attempt prompt."""
    s = Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")
    t = ScriptedTransport(lambda m, p, b: code_reply(STACK.replace("    print(", "    0 and print(")))
    gen = MuseGenerator(MetaClient(settings=s, transport=t))
    seed = PILOT_SEEDS[1]
    plan = T3.ground_truth_plan(exec_initial_state(T3, seed))
    clean = run_with_repair(T3, plan, seed, "A", gen, out_dir=tmp_path / "clean", record=False)
    pert = run_with_repair(T3, plan, seed, "A", gen, out_dir=tmp_path / "pert", record=False, calibration_offset_cm=4)
    k_clean = clean["attempts"][0]["generation"]["request_key"]
    k_pert = pert["attempts"][0]["generation"]["request_key"]
    assert k_clean == k_pert, ("perturbed attempt-1 prompt differs from the clean one: the perturbation "
                               "leaked into the code-generation request")
    assert len(t.calls) == 1  # the perturbed run reused the clean generation
    assert clean["attempts"][0]["code"] == pert["attempts"][0]["code"]


def test_pilot_seeds_never_overlap_evaluation_or_demo_seeds():
    demo_seeds = {0, 1}
    assert not set(PILOT_SEEDS) & set(exec_seeds(50)) and not set(PILOT_SEEDS) & demo_seeds
    layouts = lambda seeds: {x * 1000 + k for x in seeds for k in range(MAX_LAYOUT_TRIES)}
    assert not layouts(PILOT_SEEDS) & layouts(exec_seeds(50))


def test_offset_resolution_rules(monkeypatch):
    from core_robot import config
    from core_robot.eval.run_eval import resolve_offset

    monkeypatch.setattr(config, "CALIBRATION_OFFSET_CM", None)
    assert resolve_offset("clean", None, {100}) is None
    assert resolve_offset("perturbed", 3, set(PILOT_SEEDS)) == 3
    with pytest.raises(SystemExit, match="frozen"):
        resolve_offset("perturbed", 3, {100})  # evaluation seeds before the pilot froze m
    with pytest.raises(SystemExit):
        resolve_offset("clean", 3, {100})
    monkeypatch.setattr(config, "CALIBRATION_OFFSET_CM", 4)
    assert resolve_offset("perturbed", None, {100, 101}) == 4
    with pytest.raises(SystemExit, match="frozen offset"):
        resolve_offset("perturbed", 5, {100})
