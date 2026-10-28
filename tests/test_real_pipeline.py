"""Real-clip pipeline, offline: normalization, stages, plan-defined scene, repair, budget, outputs."""

import json
import subprocess
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import pytest
from _fakes import ScriptedTransport, code_reply, responses_payload, sam_payload, sam_text, user_parts, user_text

from core_robot.config import DEMOS_DIR, Settings
from core_robot.meta_client import BudgetGuard, MetaClient
from core_robot.perception.sam_tracking import CONCEPTS
from core_robot.prompt_freeze import frozen
from core_robot.real.normalize import ffmpeg_exe, normalize_clip, probe
from core_robot.real.pipeline import estimate_cost, run_pipeline
from core_robot.real.plan_task import JUDGED_AGAINST, PlanTask

NAME_FOR = {c: n for n, c in CONCEPTS.items()}
OBJECTS = ["red_block", "red_bin", "blue_bin"]
BOXES = {"red_block": (300, 250, 330, 282), "red_bin": (380, 220, 500, 330), "blue_bin": (120, 200, 240, 300)}
PLAN = {"objects": [{"name": "red_block", "color": "red", "kind": "block"},
                    {"name": "red_bin", "color": "red", "kind": "bin"},
                    {"name": "blue_bin", "color": "blue", "kind": "bin"}],
        "steps": [{"index": 0, "action": "pick_and_place", "object": "red_block", "destination": "red_bin"}],
        "goal": [{"subject": "red_block", "relation": "in", "target": "red_bin"}]}
WRONG = 'def run(robot, scene):\n    robot.pick("red_block")\n    robot.place("red_block", "blue_bin")\n'
RIGHT = 'def run(robot, scene):\n    robot.pick("red_block")\n    robot.place("red_block", "red_bin")\n'


@pytest.fixture(scope="module")
def phone_clip(tmp_path_factory):
    """22 s, 1280x720, flagged as rotated 90 degrees (like a phone video)."""
    d = tmp_path_factory.mktemp("clip")
    w = iio.get_writer(d / "raw.mp4", fps=5, codec="libx264", pixelformat="yuv420p", macro_block_size=8)
    for i in range(110):
        f = np.full((720, 1280, 3), 90, np.uint8)
        f[:, :200] = (255, 0, 0)
        w.append_data(f)
    w.close()
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                    "-i", str(d / "raw.mp4"), "-c", "copy", str(d / "phone.mp4")], check=True)
    return d / "phone.mp4"


def test_normalize_rotates_trims_downscales(phone_clip, tmp_path):
    assert probe(phone_clip)["rotation_deg"] == 90.0
    n = normalize_clip(phone_clip, tmp_path / "n.mp4")
    assert n["trimmed"] and n["duration_s"] <= 20.05
    assert n["size"] == [540, 960]  # upright portrait, longest side 960
    assert n["bytes"] < 20 * 1024 * 1024
    out = probe(tmp_path / "n.mp4")
    assert out["rotation_deg"] == 0.0  # rotation applied to the pixels, no flag left
    frame = iio.get_reader(tmp_path / "n.mp4").get_data(0)
    assert frame.shape == (960, 540, 3) and frame[-40:].mean(axis=(0, 1))[0] > 200  # left band now at the bottom


def test_estimate_arithmetic():
    e = estimate_cost(20.0)
    assert e["frames_billed_per_sam_video"] == 101
    assert e["worst"]["sam"] == pytest.approx(7 * (2 * 101 * 0.0002 + 12 * 0.0025))
    assert e["worst"]["codegen"] == pytest.approx(3 * (7000 * 1.25e-6 + 12000 * 4.25e-6))
    assert e["worst_total"] > e["expected_total"] > 0


class FakeRealWorld:
    """Naming, SAM, planner and code generation for a clip showing a red block going into a red bin."""

    def __init__(self, plan_replies=None):
        self.plan_replies = list(plan_replies or [json.dumps(PLAN)])
        self.codegen = []

    def __call__(self, method, path, body):
        parts = user_parts(body)
        if body["model"] == "sam-3.1":
            name = NAME_FOR.get(user_text(body).strip())
            if name not in OBJECTS:
                return sam_payload("")
            if any(p["type"] == "input_video" for p in parts):
                return sam_payload(sam_text({k: [(0, BOXES[name])] for k in range(60)}), frames_processed=60)
            return sam_payload(sam_text({0: [(0, BOXES[name])]}))
        if any(p["type"] == "input_video" for p in parts):  # planner
            text = self.plan_replies.pop(0) if len(self.plan_replies) > 1 else self.plan_replies[0]
            return responses_payload(text, input_tokens=4900, output_tokens=2700)
        if "enum" in json.dumps(body["text"]["format"]["schema"]):  # object naming
            return responses_payload(json.dumps({"objects": OBJECTS}), input_tokens=1800, output_tokens=200)
        self.codegen.append(body)  # code generation
        return code_reply(RIGHT if "previous attempt failed" in user_text(body) else WRONG,
                          input_tokens=1800, output_tokens=2500)


def _clip(tmp_path):
    src = DEMOS_DIR / "t1_red_in_red_bin" / "seed_0" / "video.mp4"  # stands in for a real clip offline
    dst = tmp_path / "my clip.mp4"
    dst.write_bytes(src.read_bytes())
    return dst


def _client(tmp_path, world, guard=None):
    s = Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")
    return MetaClient(settings=s, transport=ScriptedTransport(world), guard=guard)


def test_pipeline_end_to_end_with_condition_c_repair(tmp_path):
    world = FakeRealWorld()
    stages = []
    s = run_pipeline(_clip(tmp_path), _client(tmp_path, world), out_root=tmp_path / "real",
                     on_stage=lambda n, p: stages.append((n, p)))
    assert [n for n, _ in stages] == ["normalized", "perception", "plan", "attempt", "attempt", "outcome"]
    assert s["name"] == "my_clip" and s["prompt_hash"] == frozen()["prompt_hash"]
    assert sorted(s["perception"]["named"]) == sorted(OBJECTS) and Path(s["perception"]["overlay"]).exists()
    assert s["plan"] == PLAN and (tmp_path / "real/my_clip/plan.json").exists()
    out = s["outcome"]
    assert out["success"] and out["attempts"] == 2 and out["judged_against"] == JUDGED_AGAINST
    assert out["goal"] == ["red_block in red_bin: ok"]
    for k in (1, 2):
        a = tmp_path / f"real/my_clip/repair/attempt_{k}"
        assert (a / "code.py").exists() and (a / "video.mp4").stat().st_size > 1000
    # condition C: the retry carried the failure report and the previous final frame
    retry = world.codegen[1]
    assert "previous attempt failed" in user_text(retry)
    assert sum(p["type"] == "input_image" for p in user_parts(retry)) == 1
    saved = json.loads((tmp_path / "real/my_clip/summary.json").read_text())
    assert saved["outcome"]["success"] and saved["seed"] == 300


def test_scene_has_exactly_the_plans_objects(tmp_path):
    from core_robot.codegen.scene import exec_initial_state

    task = PlanTask("x", json.dumps(PLAN))
    state = exec_initial_state(task, 300)
    assert sorted(state) == sorted(OBJECTS)
    assert not task.check(state, state)["success"]  # the goal does not already hold


def test_invalid_plan_stops_with_reason(tmp_path):
    world = FakeRealWorld(plan_replies=['{"objects": "nope"}'])
    s = run_pipeline(_clip(tmp_path), _client(tmp_path, world), out_root=tmp_path / "real")
    assert not s["outcome"]["success"] and s["outcome"]["reason"].startswith("no valid plan")
    assert world.codegen == []


def test_budget_stop_is_reported_not_crashed(tmp_path):
    guard = BudgetGuard(0.01)  # enough for naming, not for SAM on every object plus the planner
    s = run_pipeline(_clip(tmp_path), _client(tmp_path, FakeRealWorld(), guard), out_root=tmp_path / "real")
    assert not s["outcome"]["success"] and "BudgetExceeded" in s["outcome"]["reason"]
    assert guard.spent <= guard.budget


def test_script_without_clip_says_so(tmp_path):
    import sys

    r = subprocess.run([sys.executable, "scripts/run_real_demo.py", str(tmp_path / "missing.mp4"), "--dry-run"],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 2 and "no such clip" in r.stdout
