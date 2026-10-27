"""Perception (naming, SAM tracking, keyframe fallback, overlay), video->plan, scoring and the
SAM gate, all offline against a fake Meta API that speaks the real wire formats."""

import json
from pathlib import Path

import numpy as np
import pytest
from _fakes import (
    ScriptedTransport, all_text, media_sha, responses_payload, sam_payload, sam_text, user_parts, user_text,
)

from core_robot.codegen.prompt import render_plan
from core_robot.config import DEMOS_DIR, Settings
from core_robot.eval import demo_eval
from core_robot.meta_client import MetaClient
from core_robot.perception.costs import actual_spend, estimate_perception_planning, resampled_frames
from core_robot.perception.gate import check_object
from core_robot.perception.overlay import decode_mask
from core_robot.perception.sam_tracking import CONCEPTS, select_object, track_object
from core_robot.planning.schema import Plan
from core_robot.planning.scoring import score
from core_robot.planning.video_planner import plan_from_video
from core_robot.sim.env import TabletopEnv

NAME_FOR = {c: n for n, c in CONCEPTS.items()}
HALF = {"block": (11, 11), "bin": (46, 30), "mug": (18, 26)}
ALL_DEMOS = demo_eval.demo_ids()
GROUND_TRUTH_FILES = {"ground_truth_plan.json", "initial_state.json", "final_state.json", "result.json"}
WRONG_PLAN_DEMOS = {"t2_sort_by_color/seed_1", "t4_stack_three/seed_0", "t5_line_up/seed_0"}


@pytest.fixture(scope="module")
def project():
    env = TabletopEnv()
    yield lambda p: env.project(p, "angled")
    env.close()


def gt_plan(demo_id) -> Plan:
    return Plan.model_validate_json((DEMOS_DIR / demo_id / "ground_truth_plan.json").read_text())


def wrong_plan(demo_id, gt: Plan | None = None) -> dict:
    d = (gt or gt_plan(demo_id)).model_dump(exclude_none=True)
    d["steps"] = [{**s, "index": i} for i, s in enumerate(reversed(d["steps"]))]
    if len(d["steps"]) == 1:
        d["goal"][0]["relation"] = "next_to"
    return d


class FakeMetaWorld:
    """Answers SAM and Muse requests for the demo set, telling demos apart by the media sent.

    SAM boxes follow each object's true start->end path (projected into the angled camera),
    with real one_bit masks. Naming returns the demo's objects. The planner returns the
    ground-truth plan except for WRONG_PLAN_DEMOS (the fake may read ground truth; the
    code under test must not, and must never put it in a prompt).
    """

    def __init__(self, root: Path, project, planner_invalid_first: bool = False):
        self.root, self.project = root, project
        # Ground truth is loaded up front so the fake never touches those files while the
        # code under test runs (the read guard below would flag it).
        self.gt = {d: gt_plan(d) for d in ALL_DEMOS}
        self.states = {d: tuple(json.loads((DEMOS_DIR / d / f).read_text())
                                for f in ("initial_state.json", "final_state.json")) for d in ALL_DEMOS}
        self.planner_invalid_first = planner_invalid_first
        self.media: dict[str, tuple] = {}
        self.planner_calls: dict[str, int] = {}

    def _identify(self, sha):
        if sha not in self.media:
            for p in self.root.rglob("*"):
                if p.suffix in (".mp4", ".png") and p.is_file():
                    import hashlib

                    demo = "/".join(p.relative_to(self.root).parts[:2])
                    idx = int(p.stem.split("_")[1]) if p.stem.startswith("keyframe_") else None
                    self.media.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), (demo, p.name, idx))
        return self.media[sha]

    def boxes(self, demo, name, n):
        s0, s1 = self.states[demo][0][name], self.states[demo][1][name]
        a, b = np.array(self.project(s0["position"])), np.array(self.project(s1["position"]))
        hx, hy = HALF[s0["kind"]]
        out = {}
        for k in range(n):
            c = a + (b - a) * (k / max(1, n - 1))
            out[k] = (c[0] - hx, c[1] - hy, c[0] + hx, c[1] + hy)
        return out

    def __call__(self, method, path, body):
        demo, fname, idx = self._identify(media_sha(body))
        initial = self.states[demo][0]
        if body["model"] == "sam-3.1":
            name = NAME_FOR[user_text(body).strip()]
            if name not in initial:
                return sam_payload("")
            if fname.endswith(".mp4"):
                n = _clip_frames(self.root / demo / "video_5fps.mp4")
                bx = self.boxes(demo, name, n)
                return sam_payload(sam_text({k: [(3, b)] for k, b in bx.items()}), frames_processed=max(50, n))
            n = _clip_frames(self.root / demo / "video_5fps.mp4")
            return sam_payload(sam_text({0: [(0, self.boxes(demo, name, n)[idx])]}))
        if any(p["type"] == "input_image" for p in user_parts(body)):  # object naming
            return responses_payload(json.dumps({"objects": sorted(initial)}), input_tokens=2800, output_tokens=60)
        self.planner_calls[demo] = self.planner_calls.get(demo, 0) + 1
        if self.planner_invalid_first and self.planner_calls[demo] == 1:
            bad = self.gt[demo].model_dump(exclude_none=True)
            bad["goal"][0]["relation"] = "inside"
            return responses_payload(json.dumps(bad), input_tokens=18000, output_tokens=900)
        plan = wrong_plan(demo, self.gt[demo]) if demo in WRONG_PLAN_DEMOS else self.gt[demo].model_dump(exclude_none=True)
        return responses_payload(json.dumps(plan), input_tokens=18000, output_tokens=900)


def _clip_frames(path) -> int:
    from core_robot.perception.frames import read_frames

    return len(read_frames(path)[0])


@pytest.fixture(scope="module")
def world_run(tmp_path_factory, project):
    root = tmp_path_factory.mktemp("perception")
    settings = Settings(api_key=None, cache_dir=root / "cache", log_path=root / "calls.jsonl")
    world = FakeMetaWorld(root, project)
    t = ScriptedTransport(world)
    client = MetaClient(settings=settings, transport=t, sleep=lambda s: None)
    gt_reads = []
    real_read_text = Path.read_text

    def guarded_read_text(self, *a, **k):
        if self.parent.parent.parent == DEMOS_DIR and self.name in GROUND_TRUTH_FILES:
            gt_reads.append(str(self))
        return real_read_text(self, *a, **k)

    Path.read_text = guarded_read_text
    try:
        perceptions = {d: demo_eval.perception_for(client, d, root, reuse=False) for d in ALL_DEMOS}
        plans = {d: plan_from_video(client, perceptions[d]["clip"]["path"], perceptions[d]) for d in ALL_DEMOS}
        reads_during_planning = list(gt_reads)
    finally:
        Path.read_text = real_read_text
    planner = demo_eval.evaluate_planner(client, ALL_DEMOS, root, root / "planner")
    sam = demo_eval.evaluate_sam(client, ALL_DEMOS, root)
    return {"root": root, "transport": t, "settings": settings, "perceptions": perceptions, "plans": plans,
            "planner": planner, "sam": sam, "gt_reads_during_planning": reads_during_planning, "client": client}


# ---- no ground truth in any prompt ----

def test_no_ground_truth_plan_in_any_prompt(world_run):
    bodies = world_run["transport"].bodies("muse-spark-1.3")
    assert len(bodies) >= 2 * len(ALL_DEMOS)
    for demo in ALL_DEMOS:
        gt = gt_plan(demo)
        full = gt.model_dump(exclude_none=True)
        forbidden = [json.dumps(full), json.dumps(full, indent=2), json.dumps(full["steps"]),
                     json.dumps(full["goal"]), render_plan(gt)]
        forbidden += [f"{g.subject} {g.relation} {g.target}" for g in gt.goal]
        forbidden += [f"{s.object} -> {s.destination}" for s in gt.steps]
        forbidden += [s.note for s in gt.steps if s.note]
        for body in bodies:
            text = all_text(body)
            for f in forbidden:
                assert f not in text, f"ground truth leaked into a prompt: {f!r}"


def test_planning_never_reads_ground_truth_files(world_run):
    assert world_run["gt_reads_during_planning"] == []


# ---- perception ----

def test_naming_and_tracks(world_run):
    for demo, p in world_run["perceptions"].items():
        initial = json.loads((DEMOS_DIR / demo / "initial_state.json").read_text())
        assert p["naming"]["objects"] and set(p["naming"]["objects"]) == set(initial)
        for name, t in p["tracks"].items():
            assert t["source"] == "video" and t["coverage"] == 1.0 and t["object_id"] == 3
        assert Path(p["overlay"]).stat().st_size > 1000


def test_naming_constrained_to_simulator_names(world_run):
    body = next(b for b in world_run["transport"].bodies("muse-spark-1.3")
                if any(p["type"] == "input_image" for p in user_parts(b)))
    enum = body["text"]["format"]["schema"]["properties"]["objects"]["items"]["enum"]
    assert set(enum) == set(CONCEPTS) and "red_block" in enum


def test_sam_requests_use_concepts_and_one_bit(world_run):
    sam = world_run["transport"].bodies("sam-3.1")
    assert sam and all(b["metadata"] == {"mask_encoding": "one_bit"} for b in sam)
    assert {user_text(b).strip() for b in sam} <= set(CONCEPTS.values())


def test_overlay_masks_decode(world_run):
    p = world_run["perceptions"]["t4_stack_three/seed_0"]
    rec = p["tracks"]["red_block"]["frames"][0]
    m = decode_mask(rec["mask"])
    x1, y1, x2, y2 = rec["box"]
    assert m is not None and m.shape == (y2 - y1 + 1, x2 - x1 + 1) and m.all()


def test_select_object_prefers_block_sized_match():
    # "red cube" also matched the red bin (id 1, large) in a few frames.
    small, big = (100, 100, 120, 120), (200, 200, 300, 280)
    frames = {f: {0: {"box": small}, 1: {"box": big}} for f in range(10)}
    assert select_object(frames, "block") == 0
    assert select_object(frames, "bin") == 1
    frames_rare = {f: {0: {"box": small}, **({1: {"box": big}} if f < 2 else {})} for f in range(10)}
    assert select_object(frames_rare, "bin") == 0  # id 1 seen on too few frames to count


def _client_with(handler, tmp_path):
    s = Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl", max_attempts=1)
    return MetaClient(settings=s, transport=ScriptedTransport(handler), sleep=lambda s: None)


@pytest.fixture
def clip(tmp_path):
    from core_robot.perception.frames import resample_video

    return resample_video(DEMOS_DIR / "t1_red_in_red_bin/seed_0/video.mp4", tmp_path / "clip.mp4")


def test_keyframe_fallback_on_video_error(tmp_path, clip):
    from core_robot.meta_client import ApiError

    def handler(method, path, body):
        if any(p["type"] == "input_video" for p in user_parts(body)):
            raise ApiError(400, "video rejected: too many frames")
        return sam_payload(sam_text({0: [(0, (300, 200, 320, 220))]}))

    t = track_object(_client_with(handler, tmp_path), clip["path"], "red_block", clip["n_frames"], tmp_path / "kf")
    assert t["source"] == "keyframes" and t["coverage"] == 1.0 and len(t["attempted"]) == 8
    assert t["fallback"]["reason"].startswith("HTTP 400") and t["fallback"]["used"] == "keyframes"
    assert sorted(t["frames"]) == t["attempted"] and t["attempted"][0] == 0 and t["attempted"][-1] == clip["n_frames"] - 1


def test_keyframe_fallback_on_low_coverage(tmp_path, clip):
    n = clip["n_frames"]

    def handler(method, path, body):
        if any(p["type"] == "input_video" for p in user_parts(body)):
            return sam_payload(sam_text({k: [(0, (300, 200, 320, 220))] for k in range(n // 2)}))
        return sam_payload(sam_text({0: [(0, (300, 200, 320, 220))]}))

    t = track_object(_client_with(handler, tmp_path), clip["path"], "red_block", n, tmp_path / "kf")
    assert t["fallback"]["video_coverage"] < 0.8 and t["source"] == "keyframes"


def test_no_fallback_when_video_is_good(tmp_path, clip):
    n = clip["n_frames"]
    seen = []

    def handler(method, path, body):
        seen.append([p["type"] for p in user_parts(body)])
        return sam_payload(sam_text({k: [(0, (300, 200, 320, 220))] for k in range(n)}))

    t = track_object(_client_with(handler, tmp_path), clip["path"], "red_block", n, tmp_path / "kf")
    assert t["source"] == "video" and "fallback" not in t and len(seen) == 1


# ---- planner ----

def test_planner_results_and_scores(world_run):
    s = world_run["planner"]
    assert s["demos"] == 10 and s["strict"] == 7 and s["lenient"] >= 7
    wrong = {r["demo"] for r in s["rows"] if not r["strict"]}
    assert wrong == WRONG_PLAN_DEMOS
    row = next(r for r in s["rows"] if r["demo"] == "t2_sort_by_color/seed_1")
    assert row["errors"] == ["wrong order"] and row["lenient"]  # both bins correct, order differs
    assert dict(s["top_errors"])  # most common error types are reported


def test_planner_uses_structured_output_and_video(world_run):
    body = next(b for b in world_run["transport"].bodies("muse-spark-1.3")
                if any(p["type"] == "input_video" for p in user_parts(b)))
    assert body["text"]["format"]["type"] == "json_schema"
    assert "steps" in json.dumps(body["text"]["format"]["schema"])
    text = user_text(body)
    assert "Segmentation tracks from SAM 3.1" in text and "free_table_space" in text


def test_planner_retries_once_on_validation_error(tmp_path, project):
    demo = "t3_stack_green_on_red/seed_0"
    world = FakeMetaWorld(tmp_path, project, planner_invalid_first=True)
    t = ScriptedTransport(world)
    client = MetaClient(settings=Settings(api_key=None, cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl"),
                        transport=t)
    p = demo_eval.perception_for(client, demo, tmp_path, reuse=False)
    r = plan_from_video(client, p["clip"]["path"], p)
    assert len(r["attempts"]) == 2 and r["plan"] is not None and r["error"] is None
    assert "relation" in r["attempts"][0]["error"]
    retry = [b for b in t.bodies("muse-spark-1.3") if any(x["type"] == "input_video" for x in user_parts(b))][-1]
    assert "previous reply was rejected" in user_text(retry) and "inside" in user_text(retry)
    assert r["tokens"] == (36000, 1800)


def test_planner_gives_up_after_one_retry(tmp_path):
    def handler(method, path, body):
        return responses_payload('{"objects": [], "steps": "nope"}')

    client = _client_with(handler, tmp_path)
    perception = {"naming": {"objects": ["red_block"]}, "tracks": {}, "clip": {"n_frames": 3, "size": (640, 480), "fps": 5}}
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    r = plan_from_video(client, video, perception)
    assert r["plan"] is None and len(r["attempts"]) == 2 and r["error"]
    assert score(None, gt_plan("t1_red_in_red_bin/seed_0"))["errors"] == ["invalid plan"]


# ---- scoring definitions ----

def test_scoring_strict_and_lenient():
    gt = gt_plan("t4_stack_three/seed_0")
    same = Plan.model_validate(gt.model_dump())
    noted = Plan.model_validate({**gt.model_dump(), "steps": [{**s, "note": "x"} for s in gt.model_dump()["steps"]]})
    reordered_goal = Plan.model_validate({**gt.model_dump(), "goal": list(reversed(gt.model_dump()["goal"]))})
    assert score(same, gt) == {"strict": True, "lenient": True, "errors": []}
    assert score(noted, gt)["strict"] and score(reordered_goal, gt)["strict"]  # goal is a set; notes ignored
    swapped = Plan.model_validate(wrong_plan("t4_stack_three/seed_0"))
    s = score(swapped, gt)
    assert not s["strict"] and s["lenient"] and s["errors"] == ["wrong order"]


def test_scoring_error_types():
    gt = gt_plan("t1_red_in_red_bin/seed_0")
    d = gt.model_dump(exclude_none=True)
    wrong_dest = Plan.model_validate({**d, "steps": [{**d["steps"][0], "destination": "blue_bin"}],
                                      "goal": [{**d["goal"][0], "target": "blue_bin"}]})
    s = score(wrong_dest, gt)
    assert "wrong destination" in s["errors"] and "missing goal atom" in s["errors"] and "extra goal atom" in s["errors"]
    rel = Plan.model_validate({**d, "goal": [{**d["goal"][0], "relation": "next_to"}]})
    assert score(rel, gt)["errors"] == ["wrong goal relation"] and not score(rel, gt)["lenient"]


# ---- SAM gate ----

def test_sam_gate_all_pass_with_true_tracks(world_run):
    s = world_run["sam"]
    assert s["passed"] == 10, [(r["demo"], {k: v["reason"] for k, v in r["objects"].items() if not v["pass"]})
                               for r in s["rows"] if not r["pass"]]
    t8 = next(r for r in s["rows"] if r["demo"] == "t8_nearest_to_mug/seed_0")
    assert t8["objects"]["mug"]["required"] is False


def _track(frames, coverage=1.0):
    return {"frames": frames, "coverage": coverage, "source": "video"}


def test_gate_object_rules():
    box = [90, 90, 110, 110]
    n = 20
    full = {k: {"box": box} for k in range(n)}
    assert check_object(_track(full), (100, 100), (100, 100), n)["pass"]
    assert "coverage" in check_object(_track(full, 0.79), (100, 100), (100, 100), n)["reason"]
    assert "start box misses" in check_object(_track(full), (150, 100), (100, 100), n)["reason"]
    assert "end box misses" in check_object(_track(full), (100, 100), (100, 150), n)["reason"]
    near = {k: {"box": box} for k in range(3, n - 3)}  # first and last 3 frames missing
    r = check_object(_track(near), (100, 100), (100, 100), n)
    assert r["pass"] and r["start_frame"] == 3 and r["end_frame"] == n - 4
    far = {k: {"box": box} for k in range(4, n)}  # nearest start mask is 4 frames in
    assert "no mask within 3 frames of the start" in check_object(_track(far), (100, 100), (100, 100), n)["reason"]
    assert check_object(None, (0, 0), (0, 0), n)["reason"] == "not named or no track"


def test_gate_fails_when_a_bin_is_not_named(world_run, project):
    from core_robot.perception.gate import sam_gate

    p = json.loads(json.dumps(world_run["perceptions"]["t1_red_in_red_bin/seed_0"]))
    for t in p["tracks"].values():
        t["frames"] = {int(k): v for k, v in t["frames"].items()}
    del p["tracks"]["blue_bin"]
    d = DEMOS_DIR / "t1_red_in_red_bin/seed_0"
    g = sam_gate(p, json.loads((d / "initial_state.json").read_text()),
                 json.loads((d / "final_state.json").read_text()), project)
    assert not g["pass"] and g["objects"]["blue_bin"]["reason"] == "not named or no track"


# ---- costs ----

def test_resampled_frames_matches_resampler(clip):
    assert resampled_frames(82) == clip["n_frames"]


def test_perception_estimate_arithmetic():
    e = estimate_perception_planning([{"name": "a", "n_frames": 82, "n_objects": 3},
                                      {"name": "b", "n_frames": 220, "n_objects": 3}])
    assert e["sam_video_frames_billed"] == 3 * 50 + 3 * 111
    assert e["sam_video_usd"] == pytest.approx((150 + 333) * 0.0002)
    assert e["sam_fallback_usd_worst"] == pytest.approx(6 * 8 * 0.0025)
    assert e["worst_usd"] > e["expected_usd"] > 0


def test_actual_spend_from_log(world_run):
    s = actual_spend(world_run["settings"].log_path)
    assert s["sam_frames"] > 0 and s["muse_calls"] >= 20 and s["tokens_in"] > 0
    assert s["usd"] > 0 and s.get("sam_frames_estimated", 0) == 0
    again = actual_spend(world_run["settings"].log_path, since=9e12)
    assert again["usd"] == 0
