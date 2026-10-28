"""Live code generator, offline: prompt content, code extraction, tokens, caching, repair wiring."""

from dataclasses import replace

import pytest
from _fakes import ScriptedTransport, code_reply, responses_payload, user_parts, user_text

from core_robot.codegen.muse_generator import MuseGenerator, extract_code
from core_robot.codegen.prompt import render_plan
from core_robot.codegen.scene import exec_initial_state, read_only_scene
from core_robot.config import Settings
from core_robot.demos.tasks import TASKS_BY_ID
from core_robot.eval.run_eval import RunSpec, run_matrix, token_cost
from core_robot.meta_client import MetaClient
from core_robot.repair.loop import run_with_repair

T1 = TASKS_BY_ID["t1_red_in_red_bin"]
SEED = 100
WRONG = 'def run(robot, scene):\n    robot.pick("red_block")\n    robot.place("red_block", "blue_bin")\n'
RIGHT = 'def run(robot, scene):\n    robot.pick("red_block")\n    robot.place("red_block", "red_bin")\n'


@pytest.fixture(scope="module")
def scene():
    return read_only_scene(exec_initial_state(T1, SEED))


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl")


def repairing_handler(method, path, body):
    """Wrong bin until the prompt carries a failure report, then the right bin."""
    return code_reply(RIGHT if "previous attempt failed" in user_text(body) else WRONG,
                      input_tokens=5100, output_tokens=420)


# ---- extraction ----

@pytest.mark.parametrize("result,expected_start", [
    ({"json": {"code": RIGHT}, "text": "{...}"}, "def run"),
    ({"json": {"code": "```python\n" + RIGHT + "```"}, "text": ""}, "def run"),
    ({"json": None, "text": "Here you go:\n```python\n" + RIGHT + "```\nDone."}, "def run"),
    ({"json": None, "text": "```py\nx = 1\n```\n```python\n" + RIGHT + "```"}, "def run"),
    ({"json": None, "text": RIGHT}, "def run"),
])
def test_extract_code(result, expected_start):
    code = extract_code(result)
    assert code.startswith(expected_start) and code.endswith("\n") and "```" not in code


# ---- prompt ----

def test_prompt_contents_and_request_shape(settings, scene):
    t = ScriptedTransport(lambda m, p, b: code_reply(RIGHT, input_tokens=4321, output_tokens=210))
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    plan = T1.ground_truth_plan(exec_initial_state(T1, SEED))
    out = gen(task=T1, plan=plan, attempt=1, scene=scene)
    assert {k: out[k] for k in ("code", "tokens_in", "tokens_out")} == {"code": RIGHT, "tokens_in": 4321, "tokens_out": 210}
    assert len(out["request_key"]) == 64 and out["status"] == "completed"
    (body,) = t.bodies()
    text = user_text(body)
    assert body["model"] == "muse-spark-1.3"
    assert body["text"]["format"]["type"] == "json_schema"
    assert render_plan(plan) in text
    assert "robot.place(obj, target)" in text and "GLOBAL SUCCESS RULE" in text
    assert '"red_block": {' in text and "free_table_space" in text
    assert T1.instruction not in text  # the generator works from the plan, not the task text
    assert "previous attempt failed" not in text
    assert [p["type"] for p in user_parts(body)] == ["input_text"]


def test_feedback_and_image_only_on_retry(settings, scene, tmp_path):
    img = tmp_path / "final_angled.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    t = ScriptedTransport(lambda m, p, b: code_reply(RIGHT))
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    plan = T1.ground_truth_plan(exec_initial_state(T1, SEED))
    gen(task=T1, plan=plan, attempt=2, scene=scene, feedback="Summary: FAILED (x)", image_path=str(img))
    (body,) = t.bodies()
    assert "Summary: FAILED (x)" in user_text(body) and "angled camera" in user_text(body)
    image_parts = [p for p in user_parts(body) if p["type"] == "input_image"]
    assert len(image_parts) == 1 and image_parts[0]["image_url"].startswith("data:image/png;base64,")


def test_identical_request_is_cached(settings, scene):
    t = ScriptedTransport(lambda m, p, b: code_reply(RIGHT))
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    plan = T1.ground_truth_plan(exec_initial_state(T1, SEED))
    a = gen(task=T1, plan=plan, attempt=1, scene=scene)
    b = MuseGenerator(MetaClient(settings=settings, transport=t))(task=T1, plan=plan, attempt=1, scene=scene)
    assert a == b and len(t.calls) == 1


def test_unparseable_reply_still_yields_code(settings, scene):
    t = ScriptedTransport(lambda m, p, b: responses_payload("```python\n" + RIGHT + "```"))
    out = MuseGenerator(MetaClient(settings=settings, transport=t))(
        task=T1, plan=T1.ground_truth_plan(exec_initial_state(T1, SEED)), attempt=1, scene=scene)
    assert out["code"] == RIGHT


# ---- wired into the repair loop and the evaluation harness ----

@pytest.mark.parametrize("cond,attempts,success,images", [("A", 1, False, 0), ("B", 2, True, 0), ("C", 2, True, 1)])
def test_repair_loop_with_muse_generator(settings, tmp_path, cond, attempts, success, images):
    t = ScriptedTransport(repairing_handler)
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    plan = T1.ground_truth_plan(exec_initial_state(T1, SEED))
    log = run_with_repair(T1, plan, SEED, cond, gen, out_dir=tmp_path / cond, record=False)
    assert log["attempts_used"] == attempts and log["success"] is success
    assert log["tokens_in"] == 5100 * attempts and log["tokens_out"] == 420 * attempts
    sent_images = sum(1 for b in t.bodies() for p in user_parts(b) if p["type"] == "input_image")
    assert sent_images == images


def test_run_matrix_records_live_tokens_and_cost(settings, tmp_path):
    import csv

    t = ScriptedTransport(repairing_handler)
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    run_matrix([RunSpec("t1_red_in_red_bin", SEED, "B")], gen, tmp_path, record=False)
    (row,) = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert row["run_source"] == "live" and row["success"] == "1" and row["attempts"] == "2"
    assert int(row["tokens_in"]) == 10200 and int(row["tokens_out"]) == 840
    assert float(row["est_cost_usd"]) == pytest.approx(token_cost(10200, 840), abs=1e-6)


def test_replay_mode_with_empty_cache_is_harness_error(settings, tmp_path):
    import csv

    t = ScriptedTransport(repairing_handler)
    gen = MuseGenerator(MetaClient(settings=replace(settings, replay=True), transport=t))
    run_matrix([RunSpec("t1_red_in_red_bin", SEED, "A")], gen, tmp_path, record=False)
    (row,) = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert row["failure_category"] == "harness error" and t.calls == []


def test_muse_without_key_exits_before_running(monkeypatch):
    from core_robot.eval import run_eval

    monkeypatch.setattr(run_eval, "load_settings", lambda: Settings(api_key=None, replay=False))
    with pytest.raises(SystemExit, match="MODEL_API_KEY"):
        run_eval.main(["--generator", "muse", "--tasks", "t1", "--seeds", "100", "--conditions", "A",
                       "--run-name", "pytest_never_written"])


def test_tasks_filter():
    from core_robot.eval.run_eval import planned_runs, resolve_tasks

    ids = resolve_tasks("t4,t7")
    assert ids == ["t4_stack_three", "t7_swap_red_blue"]
    runs = planned_runs(ids, [100], ("A", "B", "C"))
    assert [(r.task_id, r.condition) for r in runs] == [
        ("t4_stack_three", "A"), ("t4_stack_three", "B"), ("t4_stack_three", "C"),
        ("t7_swap_red_blue", "A"), ("t7_swap_red_blue", "B"), ("t7_swap_red_blue", "C")]
    with pytest.raises(SystemExit):
        resolve_tasks("t9")



# ---- paired attempt 1 across conditions ----

class CountingHandler:
    """Every network call returns a different code variant (like a sampling model), slowly,
    so parallel identical requests would get different answers if they were not coalesced."""

    def __init__(self):
        import threading

        self.n = 0
        self.lock = threading.Lock()

    def __call__(self, method, path, body):
        import time

        with self.lock:
            self.n += 1
            n = self.n
        time.sleep(0.3)
        code = (RIGHT if "previous attempt failed" in user_text(body) else WRONG) + f"# variant {n}\n"
        return code_reply(code, input_tokens=1700, output_tokens=700 + n)


def test_concurrent_identical_requests_are_sent_once(settings):
    from concurrent.futures import ThreadPoolExecutor

    h = CountingHandler()
    t = ScriptedTransport(h)
    client = MetaClient(settings=settings, transport=t)
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(lambda _: client.muse("same prompt", max_output_tokens=100), range(4)))
    assert len(t.calls) == 1 and len({o["text"] for o in outs}) == 1 and len({o["request_key"] for o in outs}) == 1


def test_attempt_one_is_shared_across_conditions_in_parallel(settings, tmp_path):
    import json as _json

    h = CountingHandler()
    t = ScriptedTransport(h)
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    specs = [RunSpec("t1_red_in_red_bin", SEED, c) for c in "ABC"]
    run_matrix(specs, gen, tmp_path, workers=3, record=False)
    logs = {c: _json.loads((tmp_path / "t1_red_in_red_bin" / c / f"seed_{SEED}" / "run.json").read_text()) for c in "ABC"}
    first = [logs[c]["attempts"][0] for c in "ABC"]
    assert len({a["code"] for a in first}) == 1
    assert len({a["generation"]["request_key"] for a in first}) == 1
    assert len({(a["tokens_in"], a["tokens_out"]) for a in first}) == 1
    attempt1_calls = [b for b in t.bodies() if "previous attempt failed" not in user_text(b)]
    assert len(attempt1_calls) == 1  # one billed generation for attempt 1, shared by A, B and C
    # B and C differ from A only after the first failure; each attempt starts from a fresh reset
    for c in "BC":
        assert logs[c]["attempts"][1]["feedback_given"] and logs[c]["attempts"][1]["code"] != first[0]["code"]
        assert logs[c]["attempts"][1]["initial_state"] == logs[c]["attempts"][0]["initial_state"]


def test_b_and_c_reuse_attempt_one_after_a_has_run(settings, tmp_path):
    h = CountingHandler()
    t = ScriptedTransport(h)
    gen = MuseGenerator(MetaClient(settings=settings, transport=t))
    run_matrix([RunSpec("t1_red_in_red_bin", SEED, "A")], gen, tmp_path, record=False)
    assert len(t.calls) == 1
    run_matrix([RunSpec("t1_red_in_red_bin", SEED, "B"), RunSpec("t1_red_in_red_bin", SEED, "C")], gen, tmp_path,
               workers=2, record=False)
    fresh = [b for b in t.bodies()[1:] if "previous attempt failed" not in user_text(b)]
    assert fresh == []  # attempt 1 of B and C came from the cache written by A
