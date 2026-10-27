"""Sandbox: static rejection, timeout, restricted robot, read-only scene, valid runs."""

import pytest

from core_robot.codegen.api_doc import primitives_reference
from core_robot.codegen.sandbox import run_code, validate_code
from core_robot.demos.tasks import TASKS_BY_ID

T1 = "t1_red_in_red_bin"
GOOD_T1 = """
def run(robot, scene):
    r = robot.pick("red_block")
    if not r["ok"]:
        raise RuntimeError(r["reason"])
    robot.place("red_block", "red_bin")
"""


def body(src):
    return "def run(robot, scene):\n" + "\n".join("    " + line for line in src.splitlines())


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + body("pass"),
        "from subprocess import run as r\n" + body("pass"),
        body("import socket"),
        body("open('/etc/passwd').read()"),
        body("exec('1')"),
        body("eval('1')"),
        body("__import__('os')"),
        body("robot.__class__"),
        body("robot.pick.__closure__"),
        body("scene.__class__.__bases__"),
        body("x = ().__class__.__base__.__subclasses__()"),
        body("robot._robot"),
        body("getattr(robot, 'env')"),
        body("'{0.__class__}'.format(robot)"),
        body("str.format('{0}', robot)"),
        body("type(robot)"),
        body("globals()"),
        body("compile('1', 'x', 'exec')"),
        body("g = (i for i in []); g.gi_frame.f_globals"),
        "def helper():\n    pass\n",  # no run()
        "def run(robot):\n    pass\n",  # wrong signature
        "def run(robot, scene)\n    pass\n",  # syntax error
    ],
)
def test_forbidden_code_rejected(code, tmp_path):
    assert validate_code(code)
    r = run_code(code, T1, 100, tmp_path)
    assert r["rejected"] and not r["ok"]
    assert r["exception"].startswith("rejected before running")
    assert not (tmp_path / "job.json").exists()  # never reached the child process


def test_valid_run_succeeds(tmp_path):
    r = run_code(GOOD_T1, T1, 100, tmp_path)
    assert r["ok"], r["exception"]
    assert [c["primitive"] for c in r["call_log"]] == ["pick", "place"]
    assert all(c["ok"] for c in r["call_log"])
    assert TASKS_BY_ID[T1].check(r["initial_state"], r["final_state"])["success"]
    assert (tmp_path / "video.mp4").stat().st_size > 1000
    for cam in ("angled", "side"):
        assert (tmp_path / f"final_{cam}.png").exists()


def test_infinite_loop_times_out(tmp_path):
    r = run_code(body("while True:\n    pass"), T1, 100, tmp_path, timeout_s=8, record=False)
    assert r["timed_out"] and not r["ok"]
    assert "timeout" in r["exception"]


def test_partial_call_log_survives_timeout(tmp_path):
    code = body('robot.pick("red_block")\nwhile True:\n    pass')
    r = run_code(code, T1, 100, tmp_path, timeout_s=10, record=False)
    assert r["timed_out"]
    assert [c["primitive"] for c in r["call_log"]] == ["pick"]


def test_runtime_exception_reported(tmp_path):
    r = run_code(body('robot.pick("red_block")\nraise ValueError("boom")'), T1, 100, tmp_path, record=False)
    assert not r["ok"] and not r["rejected"]
    assert r["exception"].startswith("ValueError: boom") and "line 3" in r["exception"]
    assert r["final_state"] is not None and len(r["call_log"]) == 1


@pytest.mark.parametrize(
    "escape",
    ["robot.env", "robot.call_log", "robot.held", "robot.home()", "robot.env.reset(0)", "robot.pick.closure"],
)
def test_restricted_robot_escapes_fail(escape, tmp_path):
    r = run_code(body(escape), T1, 100, tmp_path, record=False)
    assert not r["ok"]
    assert "AttributeError" in r["exception"]
    assert r["initial_state"] == r["final_state"] or _same_positions(r)


def _same_positions(r):
    return all(
        max(abs(a - b) for a, b in zip(r["initial_state"][n]["position"], r["final_state"][n]["position"])) < 1e-3
        for n in r["initial_state"]
    )


def test_robot_only_exposes_four_primitives(tmp_path):
    code = body(
        'r = robot.open_gripper()\n'
        'print(sorted(r))\n'
        'print(robot)'
    )
    r = run_code(code, T1, 100, tmp_path, record=False)
    assert r["ok"], r["exception"]
    assert "['details', 'ok', 'reason']" in r["stdout"]
    assert "pick, place, move_above, open_gripper" in r["stdout"]


def test_scene_mutation_does_not_affect_simulator(tmp_path):
    code = body(
        'scene["red_block"]["position"][0] = 9.0\n'
        'scene["red_block"]["position"] = [0.0, 0.0, 5.0]\n'
        'del scene["red_bin"]\n'
        'scene["purple_block"] = {"position": [0.5, 0.0, 0.02]}\n'
        'r = robot.pick("purple_block")\n'
        'print(r["ok"], r["reason"])\n'
        'robot.pick("red_block")\n'
        'robot.place("red_block", "red_bin")'
    )
    r = run_code(code, T1, 100, tmp_path, record=False)
    assert r["ok"], r["exception"]
    assert "False unknown object 'purple_block'" in r["stdout"]  # fake object does not exist
    assert set(r["final_state"]) == set(r["initial_state"])  # red_bin still exists, nothing added
    assert TASKS_BY_ID[T1].check(r["initial_state"], r["final_state"])["success"]


def test_scene_has_only_public_fields(tmp_path):
    code = body("for name, v in scene.items():\n    print(name, sorted(v))")
    r = run_code(code, T1, 100, tmp_path, record=False)
    assert r["ok"], r["exception"]
    lines = r["stdout"].strip().splitlines()
    assert len(lines) == 3
    assert all(line.endswith("['color', 'kind', 'orientation', 'position']") for line in lines)


def test_api_key_not_in_child_env(monkeypatch):
    from core_robot.codegen.sandbox import child_env

    monkeypatch.setenv("MODEL_API_KEY", "sk-test-should-not-leak")
    env = child_env()
    assert "MODEL_API_KEY" not in env and "PATH" in env


def test_api_doc_lists_primitives_and_rule():
    doc = primitives_reference()
    for sig in ("robot.pick(obj)", "robot.place(obj, target)", "robot.move_above(target)", "robot.open_gripper()"):
        assert sig in doc
    assert "def run(robot, scene):" in doc
    assert "read-only data, not a primitive" in doc
    assert "GLOBAL SUCCESS RULE" in doc and "destination" in doc
    assert "swapped_with" in doc
    assert "_move_to" not in doc and "home(" not in doc
