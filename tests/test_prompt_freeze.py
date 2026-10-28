"""Prompt freeze: the hash matches the frozen value, moves when a prompt changes, and run_eval
refuses changed prompts unless allowed (which is recorded)."""

import csv

import pytest

from core_robot import prompt_freeze
from core_robot.eval import run_eval
from core_robot.eval.fake_generator import FakeGenerator
from core_robot.eval.run_eval import RunSpec, run_matrix


def test_current_prompts_match_the_frozen_hash():
    f = prompt_freeze.frozen()
    assert f["tag"] == "prompts-frozen-v1"
    assert prompt_freeze.prompt_hash() == f["prompt_hash"], "a prompt changed since prompts-frozen-v1"


@pytest.mark.parametrize("module,name", [
    ("core_robot.planning.video_planner", "NOTE_RULE"),
    ("core_robot.planning.video_planner", "PLANNER_SYSTEM"),
    ("core_robot.codegen.prompt", "FREE_SPACE_HINT"),
    ("core_robot.codegen.api_doc", "CONTRACT"),
])
def test_hash_moves_when_a_prompt_text_changes(monkeypatch, module, name):
    import importlib

    mod = importlib.import_module(module)
    base = prompt_freeze.prompt_hash.__wrapped__()
    monkeypatch.setattr(mod, name, getattr(mod, name) + " (edited)")
    assert prompt_freeze.prompt_hash.__wrapped__() != base


def test_run_eval_refuses_changed_prompts(monkeypatch):
    monkeypatch.setattr(prompt_freeze, "prompt_hash", lambda: "000000000000")
    with pytest.raises(SystemExit, match="prompts changed since prompts-frozen-v1"):
        run_eval.main(["--generator", "fake", "--dry-run", "--run-name", "pytest_never_written"])
    run_eval.main(["--generator", "fake", "--dry-run", "--allow-prompt-change", "--run-name", "pytest_never_written"])


def test_runs_record_prompt_hash_commit_and_allowance(tmp_path):
    run_matrix([RunSpec("t1_red_in_red_bin", 100, "A")], FakeGenerator(), tmp_path, record=False)
    run_matrix([RunSpec("t1_red_in_red_bin", 101, "A")], FakeGenerator(), tmp_path, record=False,
               prompt_change_allowed=True)
    rows = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert {r["prompt_hash"] for r in rows} == {prompt_freeze.prompt_hash()}
    assert all(len(r["git_commit"].split("+")[0]) == 40 for r in rows)
    assert [r["prompt_change_allowed"] for r in sorted(rows, key=lambda r: r["seed"])] == ["0", "1"]
