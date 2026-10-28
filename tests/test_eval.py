"""Evaluation harness: cost estimate, resumability, determinism, labelling, fake isolation."""

import csv
import json
from pathlib import Path

import pytest

from core_robot.config import FAKE_RUNS_DIR, LIVE_RUNS_DIR, RUNS_DIR
from core_robot.eval.fake_generator import FakeGenerator
from core_robot.eval.report import FAKE_BANNER, build_markdown, build_readme_results, write_report
from core_robot.eval.run_eval import (
    CSV_FIELDS, RunSpec, estimate_codegen_cost, failure_category, main, output_root, planned_runs,
    run_matrix, run_path, token_cost,
)
from core_robot.meta_client import ReplayMiss

DEMOS = Path(__file__).resolve().parents[1] / "data" / "demos"


# ---- cost ----

def test_token_cost_per_generation():
    # 8000 * 1.25/1e6 + 1500 * 4.25/1e6 = 0.0100 + 0.006375
    assert token_cost(8000, 1500) == pytest.approx(0.016375)


def test_codegen_upper_bound_full_matrix():
    est = estimate_codegen_cost(planned_runs())
    assert est["runs"] == 120
    assert est["generations"] == 40 * 1 + 40 * 3 + 40 * 3 == 280
    assert est["usd_total"] == pytest.approx(280 * 0.016375) == pytest.approx(4.585)


def test_dry_run_prints_generations_tokens_total(capsys):
    assert main(["--dry-run", "--run-name", "pytest_dry_run_never_written"]) == 0
    out = capsys.readouterr().out
    assert "280 generations x (8000 in + 1500 out tokens)" in out and "$4.58" in out
    assert "120 runs to do" in out
    assert not (FAKE_RUNS_DIR / "pytest_dry_run_never_written").exists()


def test_limit():
    assert len(planned_runs(limit=7)) == 7
    assert planned_runs(limit=7) == planned_runs()[:7]


# ---- isolation of fake output ----

def test_fake_output_only_under_runs_fake():
    root, src = output_root("fake", "ground_truth")
    assert src == "fake" and FAKE_RUNS_DIR in root.parents and root.is_relative_to(RUNS_DIR / "fake")
    live, src2 = output_root("muse", "ground_truth")
    assert src2 == "live" and live.is_relative_to(LIVE_RUNS_DIR) and not live.is_relative_to(FAKE_RUNS_DIR)


# ---- running small matrices (no video, fast) ----

SMALL = [RunSpec("t1_red_in_red_bin", 100, "A"), RunSpec("t4_stack_three", 100, "A"),
         RunSpec("t4_stack_three", 100, "B"), RunSpec("t7_swap_red_blue", 100, "C")]


def read_csv(root):
    with open(root / "results.csv") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval_a")
    counts = run_matrix(SMALL, FakeGenerator(), root, workers=4, record=False)
    return root, counts


def test_small_matrix_rows_and_banner(small_run):
    root, counts = small_run
    assert counts["done"] == 4
    rows = read_csv(root)
    assert list(rows[0]) == CSV_FIELDS
    got = {(r["task"], r["condition"]): (r["success"], r["attempts"], r["failure_category"]) for r in rows}
    assert got == {
        ("t1_red_in_red_bin", "A"): ("1", "1", ""),
        ("t4_stack_three", "A"): ("0", "1", "wrong order"),
        ("t4_stack_three", "B"): ("1", "2", ""),
        ("t7_swap_red_blue", "C"): ("0", "3", "code error"),
    }
    assert {r["run_source"] for r in rows} == {"fake"} and {r["est_cost_usd"] for r in rows} == {"0.0"}
    md, charts = write_report(root)
    assert md.read_text().startswith(f"> **{FAKE_BANNER}**")
    assert [c.name for c in charts] == ["results_chart_clean.png"] and charts[0].stat().st_size > 10_000


def test_resume_skips_finished_and_redoes_unfinished(small_run):
    root, _ = small_run
    calls = []

    class Counting(FakeGenerator):
        def __call__(self, **kw):
            calls.append(kw["task"].id)
            return super().__call__(**kw)

    assert run_matrix(SMALL, Counting(), root, record=False)["skipped"] == 4
    assert calls == []
    p = run_path(root, SMALL[0]) / "run.json"
    log = json.loads(p.read_text())
    log["finished"] = False  # simulate a crash mid-run
    p.write_text(json.dumps(log))
    counts = run_matrix(SMALL, Counting(), root, record=False)
    assert counts == {"skipped": 3, "done": 1} and calls == ["t1_red_in_red_bin"]


def test_deterministic_across_runs(small_run, tmp_path):
    root_a, _ = small_run
    run_matrix(list(reversed(SMALL)), FakeGenerator(), tmp_path, workers=2, record=False)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "wall_clock_s"} for r in rows]
    assert strip(read_csv(root_a)) == strip(read_csv(tmp_path))


def test_harness_error_runs_are_redone_on_resume(tmp_path):
    spec = RunSpec("t1_red_in_red_bin", 100, "A")

    class Missing(FakeGenerator):
        def __call__(self, **kw):
            raise ReplayMiss("not in cache")

    run_matrix([spec], Missing(), tmp_path, record=False)
    assert read_csv(tmp_path)[0]["failure_category"] == "harness error"
    counts = run_matrix([spec], FakeGenerator(), tmp_path, record=False)
    assert counts == {"done": 1}
    assert read_csv(tmp_path)[0]["success"] == "1"


def test_replay_miss_is_harness_error(tmp_path):
    class Missing(FakeGenerator):
        def __call__(self, **kw):
            raise ReplayMiss("request abc not in cache (CORE_ROBOT_REPLAY=1)")

    run_matrix([RunSpec("t1_red_in_red_bin", 100, "B")], Missing(), tmp_path, record=False)
    (row,) = read_csv(tmp_path)
    assert row["success"] == "0" and row["failure_category"] == "harness error" and row["attempts"] == "1"
    log = json.loads((run_path(tmp_path, RunSpec("t1_red_in_red_bin", 100, "B")) / "run.json").read_text())
    assert "ReplayMiss" in log["attempts"][0]["generator_error"]


# ---- category labelling from synthetic logs ----

T4_INIT = json.loads((DEMOS / "t4_stack_three" / "seed_0" / "initial_state.json").read_text())


def mklog(outcome=None, calls=(), unmoved=(), extra=None, task="t4_stack_three"):
    attempt = {"outcome": {"success": False, "unmoved": list(unmoved), **(outcome or {})},
               "call_log": list(calls), "initial_state": T4_INIT, **(extra or {})}
    return {"task_id": task, "success": False, "attempts": [attempt]}


def pick(obj, ok=True, reason=""):
    return {"primitive": "pick", "args": {"obj": obj}, "ok": ok, "reason": reason}


@pytest.mark.parametrize("log,expected", [
    ({"task_id": "t4_stack_three", "success": True, "attempts": []}, ""),
    (mklog(extra={"generator_error": "ReplayMiss: x"}), "harness error"),
    (mklog({"sandbox_error": True, "exception": "sandbox process failed"}), "harness error"),
    (mklog({"timed_out": True, "exception": "timeout"}), "timeout"),
    (mklog({"rejected": True, "exception": "rejected before running"}), "code error"),
    (mklog({"exception": "ValueError: boom"}), "code error"),
    (mklog(calls=[pick("green_block", False, "grasp missed: gripper 0.050 m from object")]), "grasp failure"),
    (mklog(calls=[pick("green_block", False, "unreachable")]), "grasp failure"),
    (mklog(calls=[pick("green_block"), pick("red_block")]), "wrong object"),           # moved a block the plan doesn't
    (mklog(calls=[pick("green_block")]), "wrong object"),                              # never moved blue
    (mklog(calls=[pick("green_block"), pick("blue_block")],
           unmoved=[{"block": "red_block", "ok": False, "moved_cm": 4.0}]), "wrong object"),  # disturbed red
    (mklog(calls=[pick("blue_block"), pick("green_block")]), "wrong order"),
    (mklog(calls=[pick("green_block"), pick("blue_block")]), "placement error"),  # all ok, goal missed
    (mklog(calls=[pick("green_block"), pick("blue_block"),
                  {"primitive": "place", "args": {"obj": "blue_block", "target": "x"}, "ok": False,
                   "reason": "unknown target 'x'"}]), "other"),
])
def test_failure_category(log, expected):
    assert failure_category(log) == expected


# ---- README results only from live runs ----

def _csv(path, run_source):
    rows = [{"task": "t1_red_in_red_bin", "seed": 100, "condition": c, "success": 1, "attempts": 1,
             "plan_source": "ground_truth", "run_source": run_source, "failure_category": "",
             "wall_clock_s": 1.0, "tokens_in": 0, "tokens_out": 0, "est_cost_usd": 0} for c in "ABC"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_readme_results_refuse_fake(tmp_path):
    with pytest.raises(ValueError, match="refusing"):
        build_readme_results(_csv(tmp_path / "fake.csv", "fake"))
    live = build_readme_results(_csv(tmp_path / "live.csv", "live"))
    assert FAKE_BANNER not in live and "3/3" not in live and "1/1 (100%)" in live


def test_markdown_has_t8_line_and_totals():
    rows = []
    for t, ok in (("t1_red_in_red_bin", 1), ("t2_sort_by_color", 0), ("t8_nearest_to_mug", 1)):
        for s in (100, 101):
            rows.append({"task": t, "seed": s, "condition": "A", "success": ok, "attempts": 1,
                         "plan_source": "ground_truth", "run_source": "live", "failure_category": "" if ok else "other",
                         "tokens_in": 0, "tokens_out": 0, "est_cost_usd": 0})
    md = build_markdown(rows)
    assert "| **t1-t7 total** | 2/4 (50%) |" in md
    assert "| **t8 (nearest to mug), separate** | 2/2 (100%) |" in md
    assert "| **overall t1-t8** | 4/6 (67%) |" in md


# ---- predicted plans ----

def _planner_results(tmp_path, rows):
    p = tmp_path / "planner_results.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_predicted_plan_is_loaded_from_seed0_demo(tmp_path):
    from core_robot.demos.tasks import TASKS_BY_ID
    from core_robot.eval.run_eval import predicted_plan

    plan = {"objects": [{"name": "red_block", "color": "red", "kind": "block"},
                        {"name": "blue_bin", "color": "blue", "kind": "bin"}],
            "steps": [{"index": 0, "action": "pick_and_place", "object": "red_block", "destination": "blue_bin"}],
            "goal": [{"subject": "red_block", "relation": "in", "target": "blue_bin"}]}
    p = _planner_results(tmp_path, [{"demo": "t1_red_in_red_bin/seed_0", "plan": plan},
                                    {"demo": "t1_red_in_red_bin/seed_1", "plan": None}])
    got = predicted_plan(TASKS_BY_ID["t1_red_in_red_bin"], p)
    assert got.steps[0].destination == "blue_bin"  # the (wrong) prediction, not the ground truth


@pytest.mark.parametrize("rows,needle", [
    ([{"demo": "t1_red_in_red_bin/seed_0", "plan": None, "planner_error": "empty reply"}], "no valid plan"),
    ([], "no planner result"),
])
def test_missing_predicted_plan_is_never_replaced_by_ground_truth(tmp_path, monkeypatch, rows, needle):
    from core_robot.eval import run_eval

    p = _planner_results(tmp_path, rows)
    real = run_eval.predicted_plan
    monkeypatch.setattr(run_eval, "predicted_plan", lambda task, path=None: real(task, p))
    calls = []

    class Recording(FakeGenerator):
        def __call__(self, **kw):
            calls.append(kw)
            return super().__call__(**kw)

    spec = RunSpec("t1_red_in_red_bin", 100, "A")
    run_matrix([spec], Recording(), tmp_path / "out", plan_source="predicted", record=False)
    (row,) = read_csv(tmp_path / "out")
    assert row["success"] == "0" and row["failure_category"] == "harness error" and row["plan_source"] == "predicted"
    log = json.loads((run_path(tmp_path / "out", spec) / "run.json").read_text())
    assert needle in log["plan_error"] and log["plan"] is None
    assert calls == []  # nothing was generated from a substitute plan


def test_budget_stops_new_runs(tmp_path):
    counts = run_matrix(SMALL, FakeGenerator(), tmp_path, record=False, over_budget=lambda: True)
    assert counts == {"over_budget": 4}
    assert not list(tmp_path.glob("*/*/seed_*/run.json"))


def test_clean_and_perturbed_runs_do_not_collide(tmp_path):
    spec = RunSpec("t3_stack_green_on_red", 200, "A")
    run_matrix([spec], FakeGenerator(), tmp_path, record=False)
    counts = run_matrix([spec], FakeGenerator(), tmp_path, record=False, offset_cm=4)
    assert counts == {"done": 1}  # the clean run did not count as the perturbed one
    assert (run_path(tmp_path, spec) / "run.json").exists()
    assert (run_path(tmp_path, spec, "perturbed") / "run.json").exists()
    rows = read_csv(tmp_path)
    assert sorted(r["environment"] for r in rows) == ["clean", "perturbed"]
    by_env = {r["environment"]: r for r in rows}
    assert by_env["clean"]["success"] == "1" and by_env["perturbed"]["success"] == "0"  # 4 cm breaks the stack



# ---- report: paired counts, captions, incomplete cells ----

def _row(task, seed, cond, ok, env="clean", cat=""):
    return {"task": task, "seed": seed, "condition": cond, "environment": env, "success": ok, "attempts": 1,
            "plan_source": "predicted", "plan_strict": "1", "plan_lenient": "1", "run_source": "live",
            "failure_category": cat, "prompt_hash": "6faa3b19e9b6", "git_commit": "abc", "prompt_change_allowed": "0",
            "tokens_in": 0, "tokens_out": 0, "est_cost_usd": 0}


def test_paired_counts_fixed_and_broke():
    from core_robot.eval.report import paired_counts

    t = "t3_stack_green_on_red"
    rows = [_row(t, 100, "A", 0), _row(t, 100, "B", 1), _row(t, 100, "C", 0),
            _row(t, 101, "A", 1), _row(t, 101, "B", 0), _row(t, 101, "C", 1),
            _row(t, 102, "B", 1)]  # no A for seed 102: not a pair
    pc = paired_counts(rows)
    assert pc[(t, "B")] == {"fixed": 1, "broke": 1, "pairs": 2}
    assert pc[(t, "C")] == {"fixed": 0, "broke": 0, "pairs": 2}


def test_report_tables_carry_environment_hash_date_and_flag_missing_runs():
    import datetime

    rows = [_row("t1_red_in_red_bin", s, c, 1) for s in range(100, 105) for c in "ABC"]
    rows += [_row("t1_red_in_red_bin", s, "A", 0, "perturbed", "placement error") for s in range(100, 105)]
    rows += [_row("t1_red_in_red_bin", 100, "B", 1, "perturbed")]
    md = build_markdown(rows)
    today = datetime.date.today().isoformat()
    for env in ("clean", "perturbed"):
        assert f"Environment: {env}; prompt hash: 6faa3b19e9b6; date: {today}" in md
    assert md.count("prompt hash: 6faa3b19e9b6") == 8  # 4 tables per environment, each captioned
    assert "Incomplete: 4 runs of this environment were not run" in md  # perturbed B has 1 of 5
    assert "| placement error | 5 | 0 |" in md
    assert "| **total** | 1 | 0 | 1 |" in md  # perturbed: B fixed 1 of 1 pair
