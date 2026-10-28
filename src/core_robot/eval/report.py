"""results.csv -> results.md + one grouped bar chart per environment (results_chart_<env>.png).

Fake-generator results carry the FAKE_BANNER at the top of results.md and in the chart
title. README/docs result sections are built only from live runs (build_readme_results).
"""

from __future__ import annotations

import csv
import datetime
from collections import defaultdict
from pathlib import Path

from core_robot.demos.tasks import TASKS

FAKE_BANNER = "FAKE GENERATOR: harness test only, not a result"
CONDITIONS = ("A", "B", "C")
CATEGORIES = ("wrong object", "wrong order", "placement error", "grasp failure", "code error", "timeout",
              "harness error", "other")
SEEDS_PER_CELL = 5
CONDITION_LABELS = {"A": "A one-shot", "B": "B repair, text", "C": "C repair, text + frame"}
# Categorical slots 1-3 of the dataviz reference palette (validated: adjacent CVD dE >= 9.1 on light).
SERIES_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
T8 = "t8_nearest_to_mug"


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _frac(rows) -> str:
    n, k = len(rows), sum(int(r["success"]) for r in rows)
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def _is_fake(rows) -> bool:
    return any(r["run_source"] != "live" for r in rows)


ENV_LABELS = {"clean": "Clean environment", "perturbed": "Perturbed environment (calibration_offset)"}


def _env(r) -> str:
    return r.get("environment") or "clean"


def _plan_cell(task_rows, key) -> str:
    vals = {r.get(key, "") for r in task_rows} - {""}
    return "-" if not vals else ("yes" if vals == {"1"} or vals == {1} else "no")


def _caption(rows: list[dict], env: str, what: str) -> str:
    hashes = sorted({r.get("prompt_hash") or "unrecorded" for r in rows})
    return (f"*{what}. Environment: {env}; prompt hash: {', '.join(hashes)}; "
            f"date: {datetime.date.today().isoformat()}.*")


def paired_counts(rows: list[dict]) -> dict:
    """Per task and condition X in B/C, against A on the same seed: fixed (A failed, X succeeded),
    broke (A succeeded, X failed), and how many pairs exist."""
    a = {(r["task"], r["seed"]): int(r["success"]) for r in rows if r["condition"] == "A"}
    out: dict = defaultdict(lambda: {"fixed": 0, "broke": 0, "pairs": 0})
    for r in rows:
        if r["condition"] not in ("B", "C") or (r["task"], r["seed"]) not in a:
            continue
        c = out[(r["task"], r["condition"])]
        c["pairs"] += 1
        sa, sx = a[(r["task"], r["seed"])], int(r["success"])
        c["fixed"] += int(not sa and sx)
        c["broke"] += int(sa and not sx)
    return dict(out)


def _env_section(rows: list[dict], env: str, chart_name: str | None = None) -> list[str]:
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["condition"])].append(r)
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    tasks = [t.id for t in TASKS if any(r["task"] == t.id for r in rows)]
    predicted = any(r["plan_source"] == "predicted" for r in rows)
    label = ENV_LABELS.get(env, env)
    missing = sum(max(0, SEEDS_PER_CELL - len(by[(t, c)])) for t in tasks for c in conds)
    lines = [f"## {label}", ""]
    if missing:
        lines += [f"**Incomplete: {missing} runs of this environment were not run** "
                  f"(cells show k/n over the runs that exist; n < {SEEDS_PER_CELL} means missing runs).", ""]
    head = ["Task"] + (["Plan strict", "Plan lenient"] if predicted else []) + \
        [f"Task success, {CONDITION_LABELS[c]}" for c in conds]
    lines += ["### Success by task and condition", "", _caption(rows, env, "Task success k/n"), "",
              "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]

    def plan_cols(ts):
        if not predicted:
            return []
        per = [[r for r in rows if r["task"] == t] for t in ts]
        if len(ts) == 1:
            return [_plan_cell(per[0], "plan_strict"), _plan_cell(per[0], "plan_lenient")]
        s = sum(_plan_cell(p, "plan_strict") == "yes" for p in per)
        l = sum(_plan_cell(p, "plan_lenient") == "yes" for p in per)
        return [f"{s}/{len(ts)}", f"{l}/{len(ts)}"]

    for t in tasks:
        lines.append(f"| {t} | " + " | ".join(plan_cols([t]) + [_frac(by[(t, c)]) for c in conds]) + " |")
    core = [t for t in tasks if t != T8]
    total = lambda ts, c: _frac([r for t in ts for r in by[(t, c)]])
    for name, ts in (("**t1-t7 total**", core), ("**t8 (nearest to mug), separate**", [t for t in tasks if t == T8]),
                     ("**overall t1-t8**", tasks)):
        if ts:
            lines.append(f"| {name} | " + " | ".join(plan_cols(ts) + [total(ts, c) for c in conds]) + " |")

    lines += ["", "### Mean attempts to success", "", _caption(rows, env, "Mean attempts over successful runs"), "",
              "| Condition | Mean attempts (successful runs) |", "|---|---|"]
    for c in conds:
        ok = [int(r["attempts"]) for r in rows if r["condition"] == c and int(r["success"])]
        lines.append(f"| {CONDITION_LABELS[c]} | " + (f"{sum(ok) / len(ok):.2f} (n={len(ok)})" if ok else "- (n=0)") + " |")

    pc = paired_counts(rows)
    rep = [c for c in ("B", "C") if c in conds]
    if "A" in conds and rep:
        lines += ["", "### Repair versus A (paired by task and seed)", "",
                  _caption(rows, env, "Fixed: A failed and the repair condition succeeded; broke: A succeeded "
                                      "and the repair condition failed; pairs: seeds run in both"), "",
                  "| Task | " + " | ".join(f"{c} fixed | {c} broke | {c} pairs" for c in rep) + " |",
                  "|---" * (1 + 3 * len(rep)) + "|"]
        for t in tasks:
            cells = []
            for c in rep:
                v = pc.get((t, c), {"fixed": 0, "broke": 0, "pairs": 0})
                cells += [str(v["fixed"]), str(v["broke"]), str(v["pairs"])]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
        cells = []
        for c in rep:
            vs = [v for (t, cc), v in pc.items() if cc == c]
            cells += [str(sum(v[k] for v in vs)) for k in ("fixed", "broke", "pairs")]
        lines.append("| **total** | " + " | ".join(cells) + " |")

    lines += ["", "### Failure categories (failed runs, last attempt)", "",
              _caption(rows, env, "Failed runs by category"), "",
              "| Category | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for cat in CATEGORIES:
        counts = [sum(1 for r in rows if r["condition"] == c and r["failure_category"] == cat) for c in conds]
        lines.append(f"| {cat} | " + " | ".join(str(n) for n in counts) + " |")
    if chart_name:
        lines += ["", f"![Task success rate by condition, {env}]({chart_name})"]
    return lines + [""]


def build_markdown(rows: list[dict], chart_names: dict | None = None) -> str:
    lines = []
    if _is_fake(rows):
        lines += [f"> **{FAKE_BANNER}**", ""]
    plan_sources = sorted({r["plan_source"] for r in rows})
    envs = [e for e in ENV_LABELS if any(_env(r) == e for r in rows)]
    lines += ["# Evaluation results", "",
              f"{len(rows)} runs; plan source: {', '.join(plan_sources)}; "
              f"run source: {', '.join(sorted({r['run_source'] for r in rows}))}; "
              f"environments: {', '.join(envs)}; generated {datetime.date.today().isoformat()}.", ""]
    if any(r["plan_source"] == "predicted" for r in rows):
        lines += ["Plan strict / lenient: does the predicted plan (from the task's seed-0 demo) match the "
                  "ground-truth plan. Task success: did the run reach the goal in the simulator. "
                  "They are separate measurements.", ""]
    for e in envs:
        lines += _env_section([r for r in rows if _env(r) == e], e, (chart_names or {}).get(e))
    cost = sum(float(r["est_cost_usd"]) for r in rows)
    tok_in = sum(int(r["tokens_in"]) for r in rows)
    tok_out = sum(int(r["tokens_out"]) for r in rows)
    lines += [f"Tokens recorded across runs: {tok_in} in, {tok_out} out (${cost:.2f} at list price). This counts "
              "every attempt of every run, including attempts served from the cache and shared between "
              "conditions or environments, so it overstates actual spend; see scripts/spend.py.", ""]
    return "\n".join(lines)


def build_chart(rows: list[dict], path: Path) -> Path:
    """Grouped bars (one per condition) of success rate per task; one panel per environment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    envs = [e for e in ENV_LABELS if any(_env(r) == e for r in rows)]
    fig, axes = plt.subplots(len(envs), 1, figsize=(11, 4.6 * len(envs)), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, env in zip(axes[:, 0], envs):
        erows = [r for r in rows if _env(r) == env]
        conds = [c for c in CONDITIONS if any(r["condition"] == c for r in erows)]
        tasks = [t.id for t in TASKS if any(r["task"] == t.id for r in erows)]
        rate = {}
        for t in tasks:
            for c in conds:
                sel = [int(r["success"]) for r in erows if r["task"] == t and r["condition"] == c]
                rate[(t, c)] = 100 * sum(sel) / len(sel) if sel else None  # None: not run
        ax.set_facecolor(SURFACE)
        width = 0.8 / len(conds)
        for i, c in enumerate(conds):
            xs = [k + (i - (len(conds) - 1) / 2) * width for k in range(len(tasks))]
            vals = [rate[(t, c)] for t in tasks]
            ax.bar(xs, [v or 0 for v in vals], width=width - 0.03, color=SERIES_COLORS[c], label=CONDITION_LABELS[c],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
            for x, v in zip(xs, vals):
                if v is None:  # never run: say so, never draw it as 0%
                    ax.text(x, 1.5, "not\nrun", ha="center", va="bottom", fontsize=6, color=TEXT_2, zorder=4)
                elif v == 0:  # an empty bar reads as missing data unless labelled
                    ax.text(x, 1.5, "0%", ha="center", va="bottom", fontsize=6.5, color=TEXT_2, zorder=4)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t.split("_", 1)[0] + "\n" + t.split("_", 1)[1].replace("_", " ") for t in tasks],
                           fontsize=8, color=TEXT_2)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Success rate (%)", color=TEXT_2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=TEXT_2, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        title = f"Task success rate by condition: {ENV_LABELS[env].lower()}"
        if _is_fake(rows):
            title = f"{FAKE_BANNER}\n{title}"
        ax.set_title(title, color=TEXT, fontsize=11, loc="left", fontweight="bold" if _is_fake(rows) else "normal")
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  ncol=len(conds))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_report(root: Path) -> tuple[Path, list[Path]]:
    """results.md plus one chart per environment (results_chart_<env>.png)."""
    rows = load_rows(root / "results.csv")
    charts = {}
    for env in [e for e in ENV_LABELS if any(_env(r) == e for r in rows)]:
        charts[env] = build_chart([r for r in rows if _env(r) == env], root / f"results_chart_{env}.png")
    md = root / "results.md"
    md.write_text(build_markdown(rows, {e: p.name for e, p in charts.items()}))
    return md, list(charts.values())


def build_readme_results(csv_path: Path) -> str:
    """Results section for README/docs. Refuses anything that is not a live run."""
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    bad = sorted({r["run_source"] for r in rows if r["run_source"] != "live"})
    if bad:
        raise ValueError(f"refusing to build README results from non-live runs (run_source={bad}) in {csv_path}")
    return build_markdown(rows)
