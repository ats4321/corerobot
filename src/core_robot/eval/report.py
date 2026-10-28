"""results.csv -> results.md + grouped bar chart (results_chart.png).

Fake-generator results carry the FAKE_BANNER at the top of results.md and in the chart
title. README/docs result sections are built only from live runs (build_readme_results).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from core_robot.demos.tasks import TASKS

FAKE_BANNER = "FAKE GENERATOR: harness test only, not a result"
CONDITIONS = ("A", "B", "C")
CATEGORIES = ("wrong object", "wrong order", "grasp failure", "code error", "timeout", "harness error", "other")
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


def _env_section(rows: list[dict], heading: str | None) -> list[str]:
    by = defaultdict(list)
    for r in rows:
        by[(r["task"], r["condition"])].append(r)
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
    tasks = [t.id for t in TASKS if any(r["task"] == t.id for r in rows)]
    predicted = any(r["plan_source"] == "predicted" for r in rows)
    lines = [f"## {heading}", ""] if heading else []
    head = ["Task"] + (["Plan strict", "Plan lenient"] if predicted else []) + \
        [f"Task success, {CONDITION_LABELS[c]}" for c in conds]
    lines += ["### Success by task and condition", "", "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]

    def plan_cols(ts):
        if not predicted:
            return []
        per = [[r for r in rows if r["task"] == t] for t in ts]
        s = sum(_plan_cell(p, "plan_strict") == "yes" for p in per)
        l = sum(_plan_cell(p, "plan_lenient") == "yes" for p in per)
        if len(ts) == 1:
            return [_plan_cell(per[0], "plan_strict"), _plan_cell(per[0], "plan_lenient")]
        return [f"{s}/{len(ts)}", f"{l}/{len(ts)}"]

    for t in tasks:
        lines.append(f"| {t} | " + " | ".join(plan_cols([t]) + [_frac(by[(t, c)]) for c in conds]) + " |")
    core = [t for t in tasks if t != T8]
    total = lambda ts, c: _frac([r for t in ts for r in by[(t, c)]])
    for label, ts in (("**t1-t7 total**", core), ("**t8 (nearest to mug), separate**", [t for t in tasks if t == T8]),
                      ("**overall t1-t8**", tasks)):
        if ts:
            lines.append(f"| {label} | " + " | ".join(plan_cols(ts) + [total(ts, c) for c in conds]) + " |")
    lines += ["", "### Failure categories (failed runs, last attempt)", "",
              "| Category | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for cat in CATEGORIES:
        counts = [sum(1 for r in rows if r["condition"] == c and r["failure_category"] == cat) for c in conds]
        lines.append(f"| {cat} | " + " | ".join(str(n) for n in counts) + " |")
    lines += ["", "### Mean attempts to success", "", "| Condition | Mean attempts (successful runs) |", "|---|---|"]
    for c in conds:
        ok = [int(r["attempts"]) for r in rows if r["condition"] == c and int(r["success"])]
        lines.append(f"| {CONDITION_LABELS[c]} | " + (f"{sum(ok) / len(ok):.2f} (n={len(ok)})" if ok else "- (n=0)") + " |")
    return lines + [""]


def build_markdown(rows: list[dict], chart_name: str = "results_chart.png") -> str:
    lines = []
    if _is_fake(rows):
        lines += [f"> **{FAKE_BANNER}**", ""]
    plan_sources = sorted({r["plan_source"] for r in rows})
    envs = [e for e in ENV_LABELS if any(_env(r) == e for r in rows)]
    lines += ["# Evaluation results", "",
              f"{len(rows)} runs; plan source: {', '.join(plan_sources)}; "
              f"run source: {', '.join(sorted({r['run_source'] for r in rows}))}; "
              f"environments: {', '.join(envs)}.", ""]
    if any(r["plan_source"] == "predicted" for r in rows):
        lines += ["Plan strict / lenient: does the predicted plan (from the task's seed-0 demo) match the "
                  "ground-truth plan. Task success: did the run reach the goal in the simulator. "
                  "They are separate measurements.", ""]
    for e in envs:
        lines += _env_section([r for r in rows if _env(r) == e], ENV_LABELS[e])
    cost = sum(float(r["est_cost_usd"]) for r in rows)
    tok_in = sum(int(r["tokens_in"]) for r in rows)
    tok_out = sum(int(r["tokens_out"]) for r in rows)
    lines += [f"Tokens: {tok_in} in, {tok_out} out; estimated cost ${cost:.2f}.", "",
              f"![Success rate by task and condition]({chart_name})", ""]
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
                rate[(t, c)] = 100 * sum(sel) / len(sel) if sel else 0
        ax.set_facecolor(SURFACE)
        width = 0.8 / len(conds)
        for i, c in enumerate(conds):
            xs = [k + (i - (len(conds) - 1) / 2) * width for k in range(len(tasks))]
            vals = [rate[(t, c)] for t in tasks]
            ax.bar(xs, vals, width=width - 0.03, color=SERIES_COLORS[c], label=CONDITION_LABELS[c],
                   edgecolor=SURFACE, linewidth=1, zorder=3)
            for x, v in zip(xs, vals):
                if v == 0:  # an empty bar reads as missing data unless labelled
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


def write_report(root: Path) -> tuple[Path, Path]:
    rows = load_rows(root / "results.csv")
    md = root / "results.md"
    md.write_text(build_markdown(rows))
    return md, build_chart(rows, root / "results_chart.png")


def build_readme_results(csv_path: Path) -> str:
    """Results section for README/docs. Refuses anything that is not a live run."""
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    bad = sorted({r["run_source"] for r in rows if r["run_source"] != "live"})
    if bad:
        raise ValueError(f"refusing to build README results from non-live runs (run_source={bad}) in {csv_path}")
    return build_markdown(rows)
