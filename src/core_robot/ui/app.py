"""Core Robot UI (offline): browse demos, plans, generated code and repair attempts.

    python -m core_robot.ui.app

Browses demos from data/demos/ and evaluation run logs from data/runs/eval/ (live) or
data/runs/fake/ (fake generator, clearly labelled). The "Run on your own video" section runs the
real-clip pipeline (core_robot.real.pipeline) on an upload, stage by stage. Nothing calls the
Meta API unless "Replay from cache" is switched off; the live video pipeline also requires the
confirmation box and is capped by the budget guard.
"""

from __future__ import annotations

import json
from pathlib import Path

from core_robot.config import DATA_DIR, DEMOS_DIR, FAKE_RUNS_DIR, LIVE_RUNS_DIR, exec_seeds, load_settings
from core_robot.demos.tasks import TASKS_BY_ID
from core_robot.eval.report import FAKE_BANNER

MAX_ATTEMPTS_SHOWN = 3


LIVE_UI_RUN_SET = "runs/eval/ui"


def run_live(task_id: str, condition: str, seed: int) -> str | None:
    """Live code generation (Muse Spark) for one repair run, using the ground-truth plan for
    the execution layout. Writes a run log under data/runs/eval/ui/. Returns an error or None.
    Costs real money (at most 3 generations); cached requests are free."""
    from core_robot.codegen.muse_generator import MuseGenerator
    from core_robot.codegen.scene import exec_initial_state
    from core_robot.meta_client import MetaClient
    from core_robot.repair.loop import run_with_repair

    settings = load_settings()
    if not settings.api_key and not settings.replay:
        return "Live mode needs MODEL_API_KEY in .env (or CORE_ROBOT_REPLAY=1 to replay cached responses)."
    task = TASKS_BY_ID[task_id]
    plan = task.ground_truth_plan(exec_initial_state(task, seed))
    out = DATA_DIR / LIVE_UI_RUN_SET / task_id / condition / f"seed_{seed}"
    meta = {"run_source": "live", "plan_source": "ground_truth", "generator": "muse"}
    run_with_repair(task, plan, seed, condition, MuseGenerator(MetaClient(settings=settings)), out_dir=out, meta=meta)
    return None


# ---- artifact loaders (pure functions; tested without a browser) ----

def list_demos() -> list[str]:
    return sorted(f"{p.parent.name}/{p.name}" for p in DEMOS_DIR.glob("*/seed_*") if (p / "video.mp4").exists())


def list_run_sets() -> list[str]:
    """Evaluation output directories that contain run logs, live first."""
    sets = []
    for base in (LIVE_RUNS_DIR, FAKE_RUNS_DIR):
        if base.exists():
            sets += sorted(str(p.relative_to(DATA_DIR)) for p in base.iterdir() if p.is_dir())
    return sets


def load_demo(demo_id: str) -> dict:
    """{"video", "plan", "error"} for a demo id like "t4_stack_three/seed_0"."""
    if not demo_id:
        return {"video": None, "plan": None, "error": "Choose a demo."}
    if demo_id not in list_demos():  # only known demo ids; never build paths from free text
        return {"video": None, "plan": None, "error": f"Demo not found: {demo_id}"}
    d = DEMOS_DIR / demo_id
    if not (d / "video.mp4").exists():
        return {"video": None, "plan": None, "error": f"Demo not found: {demo_id}"}
    try:
        plan = json.loads((d / "ground_truth_plan.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"video": str(d / "video.mp4"), "plan": None, "error": f"Could not read the demo plan: {e}"}
    return {"video": str(d / "video.mp4"), "plan": plan, "error": None}


def load_run(run_set: str, task_id: str, condition: str, seed, replay: bool = True) -> dict:
    """{"error", "banner", "summary", "plan", "attempts": [{code, video, report, outcome}]}."""
    empty = {"error": None, "banner": "", "summary": "", "plan": None, "attempts": []}
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return {**empty, "error": f"Execution seed must be an integer, got {seed!r}."}
    if seed not in exec_seeds():
        return {**empty, "error": f"Execution seed {seed} is not an evaluation seed; use one of {exec_seeds()}."}
    if not run_set and replay:
        return {**empty, "error": "No evaluation runs found. Run: python -m core_robot.eval.run_eval --generator fake"}
    rs = Path(run_set or "")
    if replay and (rs.is_absolute() or ".." in rs.parts or rs.parts[:1] != ("runs",)):
        return {**empty, "error": f"Invalid run set: {run_set!r}"}
    if task_id not in TASKS_BY_ID or condition not in ("A", "B", "C"):
        return {**empty, "error": f"Unknown task or condition: {task_id!r}, {condition!r}"}
    if not replay:
        try:
            err = run_live(task_id, condition, seed)
        except Exception as e:  # API errors etc. are shown on the page
            err = f"{type(e).__name__}: {e}"
        if err:
            return {**empty, "error": f"Live mode: {err}"}
        run_set = LIVE_UI_RUN_SET
    path = DATA_DIR / run_set / task_id / condition / f"seed_{seed}" / "run.json"
    if not path.exists():
        return {**empty, "error": f"No run log for {task_id}, condition {condition}, seed {seed} in {run_set}."}
    try:
        log = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {**empty, "error": f"Could not read {path}: {e}"}
    banner = FAKE_BANNER if log.get("run_source") != "live" else ""
    status = "SUCCESS" if log.get("success") else "FAILED"
    summary = (f"**{status}** after {log.get('attempts_used', 0)} attempt(s); condition {condition}, "
               f"seed {seed}, plan source {log.get('plan_source', '?')}, run source {log.get('run_source', '?')}.")
    attempts = []
    for a in log.get("attempts", [])[:MAX_ATTEMPTS_SHOWN]:
        out = a.get("outcome", {})
        goals = "; ".join(_goal_text(g) for g in out.get("per_goal", []))
        verdict = "success" if out.get("success") else "failed"
        attempts.append({
            "code": a.get("code") or "",
            "video": a.get("video") if a.get("video") and Path(a["video"]).exists() else None,
            "report": a.get("report") or a.get("generator_error") or "",
            "outcome": f"Attempt {a['attempt']}: {verdict}" + (f" - {goals}" if goals else ""),
            "image_given": a.get("image_given"),
            "feedback_given": bool(a.get("feedback_given")),
        })
    return {**empty, "banner": banner, "summary": summary, "plan": log.get("plan"), "attempts": attempts}


# ---- page ----

def build_app():
    import gradio as gr

    demos, run_sets = list_demos(), list_run_sets()

    def show_demo(demo_id):
        d = load_demo(demo_id)
        return d["video"], d["plan"], _err(d["error"])

    def show_run(run_set, demo_id, condition, seed, replay):
        try:
            task_id = (demo_id or "").split("/")[0]
            r = load_run(run_set, task_id, condition, seed, replay)
        except Exception as e:  # never crash the page
            r = {"error": f"Unexpected error: {type(e).__name__}: {e}", "banner": "", "summary": "",
                 "plan": None, "attempts": []}
        header = ""
        if r["banner"]:
            header += f"### ⚠ {r['banner']}\n\n"
        header += _err(r["error"]) if r["error"] else r["summary"]
        outs = [header]
        for i in range(MAX_ATTEMPTS_SHOWN):
            a = r["attempts"][i] if i < len(r["attempts"]) else None
            outs += [
                gr.update(visible=a is not None),
                f"#### {a['outcome']}" if a else "",
                a["video"] if a else None,
                a["code"] if a else "",
                a["report"] if a else "",
            ]
        return outs

    with gr.Blocks(title="Core Robot") as app:
        gr.Markdown("# Core Robot\nLearn a tabletop task from one demonstration, then perform it in simulation. "
                    "Offline viewer: everything shown is read from saved artifacts.")
        with gr.Row():
            default_demo = "t4_stack_three/seed_0" if "t4_stack_three/seed_0" in demos else (demos[0] if demos else None)
            demo_dd = gr.Dropdown(demos, value=default_demo, label="Demonstration")
            runs_dd = gr.Dropdown(run_sets, value=run_sets[0] if run_sets else None, label="Evaluation run set")
        demo_status = gr.Markdown()
        with gr.Row():
            demo_video = gr.Video(label="Demo video (angled camera)", interactive=False)
            plan_json = gr.JSON(label="Plan")
        with gr.Row():
            cond = gr.Radio(["A", "B", "C"], value="C", label="Condition",
                            info="A one-shot; B repair with text report; C repair with report + frame")
            seed = gr.Number(value=exec_seeds()[0], precision=0, label="Execution seed")
            replay = gr.Checkbox(value=True, label="Replay from cache")
            load_btn = gr.Button("Show run", variant="primary")
        run_status = gr.Markdown()
        attempt_outputs = []
        for i in range(MAX_ATTEMPTS_SHOWN):
            with gr.Group(visible=False) as grp:
                title = gr.Markdown()
                with gr.Row():
                    vid = gr.Video(label=f"Attempt {i + 1}: simulator video", interactive=False)
                    with gr.Column():
                        code = gr.Code(label="Generated code", language="python", interactive=False)
                        report = gr.Textbox(label="Failure report", lines=8, interactive=False)
            attempt_outputs += [grp, title, vid, code, report]

        demo_dd.change(show_demo, demo_dd, [demo_video, plan_json, demo_status])
        run_inputs = [runs_dd, demo_dd, cond, seed, replay]
        load_btn.click(show_run, run_inputs, [run_status] + attempt_outputs)
        app.load(show_demo, demo_dd, [demo_video, plan_json, demo_status])
        app.load(show_run, run_inputs, [run_status] + attempt_outputs)
        _live_section(gr, replay)
    return app


def live_view(events):
    """Turn pipeline events into the live section's component values (one tuple per event)."""
    state = {"status": [], "normalized": None, "overlay": None, "plan": None,
             "attempts": [("", None, "") for _ in range(MAX_ATTEMPTS_SHOWN)], "outcome": ""}
    for name, p in events:
        if name == "error":
            state["status"].append(_err(p["message"]))
        elif name == "normalized":
            state["normalized"] = p["path"]
            state["status"].append(f"Normalized: {p['size'][0]}x{p['size'][1]}, {p['duration_s']} s")
        elif name == "perception":
            state["overlay"] = p["overlay"]
            state["status"].append(f"Perception: named {', '.join(p['named']) or 'nothing'}; "
                                   f"phrases {p['phrases']}; coverage {p['coverage']}")
        elif name == "plan":
            state["plan"] = p["plan"]
            state["status"].append("Plan ready" if p["plan"] else _err(f"No valid plan: {p['error']}"))
        elif name == "attempt":
            k = p["attempt"] - 1
            ok = (p.get("outcome") or {}).get("success")
            if k < MAX_ATTEMPTS_SHOWN:
                state["attempts"][k] = (p.get("code") or "", p.get("video"),
                                        ("success" if ok else (p.get("report") or "")))
            state["status"].append(f"Attempt {p['attempt']}: {'success' if ok else 'failed'}")
        elif name == "outcome":
            head = "**SUCCESS**" if p.get("success") else "**FAILED**"
            body = p.get("judged_against") or p.get("reason") or ""
            goals = "\n".join(f"- {g}" for g in p.get("goal", []))
            state["outcome"] = f"{head}: {body}\n\n{goals}".strip()
        elif name == "spend":
            state["status"].append(f"Spend: ${p['spent']:.3f} of ${p['budget']:.2f}")
        flat = [v for a in state["attempts"] for v in a]
        yield ("\n\n".join(state["status"]), state["normalized"], state["overlay"], state["plan"], *flat,
               state["outcome"])


def _live_section(gr, replay):
    from core_robot.ui.live import DEFAULT_BUDGET, estimate_text, events

    gr.Markdown("## Run on your own video\nWith **Replay from cache** off this calls the Meta API (perception, "
                "planning, code generation) and costs money; with it on, only cached responses are used. Prompts "
                "are the frozen simulator prompts. Success is judged against the plan's own goal, not against "
                "what the person in the clip did.")
    with gr.Row():
        upload = gr.Video(label="Your demonstration video", sources=["upload"])
        with gr.Column():
            estimate = gr.Markdown("Upload a video to see the cost estimate.")
            budget = gr.Number(value=DEFAULT_BUDGET, label="Budget, USD (hard cap)", precision=2)
            confirm = gr.Checkbox(value=False, label="I confirm live API spend up to the budget above")
            go = gr.Button("Run pipeline", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        norm_video = gr.Video(label="Normalized clip", interactive=False)
        overlay = gr.Video(label="SAM 3.1 overlay", interactive=False)
        plan = gr.JSON(label="Plan")
    attempt_parts = []
    for i in range(MAX_ATTEMPTS_SHOWN):
        with gr.Row():
            code = gr.Code(label=f"Attempt {i + 1}: generated code", language="python", interactive=False)
            vid = gr.Video(label=f"Attempt {i + 1}: simulator video", interactive=False)
            rep = gr.Textbox(label=f"Attempt {i + 1}: outcome / failure report", lines=6, interactive=False)
        attempt_parts += [code, vid, rep]
    outcome = gr.Markdown()

    upload.change(estimate_text, upload, estimate)

    def run(video, replay_on, confirmed, budget_usd):
        yield from live_view(events(video, bool(replay_on), bool(confirmed), float(budget_usd or DEFAULT_BUDGET)))

    go.click(run, [upload, replay, confirm, budget], [status, norm_video, overlay, plan] + attempt_parts + [outcome])


def _goal_text(g: dict) -> str:
    text = f"{g['subject']} {g['relation']} {g['target']}: {'ok' if g['ok'] else 'NO'}"
    if not g["ok"] and g.get("error_cm") is not None:
        text += f" ({g['error_cm']} cm off)"
    return text


def _err(msg: str | None) -> str:
    return f"<span style='color:#b3261e'>**Error:** {msg}</span>" if msg else ""


def _ensure_video_probe() -> None:
    """Keep gr.Video working when the system ffprobe cannot execute.

    Gradio probes every output video with ffprobe and only tolerates ffprobe's own errors;
    a binary that cannot run (e.g. an x86_64 ffprobe on an arm64 Mac without Rosetta)
    raises OSError and breaks the page. Our videos are always H.264 mp4 with faststart
    (sim/render.py), which browsers play, so in that case we skip the probe for .mp4.
    """
    import subprocess

    from gradio import processing_utils

    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=10)
        return
    except FileNotFoundError:
        return  # gradio skips probing when ffmpeg is absent
    except OSError as e:
        print(f"[core_robot.ui] ffprobe is not runnable ({e}); treating .mp4 outputs as browser-playable")
        processing_utils.video_is_playable = lambda path: str(path).lower().endswith(".mp4")


def main():
    _ensure_video_probe()
    build_app().launch(allowed_paths=[str(DATA_DIR)])


if __name__ == "__main__":
    main()
