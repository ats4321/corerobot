"""Video -> Plan with Muse Spark: structured output, validated against planning.schema.Plan,
one retry with the validation error on failure.

Inputs are only the demo video and perception output (object names and SAM tracks).
Ground-truth plans are never read here and never appear in a prompt.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from pydantic import ValidationError

from core_robot.perception.naming import VOCABULARY
from core_robot.perception.perceive import summary_text
from core_robot.planning.schema import FREE_TABLE_SPACE, NOTE_MAX_LEN, RELATIONS, SIM_OBJECTS, Plan
from core_robot.verify import relations

# Generic: names no task and no object.
NOTE_RULE = ("when the choice of which object to move depends on a spatial relationship to another object "
             "rather than the object's color or name, add a note to that step describing the rule. "
             "Otherwise omit the note.")

PLANNER_SYSTEM = ("You watch a demonstration of a robot arm moving objects on a table and write down "
                  "the plan it demonstrates, as JSON matching the given schema.")


def relation_definitions() -> str:
    doc = inspect.getdoc(relations)
    return doc.split("Relations (subject relation target):", 1)[1].strip("\n")


def planner_prompt(perception: dict) -> str:
    names = perception["naming"]["objects"]
    objects = "\n".join(
        f"- {n}: {VOCABULARY[n]} (color {SIM_OBJECTS[n].color}, kind {SIM_OBJECTS[n].kind})" for n in names
    )
    return "\n".join([
        "The video shows a robot arm, filmed from behind its right shoulder. The image's left is the "
        "robot's left (+y); the far side of the table is the robot's front (+x).",
        "", "## Objects detected in the first frame (use exactly these names)", objects or "(none detected)",
        "", "## Segmentation tracks from SAM 3.1", summary_text(perception),
        "", "## How to write the plan",
        "- objects: every object listed above, with its color and kind.",
        "- steps: one pick_and_place per object the robot moves, in the order shown. index starts at 0.",
        f'- destination: the object it was put in or on, or "{FREE_TABLE_SPACE}" when it was set on the '
        "bare table. Never coordinates.",
        f"- note (at most {NOTE_MAX_LEN} characters): {NOTE_RULE}",
        "- goal: the relations that hold at the end and capture what the demonstration was for. "
        f"Allowed relations: {', '.join(RELATIONS)}.",
        "", "## Relation definitions", relation_definitions(),
    ])


FIRST_MAX_OUTPUT_TOKENS = 6000
RETRY_MAX_OUTPUT_TOKENS = 16000


def _reply_error(r: dict) -> str | None:
    """Why a reply is unusable before schema validation, or None."""
    if not (r.get("text") or "").strip():
        if r.get("incomplete_reason"):
            return f"empty reply: response {r.get('status')} ({r['incomplete_reason']}), no output text"
        return "empty reply: no output text"
    return r.get("json_error")


def plan_from_video(client, clip: str | Path, perception: dict, max_output_tokens: int = FIRST_MAX_OUTPUT_TOKENS,
                    retry_max_output_tokens: int = RETRY_MAX_OUTPUT_TOKENS,
                    reasoning_effort: str | None = None) -> dict:
    """{"plan": Plan | None, "attempts": [...], "tokens": (in, out), "error": str | None}.

    At most one retry, on an empty reply, invalid JSON, or a schema validation error. The
    retry gets a larger output budget (a reply that spent its whole budget on reasoning has
    no text). After a validation error the retry also sees the error and the previous reply.
    Every attempt records the raw reply text, status, incomplete reason, usage and budget.
    """
    schema = Plan.model_json_schema()
    base = planner_prompt(perception)
    attempts, t_in, t_out = [], 0, 0
    prompt, budget = base, max_output_tokens
    for _ in range(2):  # first try + one retry
        r = client.muse(prompt, video=Path(clip), schema=schema, system=PLANNER_SYSTEM,
                        max_output_tokens=budget, reasoning_effort=reasoning_effort)
        t_in, t_out = t_in + r["usage"][0], t_out + r["usage"][1]
        error = _reply_error(r)
        plan = None
        if error is None:
            try:
                plan = Plan.model_validate(r["json"])
            except ValidationError as e:
                error = "; ".join(f"{'.'.join(str(x) for x in d['loc'])}: {d['msg']}" for d in e.errors())
        attempts.append({"reply": r["text"], "status": r.get("status"), "incomplete_reason": r.get("incomplete_reason"),
                         "usage": list(r["usage"]), "max_output_tokens": budget,
                         "reasoning_effort": reasoning_effort, "error": error})
        if plan is not None:
            return {"plan": plan, "attempts": attempts, "tokens": (t_in, t_out), "error": None}
        budget = retry_max_output_tokens
        if (r.get("text") or "").strip():  # a reply that failed validation: show it and the error
            prompt = (base + "\n\n## Your previous reply was rejected\n" + error
                      + "\nPrevious reply:\n" + r["text"][:4000] + "\nReturn corrected JSON that satisfies the schema.")
        else:
            prompt = base
    return {"plan": None, "attempts": attempts, "tokens": (t_in, t_out), "error": attempts[-1]["error"]}
