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
        f"- note (optional, at most {NOTE_MAX_LEN} characters): only when which object was chosen depends on "
        "a spatial relationship rather than its identity, describe the rule (for example, which object it was "
        "nearest to). Otherwise omit it.",
        "- goal: the relations that hold at the end and capture what the demonstration was for. "
        f"Allowed relations: {', '.join(RELATIONS)}.",
        "", "## Relation definitions", relation_definitions(),
    ])


def plan_from_video(client, clip: str | Path, perception: dict, max_output_tokens: int = 6000) -> dict:
    """{"plan": Plan | None, "attempts": [{"reply", "error"}], "tokens": (in, out), "error": str | None}."""
    schema = Plan.model_json_schema()
    base = planner_prompt(perception)
    attempts, t_in, t_out = [], 0, 0
    prompt = base
    for _ in range(2):  # first try + one retry on a validation error
        r = client.muse(prompt, video=Path(clip), schema=schema, system=PLANNER_SYSTEM,
                        max_output_tokens=max_output_tokens)
        t_in, t_out = t_in + r["usage"][0], t_out + r["usage"][1]
        error = r.get("json_error")
        plan = None
        if error is None:
            try:
                plan = Plan.model_validate(r["json"])
            except ValidationError as e:
                error = "; ".join(f"{'.'.join(str(x) for x in d['loc'])}: {d['msg']}" for d in e.errors())
        attempts.append({"reply": r["text"][:8000], "error": error})
        if plan is not None:
            return {"plan": plan, "attempts": attempts, "tokens": (t_in, t_out), "error": None}
        prompt = (base + "\n\n## Your previous reply was rejected\n" + error
                  + "\nPrevious reply:\n" + r["text"][:4000] + "\nReturn corrected JSON that satisfies the schema.")
    return {"plan": None, "attempts": attempts, "tokens": (t_in, t_out), "error": attempts[-1]["error"]}
