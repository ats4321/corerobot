"""Text rendering of a Plan for the code-generation prompt."""

from __future__ import annotations

from core_robot.planning.schema import Plan


def render_plan(plan: Plan) -> str:
    """Objects, numbered steps (with any note in brackets), and goal relations as plain text."""
    lines = ["Objects:"]
    lines += [f"  - {o.name} ({o.color} {o.kind})" for o in plan.objects]
    lines.append("Steps:")
    for s in plan.steps:
        line = f"  {s.index + 1}. {s.action} {s.object} -> {s.destination}"
        if s.note:
            line += f"  [note: {s.note}]"
        lines.append(line)
    lines.append("Goal (all must hold at the end):")
    lines += [f"  - {g.subject} {g.relation} {g.target}" for g in plan.goal]
    return "\n".join(lines)


CODEGEN_SYSTEM = """\
You write Python that controls a simulated robot arm through four primitives.
Follow the API reference exactly. Do not import anything.
Reply with a JSON object {"code": "<python source>"} whose code defines run(robot, scene)."""

FREE_SPACE_HINT = """\
"free_table_space" in a step means: choose an empty table point (x, y) yourself, using the
positions in scene, and pass it to robot.place(obj, (x, y)). Reachable table area:
x from 0.32 to 0.72 m, y from -0.36 to 0.36 m. Keep about 8 cm from other objects' centers
(bins are 16 cm wide). +y is the robot's left."""


def _scene_json(scene: dict) -> str:
    import json

    rounded = {
        n: {**v, "position": [round(c, 4) for c in v["position"]],
            "orientation": [round(c, 4) for c in v["orientation"]]}
        for n, v in scene.items()
    }
    return json.dumps(rounded, indent=1, sort_keys=True)


def build_codegen_prompt(plan: Plan, scene: dict, api_reference: str, feedback: str | None = None,
                         with_image: bool = False) -> tuple[str, str]:
    """(system, user text) for one code-generation call. The task's own instruction is not
    included: the generator works only from the plan, the API and the scene."""
    parts = [
        "## Plan to carry out", render_plan(plan), "",
        "## Starting scene (read-only data passed as `scene`)", _scene_json(scene), "",
        "## Choosing table points", FREE_SPACE_HINT, "",
        "## Robot API and rules", api_reference,
    ]
    if feedback:
        parts += ["", "## Your previous attempt failed",
                  "Here is the failure report from the simulator. Fix the code so every goal holds.", feedback]
        if with_image:
            parts.append("The attached image is the final view of the previous attempt from the angled camera "
                         "(image left is the robot's left, +y).")
    parts += ["", 'Reply with JSON {"code": "..."} only.']
    return CODEGEN_SYSTEM, "\n".join(parts)
