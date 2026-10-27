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
