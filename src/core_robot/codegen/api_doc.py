"""Primitives reference for the code-generation prompt, generated from the source.

Signatures and docstrings come from sim/primitives.py (Robot) and the success rule from
verify/relations.py, so the prompt never drifts from the code that actually runs.
"""

from __future__ import annotations

import inspect

from core_robot.codegen.sandbox import SAFE_BUILTINS
from core_robot.sim.primitives import Robot
from core_robot.verify import relations

PRIMITIVES = ("pick", "place", "move_above", "open_gripper")

CONTRACT = """\
Write Python that defines exactly one entry point:

    def run(robot, scene):
        ...

robot: the only way to act. It has exactly four methods, documented below. Nothing else
    on it is accessible.
scene: read-only data, not a primitive. A copy of the starting scene for this layout:
    {name: {"position": [x, y, z], "orientation": [qx, qy, qz, qw], "color": str, "kind": str}}
    kind is "block", "bin" or "mug". Positions are meters: table top z = 0, +x away from
    the robot, +y to the robot's left. Changing scene does not change the world; it is
    not updated as objects move.

Rules: no imports; no names or attributes starting with "_"; no open/exec/eval/getattr.
Available builtins: {builtins}.
The run has a time limit. Check each primitive's "ok" before continuing."""


def _signature(name: str) -> str:
    sig = inspect.signature(getattr(Robot, name))
    params = [p for p in sig.parameters.values() if p.name != "self"]
    return f"robot.{name}({', '.join(str(p).split(':')[0] for p in params)})"


def primitives_reference() -> str:
    """The full reference text: contract, robot primitives, and the global success rule."""
    parts = [CONTRACT.replace("{builtins}", ", ".join(SAFE_BUILTINS)), "", "## Robot primitives", ""]
    for name in PRIMITIVES:
        doc = inspect.getdoc(getattr(Robot, name))
        parts += [f"### {_signature(name)}", "", doc, ""]
    parts += [
        "## Goal relations",
        "",
        inspect.getdoc(relations).split("Relations (subject relation target):", 1)[1].strip("\n"),
        "",
        "## Success rule",
        "",
        "Success = every goal relation holds AND:",
        inspect.getdoc(relations.unmoved_blocks_still),
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    print(primitives_reference())
