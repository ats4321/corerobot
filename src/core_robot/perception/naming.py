"""Object naming: Muse Spark lists which simulator objects appear in the first demo frame.

The answer is constrained (structured output) to simulator object names. The vocabulary
describes what each name looks like; it says nothing about the task.
"""

from __future__ import annotations

from pathlib import Path

from core_robot.sim.env import OBJECTS

DESCRIPTIONS = {
    "block": "a small {color} cube (4 cm)",
    "bin": "a {color} open-top square bin (16 cm wide, low walls)",
    "mug": "a {color} cylinder (a mug)",
}
VOCABULARY = {o.name: DESCRIPTIONS[o.kind].format(color=o.color) for o in OBJECTS}

NAMING_SYSTEM = "You identify objects on a table in a robot simulation image. Answer only with the given names."


def naming_schema() -> dict:
    return {
        "type": "object",
        "properties": {"objects": {"type": "array", "items": {"type": "string", "enum": list(VOCABULARY)}}},
        "required": ["objects"],
        "additionalProperties": False,
    }


def naming_prompt() -> str:
    lines = ["Which of these objects are visible on the table in this image? Ignore the robot arm.", ""]
    lines += [f"- {name}: {desc}" for name, desc in VOCABULARY.items()]
    lines += ["", 'Reply with JSON {"objects": [names...]} listing every visible object exactly once.']
    return "\n".join(lines)


def name_objects(client, first_frame: str | Path) -> dict:
    """{"objects": [simulator names, vocabulary order], "tokens": (in, out), "error": str | None}."""
    r = client.muse(naming_prompt(), images=[Path(first_frame)], schema=naming_schema(), system=NAMING_SYSTEM,
                    max_output_tokens=2000)
    data = r.get("json")
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        return {"objects": [], "tokens": r["usage"], "error": r.get("json_error") or "no objects list in reply"}
    seen = {n for n in data["objects"] if n in VOCABULARY}
    unknown = sorted({str(n) for n in data["objects"]} - set(VOCABULARY))
    return {"objects": [n for n in VOCABULARY if n in seen], "tokens": r["usage"],
            "error": f"ignored unknown names {unknown}" if unknown else None}
