"""Failure report fed back to the code generator on a failed attempt.

The final-frame images (angled and side cameras) are rendered and saved by the sandbox
process at the end of every run; the report lists their paths.
"""

from __future__ import annotations


def _fmt(v) -> str:
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _call(c: dict) -> str:
    args = ", ".join(repr(v) for v in c.get("args", {}).values())
    return f"{c['primitive']}({args})"


def build_failure_report(result: dict, check: dict) -> str:
    """Plain-text report of why an attempt failed (or a one-liner if it succeeded).

    result: sandbox.run_code output. check: verify.checker.check output.
    """
    if check["success"]:
        return "Summary: success; every goal holds and no unmoved block was disturbed."
    final = result.get("final_state") or {}
    lines = []

    failed_goals = [g for g in check["per_goal"] if not g["ok"]]
    disturbed = [u for u in check.get("unmoved", []) if not u["ok"]]
    first_bad = next((c for c in result.get("call_log", []) if not c.get("ok")), None)

    if result.get("sandbox_error"):
        why = "the sandbox harness failed; this is not caused by the generated code"
    elif result.get("rejected"):
        why = "code was rejected before running"
    elif result.get("timed_out"):
        why = "run timed out"
    elif result.get("exception"):
        why = "run raised an exception"
    elif first_bad:
        why = f"{first_bad['primitive']} failed"
    else:
        why = "the code ran but the goal was not reached"
    lines.append(
        f"Summary: FAILED ({why}); {len(failed_goals)}/{len(check['per_goal'])} goals not met"
        + (f"; {len(disturbed)} block(s) that must stay put were moved" if disturbed else "")
        + "."
    )

    if result.get("exception"):
        lines.append(f"Exception: {result['exception']}")
    calls = result.get("call_log", [])
    if first_bad:
        idx = calls.index(first_bad) + 1
        lines.append(f"First failed primitive: call #{idx} {_call(first_bad)} -> {first_bad.get('reason', '')}")
    elif calls:
        lines.append(f"All {len(calls)} primitive calls returned ok: " + ", ".join(_call(c) for c in calls))
    else:
        lines.append("No primitive was called.")

    if failed_goals:
        lines.append("Goals not met:")
        for g in failed_goals:
            where = ""
            if g["subject"] in final and g["target"] in final:
                where = (f" ({g['subject']} at {_fmt(final[g['subject']]['position'])}, "
                         f"{g['target']} at {_fmt(final[g['target']]['position'])})")
            dist = f"{g['error_cm']} cm from goal" if g["error_cm"] is not None else "not evaluated"
            vec = g.get("error_vec_cm")
            if vec is not None:
                dist += (f", off by dx {vec[0]:+.1f} cm, dy {vec[1]:+.1f} cm "
                         "(world frame: +x away from the robot, +y to its left)")
            lines.append(f"  - {g['subject']} {g['relation']} {g['target']}: {dist}{where}; {g['detail']}")
    if disturbed:
        lines.append("Blocks that no step moves but that moved (must stay within 2 cm):")
        for u in disturbed:
            lines.append(f"  - {u['block']} moved {u['moved_cm']} cm")

    frames = result.get("frames") or {}
    if frames:
        lines.append("Final frames: " + ", ".join(f"{cam}: {path}" for cam, path in frames.items()))
    return "\n".join(lines)
