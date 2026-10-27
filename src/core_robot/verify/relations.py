"""Goal relations and the success check, judged only from simulator state.

Frames: meters, table top at z = 0, +x away from the robot, +y to the robot's left
(the left side of the overhead image).

Relations (subject relation target):
  in            subject's center is inside the target bin's inner footprint and below its rim
  on            subject rests on top of target: centers within 2 cm horizontally and the
                subject's bottom within 1.5 cm of the target's top
  left_of       subject is at least 3 cm further left (+y) than target, and the two are
                within 15 cm of each other front-to-back (x)
  right_of      the mirror of left_of (at least 3 cm further -y)
  next_to       centers within 9 cm horizontally, and neither is on top of the other
  swapped_with  subject ends within 3 cm (horizontally) of where target started, and
                target ends within 3 cm of where subject started
"""

from __future__ import annotations

import math

from core_robot.planning.schema import GoalAtom, Plan
from core_robot.sim.env import BIN_INNER, BIN_WALL_H, BLOCK_SIZE, MUG_HEIGHT, OBJECTS

HEIGHT = {"block": BLOCK_SIZE, "mug": MUG_HEIGHT}
KIND = {o.name: o.kind for o in OBJECTS}

ON_XY_TOL = 0.02
ON_Z_TOL = 0.015
LR_MIN_DY = 0.03
LR_MAX_DX = 0.15
NEXT_TO_MAX = 0.09
SWAP_TOL = 0.03
STILL_TOL = 0.02


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_on(s, t, s_name, t_name) -> tuple[bool, str]:
    if KIND[t_name] == "bin":
        return False, f"{t_name} is a bin; use 'in'"
    t_top = t[2] + HEIGHT[KIND[t_name]] / 2
    s_bottom = s[2] - HEIGHT[KIND[s_name]] / 2
    dxy, dz = _xy_dist(s, t), s_bottom - t_top
    ok = dxy <= ON_XY_TOL and abs(dz) <= ON_Z_TOL
    return ok, f"horizontal offset {dxy:.3f} m (max {ON_XY_TOL}), bottom-to-top gap {dz:+.3f} m (max ±{ON_Z_TOL})"


def holds(atom: GoalAtom, final: dict, initial: dict) -> tuple[bool, str]:
    """Evaluate one goal atom. Returns (holds, human-readable measurement)."""
    sn, tn, rel = atom.subject, atom.target, atom.relation
    if sn not in final or tn not in final:
        return False, f"object missing from scene: {sn if sn not in final else tn}"
    s, t = final[sn]["position"], final[tn]["position"]
    if rel == "in":
        if KIND[tn] != "bin":
            return False, f"{tn} is not a bin"
        dx, dy = abs(s[0] - t[0]), abs(s[1] - t[1])
        half = BIN_INNER / 2
        ok = dx <= half and dy <= half and s[2] < t[2] + BIN_WALL_H
        return ok, f"offset from bin center ({dx:.3f}, {dy:.3f}) m (max {half}), height {s[2]:.3f} m (rim {BIN_WALL_H})"
    if rel == "on":
        return _is_on(s, t, sn, tn)
    if rel in ("left_of", "right_of"):
        dy = (s[1] - t[1]) if rel == "left_of" else (t[1] - s[1])
        dx = abs(s[0] - t[0])
        ok = dy >= LR_MIN_DY and dx <= LR_MAX_DX
        return ok, f"lateral margin {dy:+.3f} m (min {LR_MIN_DY}), front-back offset {dx:.3f} m (max {LR_MAX_DX})"
    if rel == "next_to":
        d = _xy_dist(s, t)
        stacked = "bin" not in (KIND[sn], KIND[tn]) and (_is_on(s, t, sn, tn)[0] or _is_on(t, s, tn, sn)[0])
        return d <= NEXT_TO_MAX and not stacked, f"center distance {d:.3f} m (max {NEXT_TO_MAX}), stacked={stacked}"
    if rel == "swapped_with":
        d1 = _xy_dist(s, initial[tn]["position"])
        d2 = _xy_dist(t, initial[sn]["position"])
        ok = d1 <= SWAP_TOL and d2 <= SWAP_TOL
        return ok, f"{sn} is {d1:.3f} m from {tn}'s start, {tn} is {d2:.3f} m from {sn}'s start (max {SWAP_TOL})"
    return False, f"unknown relation {rel!r}"


def unnamed_blocks_still(plan: Plan, initial: dict, final: dict) -> list[tuple[str, bool, str]]:
    """GLOBAL SUCCESS RULE: blocks not named in the plan's steps must move less than 2 cm.

    A block is "named in the plan's steps" if it is the object or the destination of
    any step. Every other block must end within 2 cm (horizontal distance) of where it
    started; knocking over or relocating a block the plan does not mention fails the task.
    """
    named = plan.named_in_steps()
    out = []
    for name, s0 in initial.items():
        if s0["kind"] != "block" or name in named:
            continue
        d = _xy_dist(final[name]["position"], s0["position"])
        out.append((name, d < STILL_TOL, f"{name} moved {d:.3f} m (max {STILL_TOL})"))
    return out


def check_success(plan: Plan, initial: dict, final: dict) -> dict:
    """Success = every goal atom holds AND the global rule (unnamed_blocks_still) holds.

    Nothing else is tested. Returns {"success", "goal": [...], "global_rule": [...]}.
    """
    goal = []
    for atom in plan.goal:
        ok, why = holds(atom, final, initial)
        goal.append({"atom": f"{atom.subject} {atom.relation} {atom.target}", "ok": ok, "detail": why})
    still = [{"block": n, "ok": ok, "detail": why} for n, ok, why in unnamed_blocks_still(plan, initial, final)]
    return {
        "success": all(g["ok"] for g in goal) and all(s["ok"] for s in still),
        "goal": goal,
        "global_rule": still,
    }
