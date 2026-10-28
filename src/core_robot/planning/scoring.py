"""Plan scoring against ground truth. Both definitions were fixed before any results.

Everything is compared through Plan.scoring_view(), so step notes never affect a score.

strict (the gate): object name sets equal; steps equal as an ordered list of
    (action, object, destination); goal atoms equal as a set of (subject, relation, target).
lenient (reported beside it): goal atoms equal as a set; objects and steps ignored.
"""

from __future__ import annotations

from core_robot.planning.schema import Plan

ERROR_TYPES = ("invalid plan", "wrong objects", "wrong step count", "wrong step object", "wrong destination",
               "wrong order", "wrong goal relation", "missing goal atom", "extra goal atom")


def _parts(plan: Plan):
    v = plan.scoring_view()
    objects = {o["name"] for o in v["objects"]}
    steps = [(s["action"], s["object"], s["destination"]) for s in v["steps"]]
    goal = {(g["subject"], g["relation"], g["target"]) for g in v["goal"]}
    return objects, steps, goal


def strict_match(pred: Plan, gt: Plan) -> bool:
    return _parts(pred) == _parts(gt)


def lenient_match(pred: Plan, gt: Plan) -> bool:
    return _parts(pred)[2] == _parts(gt)[2]


def plan_errors(pred: Plan | None, gt: Plan) -> list[str]:
    """Every difference type found (empty when strict_match holds)."""
    if pred is None:
        return ["invalid plan"]
    po, ps, pg = _parts(pred)
    go, gs, gg = _parts(gt)
    errors = []
    if po != go:
        errors.append("wrong objects")
    if len(ps) != len(gs):
        errors.append("wrong step count")
    elif ps != gs:
        if sorted(ps) == sorted(gs):
            errors.append("wrong order")
        else:
            if any(p[1] != g[1] for p, g in zip(ps, gs)):
                errors.append("wrong step object")
            if any(p[1] == g[1] and p[2] != g[2] for p, g in zip(ps, gs)):
                errors.append("wrong destination")
    if pg != gg:
        # Atoms compared one by one (a pair can carry several relations). A missing atom whose
        # (subject, target) pair appears among the extra atoms is a wrong relation.
        missing, extra = gg - pg, pg - gg
        same_pair = lambda a, atoms: any(b[0] == a[0] and b[2] == a[2] for b in atoms)
        if any(same_pair(a, extra) for a in missing):
            errors.append("wrong goal relation")
        if any(not same_pair(a, extra) for a in missing):
            errors.append("missing goal atom")
        if any(not same_pair(a, missing) for a in extra):
            errors.append("extra goal atom")
    return errors


def score(pred: Plan | None, gt: Plan) -> dict:
    return {
        "strict": pred is not None and strict_match(pred, gt),
        "lenient": pred is not None and lenient_match(pred, gt),
        "errors": plan_errors(pred, gt),
    }
