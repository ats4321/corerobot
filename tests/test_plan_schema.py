"""Plan schema validation and relation semantics."""

import copy

import pytest
from pydantic import ValidationError

from core_robot.planning.schema import RELATIONS, Plan
from core_robot.verify.relations import check_success, holds

VALID = {
    "objects": [
        {"name": "red_block", "color": "red", "kind": "block"},
        {"name": "green_block", "color": "green", "kind": "block"},
        {"name": "red_bin", "color": "red", "kind": "bin"},
    ],
    "steps": [
        {"index": 0, "action": "pick_and_place", "object": "red_block", "destination": "red_bin"},
        {"index": 1, "action": "pick_and_place", "object": "green_block", "destination": "free_table_space"},
    ],
    "goal": [{"subject": "red_block", "relation": "in", "target": "red_bin"}],
}


def mutate(fn):
    d = copy.deepcopy(VALID)
    fn(d)
    return d


def test_valid_plan_parses():
    plan = Plan.model_validate(VALID)
    assert plan.named_in_steps() == {"red_block", "red_bin", "green_block"}


def test_relations_list():
    assert set(RELATIONS) == {"in", "on", "left_of", "right_of", "next_to", "swapped_with"}


@pytest.mark.parametrize(
    "bad",
    [
        mutate(lambda d: d["objects"].append({"name": "purple_block", "color": "purple", "kind": "block"})),
        mutate(lambda d: d["objects"].__setitem__(0, {"name": "red_block", "color": "blue", "kind": "block"})),
        mutate(lambda d: d["steps"][0].update(object="Red Block")),
        mutate(lambda d: d["steps"][0].update(destination=[0.5, 0.1])),
        mutate(lambda d: d["steps"][0].update(destination="0.5, 0.1")),
        mutate(lambda d: d["steps"][0].update(destination={"x": 0.5, "y": 0.1})),
        mutate(lambda d: d["steps"][0].update(action="push")),
        mutate(lambda d: d["steps"][0].update(object="red_bin")),
        mutate(lambda d: d["steps"][1].update(index=5)),
        mutate(lambda d: d["steps"][0].update(destination="blue_bin")),  # real name, not in objects
        mutate(lambda d: d["goal"][0].update(relation="inside")),
        mutate(lambda d: d["goal"][0].update(relation="above")),
        mutate(lambda d: d["goal"][0].update(target="table")),
        mutate(lambda d: d.update(goal=[])),
        mutate(lambda d: d.update(extra="field")),
    ],
)
def test_invalid_plans_rejected(bad):
    with pytest.raises(ValidationError):
        Plan.model_validate(bad)


def test_check_scene():
    plan = Plan.model_validate(VALID)
    plan.check_scene(["red_block", "green_block", "red_bin", "blue_bin"])
    with pytest.raises(ValueError):
        plan.check_scene(["red_block", "red_bin"])


def st(**pos):
    kinds = {"red_block": "block", "green_block": "block", "blue_block": "block", "red_bin": "bin", "mug": "mug"}
    return {n: {"position": list(p), "orientation": [0, 0, 0, 1], "color": n.split("_")[0], "kind": kinds[n]}
            for n, p in pos.items()}


def atom(s, r, t):
    from core_robot.planning.schema import GoalAtom
    return GoalAtom(subject=s, relation=r, target=t)


def test_relation_semantics():
    s = st(red_block=(0.5, 0.0, 0.02), green_block=(0.5, 0.0, 0.06), blue_block=(0.5, -0.06, 0.02),
           red_bin=(0.5, 0.25, 0.0), mug=(0.4, 0.25, 0.045))
    assert holds(atom("green_block", "on", "red_block"), s, s)[0]
    assert not holds(atom("red_block", "on", "green_block"), s, s)[0]
    assert holds(atom("red_block", "left_of", "blue_block"), s, s)[0]  # +y is left
    assert holds(atom("blue_block", "right_of", "red_block"), s, s)[0]
    assert not holds(atom("blue_block", "left_of", "red_block"), s, s)[0]
    assert holds(atom("blue_block", "next_to", "red_block"), s, s)[0]
    assert not holds(atom("green_block", "next_to", "red_block"), s, s)[0]  # stacked, not next to
    assert not holds(atom("red_block", "in", "red_bin"), s, s)[0]
    inside = st(red_block=(0.52, 0.27, 0.026), red_bin=(0.5, 0.25, 0.0))
    assert holds(atom("red_block", "in", "red_bin"), inside, inside)[0]


def test_swapped_with_uses_initial_state():
    init = st(red_block=(0.4, 0.1, 0.02), blue_block=(0.6, -0.1, 0.02))
    swapped = st(red_block=(0.61, -0.1, 0.02), blue_block=(0.4, 0.12, 0.02))
    assert holds(atom("red_block", "swapped_with", "blue_block"), swapped, init)[0]
    assert not holds(atom("red_block", "swapped_with", "blue_block"), init, init)[0]
    half = st(red_block=(0.6, -0.1, 0.02), blue_block=(0.6, -0.04, 0.02))
    assert not holds(atom("red_block", "swapped_with", "blue_block"), half, init)[0]


def test_global_rule_unnamed_blocks_must_stay():
    plan = Plan.model_validate({
        "objects": [{"name": n, "color": n.split("_")[0], "kind": "bin" if "bin" in n else "block"}
                    for n in ("red_block", "blue_block", "red_bin")],
        "steps": [{"index": 0, "action": "pick_and_place", "object": "red_block", "destination": "red_bin"}],
        "goal": [{"subject": "red_block", "relation": "in", "target": "red_bin"}],
    })
    init = st(red_block=(0.4, 0.1, 0.02), blue_block=(0.6, -0.1, 0.02), red_bin=(0.5, 0.25, 0.0))
    done = st(red_block=(0.5, 0.25, 0.026), blue_block=(0.605, -0.1, 0.02), red_bin=(0.5, 0.25, 0.0))
    assert check_success(plan, init, done)["success"]
    bumped = st(red_block=(0.5, 0.25, 0.026), blue_block=(0.63, -0.1, 0.02), red_bin=(0.5, 0.25, 0.0))
    r = check_success(plan, init, bumped)
    assert not r["success"] and all(g["ok"] for g in r["goal"])
