"""FAKE GENERATOR: harness test only, not a result.

A scripted stand-in for the code generator, used to exercise the evaluation harness
offline. Its behavior is fixed by construction:

  t1-t3  correct code on every attempt
  t4     wrong stacking order on attempt 1; correct once feedback is given
  t5     lines blocks up right-to-left on attempt 1; correct once feedback is given
  t6     moves the wrong block on attempt 1; correct once feedback is given
  t7     raises an exception on every attempt (never fixed)
  t8     always moves the green block, ignoring the mug (never fixed)

So condition A can only pass t1-t3 (plus t8 layouts where green happens to be nearest),
while B and C also pass t4-t6.
"""

from __future__ import annotations

RUN_SOURCE = "fake"

# Inline helpers for generated code (the sandbox allows no imports).
_SPOT_HELPERS = '''
RADIUS = {"block": 0.03, "bin": 0.09, "mug": 0.055}


def clearance(x, y, scene, ignore=(), avoid=()):
    c = 1.0
    for name, v in scene.items():
        if name in ignore:
            continue
        px, py = v["position"][0], v["position"][1]
        c = min(c, ((x - px) ** 2 + (y - py) ** 2) ** 0.5 - RADIUS[v["kind"]])
    for ax, ay in avoid:
        c = min(c, ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 - 0.05)
    return c


def grid(step=0.02):
    xs = [0.36 + step * i for i in range(int((0.68 - 0.36) / step) + 1)]
    ys = [-0.32 + step * j for j in range(int(0.64 / step) + 1)]
    return [(x, y) for x in xs for y in ys]


def free_spot(scene, ignore=(), avoid=()):
    return max(grid(), key=lambda p: (round(clearance(p[0], p[1], scene, ignore, avoid), 4), -abs(p[0] - 0.5)))


def row_spots(scene, order, spacing=0.10):
    best = None
    for x, cy in grid(0.01):
        spots = [(x, cy + spacing * (1 - i)) for i in range(len(order))]
        if min(s[1] for s in spots) < -0.32 or max(s[1] for s in spots) > 0.32:
            continue
        ok = True
        for i, (sx, sy) in enumerate(spots):
            for later in order[i + 1:]:
                lx, ly = scene[later]["position"][0], scene[later]["position"][1]
                if ((sx - lx) ** 2 + (sy - ly) ** 2) ** 0.5 < 0.07:
                    ok = False
        key = ((x - 0.5) ** 2 + cy ** 2) ** 0.5
        if ok and (best is None or key < best[0]):
            best = (key, spots)
    return best[1]
'''


def _moves(pairs: list[tuple[str, str]]) -> str:
    body = "".join(
        f'    r = robot.pick("{o}")\n    if not r["ok"]:\n        raise RuntimeError(r["reason"])\n'
        f'    robot.place("{o}", {d})\n'
        for o, d in pairs
    )
    return "def run(robot, scene):\n" + body


CORRECT = {
    "t1_red_in_red_bin": _moves([("red_block", '"red_bin"')]),
    "t2_sort_by_color": _moves([("red_block", '"red_bin"'), ("blue_block", '"blue_bin"')]),
    "t3_stack_green_on_red": _moves([("green_block", '"red_block"')]),
    "t4_stack_three": _moves([("green_block", '"red_block"'), ("blue_block", '"green_block"')]),
    "t5_line_up": _SPOT_HELPERS + '''

def run(robot, scene):
    order = ["red_block", "green_block", "blue_block"]
    for name, spot in zip(order, row_spots(scene, order)):
        robot.pick(name)
        robot.place(name, spot)
''',
    "t6_move_named_only": _moves([("green_block", '"blue_bin"')]),
    "t7_swap_red_blue": _SPOT_HELPERS + '''

def run(robot, scene):
    r0 = tuple(scene["red_block"]["position"][:2])
    b0 = tuple(scene["blue_block"]["position"][:2])
    robot.pick("red_block")
    robot.place("red_block", free_spot(scene, avoid=(r0, b0)))
    robot.pick("blue_block")
    robot.place("blue_block", r0)
    robot.pick("red_block")
    robot.place("red_block", b0)
''',
}

WRONG = {
    "t4_stack_three": _moves([("blue_block", '"red_block"'), ("green_block", '"blue_block"')]),
    "t5_line_up": _SPOT_HELPERS + '''

def run(robot, scene):
    order = ["red_block", "green_block", "blue_block"]
    spots = row_spots(scene, order)
    for name, spot in zip(order, reversed(spots)):
        robot.pick(name)
        robot.place(name, spot)
''',
    "t6_move_named_only": _moves([("red_block", '"blue_bin"')]),
    "t7_swap_red_blue": 'def run(robot, scene):\n    raise RuntimeError("fake generator: swap not implemented")\n',
    "t8_nearest_to_mug": _moves([("green_block", '"red_bin"')]),
}

FIXED_BY_FEEDBACK = {"t4_stack_three", "t5_line_up", "t6_move_named_only"}


class FakeGenerator:
    """generate_fn for run_with_repair. Deterministic; reports zero tokens."""

    name = "fake"
    run_source = RUN_SOURCE

    def __call__(self, *, task, plan, attempt, scene=None, feedback=None, image_path=None) -> dict:
        tid = task.id
        if tid in CORRECT and tid not in WRONG:
            code = CORRECT[tid]
        elif tid in FIXED_BY_FEEDBACK and attempt >= 2 and feedback is not None:
            code = CORRECT[tid]
        else:
            code = WRONG[tid]
        return {"code": code, "tokens_in": 0, "tokens_out": 0}
