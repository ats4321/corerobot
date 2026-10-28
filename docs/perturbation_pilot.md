# Perturbation pilot: choosing the calibration_offset magnitude

**Chosen and frozen: m = 4 cm** (`CALIBRATION_OFFSET_CM = 4` in `src/core_robot/config.py`),
fixed before any perturbed run on the evaluation seeds 100-104.

## What the perturbation is

`calibration_offset` (opt-in, off by default) shifts the gripper's release position by a
constant offset in every `place` of a run: magnitude m cm, direction drawn from a random
stream seeded by the execution seed. The robot does not know: `place` returns `ok=True` and
reports the commanded point as the object's position. The checker, failure report, logs and
videos use the true simulator state. The code-generation prompt and API reference never
mention it (tested). See `src/core_robot/sim/perturbation.py`.

## Setup

- Condition A (one shot), tasks t1-t7, one run per task and seed.
- **Ground-truth plans.** The perturbation affects execution only, so the pilot uses
  ground-truth plans to measure its effect on execution without mixing in planner errors.
- **Pilot seeds 200-204.** They are never used for evaluation (evaluation uses 100-104;
  tested to be disjoint, including the derived layout seeds).
- The code for each (task, seed) is generated once and reused for every m (the prompt does
  not depend on the perturbation), so the columns differ only in the offset.
- Rule fixed before the pilot: pick the smallest m in {2, 3, 4, 5} where condition A succeeds
  on roughly 40-60% of runs; if none does, stop and report.

## Results (condition A successes out of 5 seeds)

| Task | clean | m=2 | m=3 | m=4 | m=5 |
|---|---|---|---|---|---|
| t1_red_in_red_bin | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| t2_sort_by_color | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| t3_stack_green_on_red | 5/5 | 3/5 | 0/5 | 0/5 | 0/5 |
| t4_stack_three | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| t5_line_up | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| t6_move_named_only | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| t7_swap_red_blue | 5/5 | 5/5 | 2/5 | 0/5 | 0/5 |
| **total** | 35/35 (100%) | 28/35 (80%) | 22/35 (63%) | **20/35 (57%)** | 19/35 (54%) |

m = 2 (80%) and m = 3 (63%) are above the range; m = 4 (57%) is the smallest in it.

## Caveat

The effect is strongly task dependent. Placing into a 14 cm bin (t1, t2, t6) and lining up
(t5, judged on left/right order) tolerate offsets up to 5 cm, while stacking (2 cm tolerance)
and swapping (3 cm tolerance) fail almost always from m = 3. The 57% total is an average of
tasks that are nearly unaffected and tasks that nearly always fail, not a uniform difficulty.
Failed perturbed runs are labelled "other" by the failure categories (the code ran and every
primitive reported ok, but objects ended off target), plus one "wrong object" at m = 4 and 5
(a block that should not move was disturbed).

Pilot run logs: `data/runs/eval/pilot_clean`, `pilot_m2` ... `pilot_m5` (not committed).
