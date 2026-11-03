# Findings v1

Date: 2026-10-28. Prompt hash `6faa3b19e9b6` (tag `prompts-frozen-v1`). All results come from the
simulator state (never from asking a model). Source logs: `data/runs/eval/muse_predicted_v2`,
`data/runs/planner/results.json`, `data/runs/perception/sam_gate.json` (not committed).

## Setup

- 8 tasks x execution seeds 100-104. Plans are **predicted** from each task's seed-0 demo video
  (Muse Spark on the angled-camera video plus SAM 3.1 tracks), not ground truth.
- Conditions: A one shot; B up to 3 attempts with a text failure report; C as B plus the final
  angled-camera frame. Attempt 1 is the same request in A, B and C (and in both environments), so
  B and C can only differ from A after a failed first attempt.
- Environments: clean, and perturbed (`calibration_offset`: every `place` releases 4 cm off in a
  direction seeded per run; `place` still reports success at the commanded point).

## Perception and planning

- SAM gate (every block and bin found and tracked; first/last boxes contain the true start/end):
  **10/10 demos**, after replacing the concept "red bin" / "blue bin" (0 instances in 4 demos) with
  phrases chosen by an image-only plausibility probe ("red box", "blue box"; "white cylinder" for
  the mug).
- Planner, strict match against the ground-truth plan: **9/10 demos**; lenient (goal atoms only):
  **9/10**. The failure is t5_line_up seed 0: the plan adds a step "blue_block -> red_block". In the
  demo the red block is set down just in front of the blue one, which from the angled camera reads
  as a stack, and occlusion makes the blue block's box shift early, so the track summary says it
  moved before it did.
- The t8 plan ("put the block nearest the mug in the bin") names `blue_block` with **no note**
  about the rule, even after a generic prompt instruction asking for such notes. Scoring ignores
  notes, so this plan counts as a strict match.

## Clean environment (40 runs per condition)

| | A | B | C |
|---|---|---|---|
| t1-t7 | 35/35 | 35/35 | 35/35 |
| t8 | 1/5 | 5/5 | 5/5 |
| overall | **36/40** | **40/40** | **40/40** |

- Repair fixed 4 runs in B and 4 in C, all t8, and broke none.
- How t8 was fixed: the failure report lists the unmet goal, and the goal comes from the task's
  ground truth for that layout ("green_block in red_bin"). The report therefore names the block
  that should have moved, which carries the selection rule the plan lacked. The repaired code
  then chose the block nearest the mug. This is repair working from information in the report,
  not evidence that the planner understood the demonstration.

## Perturbed environment (calibration_offset 4 cm)

- **A: 21/40** (t1, t2, t5, t6 5/5; t3, t4, t7 0/5; t8 1/5). 40/40 attempt-1 requests were cache
  hits shared with the clean runs (no new generations).
- **B and C: incomplete.** The budget stop left 42 of 80 runs not run (t4 one seed, t5-t8). They
  were not run afterwards.
- Of the runs that exist, **0 of 9 repairable pairs were fixed** in B and 0 of 9 in C (t3 x5,
  t4 x4); repair broke nothing. Excluding runs whose repair output was truncated: 0 of 3 (B) and
  0 of 6 (C).
- Cause: no primitive can aim a stack off-centre. `place(obj, "red_block")` always aims at the
  block's centre and `place(obj, (x, y))` sets the object on the table, so code cannot compensate
  for a release offset when stacking. In the logs the model repeats the same place, or returns to
  its first code.
- Failure labels (last attempt): A placement error 15, wrong object 4; B placement error 8,
  truncated output 1; C placement error 8, wrong object 1. "Placement error": every primitive
  returned ok but the goal was not met.

## Generation limits

- **9 of 36** repair generations in the perturbed B/C runs hit the 12,000-token output cap;
  **7** of them returned no code (rejected before running). Runs with a truncated repair attempt:
  B 6, C 3.
- Condition C lost its image in **3 runs** (all t4): after a rejected attempt there is no new
  simulator frame, so the next attempt went without one and became the same request as B.

## Cost

| Call | Measured average |
|---|---|
| Planner (video + tracks) | $0.0178 (4,885 in / 2,752 out tokens) |
| Code generation, attempt 1 | $0.0128 (1,743 in / 2,494 out) |
| Repair generation | $0.030 (avg 6,395 out tokens; up to the 12,000 cap) |
| SAM 3.1 | $0.0002 per video frame (min 50 per request); $0.0025 per image |

Total API spend to 2026-10-28: **$3.73** (from the call log; cache hits are free). The project total,
including the later synthetic-clip runs, is in the README.

## Budget overshoot and fix

The perturbed B/C session was given $1.00 and spent **$1.164**. The budget check ran only before a
run started, so runs already in flight finished past the limit, and the estimate assumed $0.0129
per repair generation while repairs averaged about $0.030. Fix: the budget guard now reserves each
call's worst-case cost (input bound plus the full output cap) before sending, counts all
outstanding reservations across threads, and refuses any call that could exceed the budget; the
reservation is replaced by the actual cost when the response arrives.
