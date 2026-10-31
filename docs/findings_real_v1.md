# Findings: real-clip pipeline v1 (one synthetic clip)

Date: 2026-10-31. Prompt hash `6faa3b19e9b6` (tag `prompts-frozen-v1`) in every run below; no
prompt was changed. Clip: `data/real/synthetic_sort_red_blue.mp4`, an **AI-generated** clip (not
real footage) in which a hand puts a red cube into a red box, then a blue cube into a blue box. It
contains generation artifacts (a cube split into pieces by a ghosted hand, a phantom blue cube).
Source logs (not committed): `data/runs/real/synthetic_sort_red_blue/` (first run),
`data/runs/real/footage_real_v1/` (real-footage mode), `data/runs/real/ablation_tracks_v1/`.

**This is one synthetic clip. It is not a success rate and says nothing yet about real footage.**

## Result on the clip

Two kinds of check, kept separate:

- `plan_matches_clip`: the generated plan has the same steps (object, destination, order) and the
  same goal atoms as the hand-labelled `data/real/synthetic_sort_red_blue.expected_plan.json`
  (red_block -> red_bin, then blue_block -> blue_bin).
- Simulator success: the robot carried out the plan's own goals in a simulator scene built from the
  plan (seed 300). It does not check the plan against the clip.

| Run | Perception | plan_matches_clip | Simulator | Spend |
|---|---|---|---|---|
| First run | simulator mode | True | success, attempt 1 | $0.113 |
| Real-footage mode | real mode | True | success, attempt 1 | $0.012 (planner only; rest cached) |

Both runs produced the same plan. In the first run that plan came from wrong tracks (below).

## Real-footage perception mode (`--footage real`)

The simulator perception path assumed the simulator camera. On this clip it failed on 2 of 4
objects: the red box was rejected as "off the table" / wrong bin size, and `red_block` was given the
red **box**, because "red cube" returned the box, the cube and a cube fragment, and the selection
keeps only ids seen on at least half as many frames as the longest track.

Real mode (`src/core_robot/perception/real_mode.py`, default for `run_real_demo.py`; the simulator
path is unchanged and byte-identical on the 10 demos):

- Plausibility uses only image-relative measures: box area as a fraction of the frame, aspect
  ratio, and mask colour.
- All instances from all SAM requests for a colour are pooled. Bins (and the mug) take the largest
  plausible instance, blocks the smallest; an instance goes to one name only. Never by track length.
- Fragments of one SAM request are merged when they overlap in space, are at most 3 frames apart
  (or overlap in time) and have comparable sizes; each merge is reported.

Chosen tracks on the clip:

| Object | Before (simulator mode) | Real mode |
|---|---|---|
| red_bin | not found | red box ("red cube" id 0), frames 0-53 |
| red_block | red box ("red cube" id 0) | red cube, ids 1+2 merged (ghost fragment), frames 0-15 |
| blue_bin | "blue bin" id 0 | same |
| blue_block | "blue cube" id 0, frames 0-36 | same |

## Limitation: the phantom blue cube

`blue_block`'s track ends at frame 36 (7.2 s) and the summary says it "did not move". By about
6.0 s a second, phantom blue cube (a generation artifact) is already in the blue box, and SAM never
returns it as a separate instance, nor does the tracked cube continue into the box. Nothing in the
tracks shows the blue cube being placed; the plan gets that step from the video. Real mode does not
fix this.

## Track-summary ablation

One planner call per variant, `--plan-only`, seed 300, real-mode perception (cached); the variant
name and the exact track text sent are in each run's `summary.json`.

| Variant | Track text sent to the planner | plan_matches_clip | Spend |
|---|---|---|---|
| (a) tracks | real-mode summary (correct tracks) | True | $0 (cached call from the real-mode run) |
| (b) omit | none: video and object names only | True | $0.014 (live) |
| (c) replay | the first run's wrong summary (red_bin "not found", red_block = the box, "did not move") | True | $0 (the first run's identical cached call) |

All three plans are identical: red_block -> red_bin, then blue_block -> blue_bin, with the same two
goal atoms. On this clip the planner takes the task from the video and the object names; the
track summary, right, wrong or absent, did not change the plan. One clip and one sample per variant
cannot show that the summary never matters, only that it did not matter here. The demo-video
results in `docs/findings_v1.md` (t5_line_up seed 0) include a wrong plan alongside a misleading
track summary; that case was not ablated.
