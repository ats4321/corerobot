# data/real

Both clips here are **AI-generated (synthetic) videos, not real camera footage**:

- `synthetic_sort_red_blue.mp4`: a hand puts a red cube into a red box, then a blue cube into a blue
  box. Contains generation artifacts (a ghosted hand, a phantom blue cube).
- `synthetic_stack_blue_on_red.mp4`: a hand places a blue cube on top of a red cube (close-up).

`<clip>.expected_plan.json` is the hand-labelled plan used for `plan_matches_clip`.

They are committed, with the model responses they replay from (`data/cache/model/`, 12 files), so
the pipeline can be replayed at $0 without an API key:

```bash
CORE_ROBOT_REPLAY=1 .venv/bin/python scripts/run_real_demo.py data/real/synthetic_sort_red_blue.mp4
CORE_ROBOT_REPLAY=1 .venv/bin/python scripts/run_real_demo.py data/real/synthetic_stack_blue_on_red.mp4
```

Results: [docs/findings_real_v1.md](../../docs/findings_real_v1.md). Other clips you add here stay gitignored.
