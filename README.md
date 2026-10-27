# Core Robot

Learns a tabletop task from one demonstration video and performs it with a simulated Franka Panda arm in PyBullet, using Meta's SAM 3.1 and Muse Spark models.

See [PROJECT.md](PROJECT.md) for the full brief.

Status: simulator, primitives, 8 tasks with scripted demos, sandbox, checker, repair loop, evaluation harness and an offline UI are built and tested offline. The live model stages (SAM perception, Muse Spark planning and code generation) are not wired up yet.

## Setup

`requirements.txt` pins exact versions from `pip freeze` on **macOS arm64 with Python 3.12**. Other platforms or Python versions may need different versions.

```bash
python3.12 -m venv .venv                     # a native (arm64) Python >= 3.10
.venv/bin/pip install --find-links <dir-with-pybullet-wheel> -r requirements.txt   # see pybullet note
.venv/bin/pip install -e .
echo "MODEL_API_KEY=..." > .env              # gitignored; loaded with python-dotenv
```

### pybullet on macOS

pybullet has no macOS wheels, and on recent macOS SDKs building the sdist fails in its bundled zlib (`fdopen` macro clash). Build it once locally:

```bash
.venv/bin/pip download --no-deps --no-binary :all: pybullet==3.2.7 -d /tmp/pyb
tar xzf /tmp/pyb/pybullet-3.2.7.tar.gz -C /tmp/pyb
# remove the `#define fdopen(fd, mode) NULL` lines (the macOS one, line 128, is the one that breaks):
sed -i '' '/#define fdopen(fd, mode) NULL/d' /tmp/pyb/pybullet-3.2.7/examples/ThirdPartyLibs/zlib/zutil.h
.venv/bin/pip wheel --no-deps -w /tmp/pyb/wheels /tmp/pyb/pybullet-3.2.7   # a few minutes
```

Then pass `--find-links /tmp/pyb/wheels` to the install above.

## Commands

```bash
# tests (offline, ~40 s)
.venv/bin/pytest -q

# simulator baseline: red block into the red bin for 10 seeds, plus an mp4 in data/runs/
.venv/bin/python scripts/handwritten_demo.py

# regenerate the 10 scripted demos -> data/demos/<task>/seed_<n>/ (video, plan, states, result)
.venv/bin/python scripts/make_demos.py

# evaluation harness with the FAKE generator (harness test only) -> data/runs/fake/
.venv/bin/python -m core_robot.eval.run_eval --generator fake --dry-run   # planned runs + cost bound
.venv/bin/python -m core_robot.eval.run_eval --generator fake             # 120 runs, ~4 min, resumable

# offline UI (reads saved demos and run logs) -> http://127.0.0.1:7860
.venv/bin/python -m core_robot.ui.app

# primitives reference used in the code-generation prompt
.venv/bin/python -m core_robot.codegen.api_doc

# live API smoke test (needs .env); CORE_ROBOT_REPLAY=1 serves only from the cache
.venv/bin/python scripts/smoke_models.py
```

## Results

No live results yet. This section is generated only from live runs (`run_source=live`) by `core_robot.eval.report.build_readme_results`, which refuses fake-generator output.

## Sample data

All files in `data/samples/` (`table_objects.png`, `red_block_move.mp4`, `red_block_move_3s.mp4`) are synthetic placeholders drawn by `scripts/make_placeholder_samples.py`, not real footage.
