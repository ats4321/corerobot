# Core Robot

Learns a tabletop task from one demonstration video and performs it with a simulated Franka Panda arm in PyBullet, using Meta's SAM 3.1 and Muse Spark models.

See [PROJECT.md](PROJECT.md) for the full brief.

## Setup

Requires a native Python >= 3.10 (developed on 3.12, arm64).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
echo "MODEL_API_KEY=..." > .env   # gitignored; loaded with python-dotenv
```

On macOS with a recent SDK, `pip install pybullet` fails compiling its bundled zlib
(`fdopen` macro clash). Workaround: download the sdist, delete the line
`#define fdopen(fd, mode) NULL` near line 128 of `examples/ThirdPartyLibs/zlib/zutil.h`,
and `pip wheel` it locally.

## Checks

```bash
.venv/bin/pytest -q                          # offline unit tests
.venv/bin/python scripts/smoke_models.py     # live API smoke test (cached in data/cache/model)
CORE_ROBOT_REPLAY=1 .venv/bin/python scripts/smoke_models.py   # cache-only replay
.venv/bin/python scripts/handwritten_demo.py # sim baseline: red_block -> red_bin, 10 seeds + mp4 in data/runs/
.venv/bin/python scripts/make_demos.py        # 10 scripted demos -> data/demos/<task>/seed_<n>/
```

## Sample data

All files in `data/samples/` (`table_objects.png`, `red_block_move.mp4`, `red_block_move_3s.mp4`) are synthetic placeholders drawn by `scripts/make_placeholder_samples.py`, not real footage.
