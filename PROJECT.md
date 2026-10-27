# Core Robot — Project Brief

Core Robot learns a tabletop task from one demonstration video and then performs it with a simulated robot arm.

## Pipeline

demo video -> SAM 3.1 segments and tracks objects -> Muse Spark turns video + masks into a structured plan (JSON) -> Muse Spark writes Python that calls robot primitives -> PyBullet simulator (Franka Panda) runs it -> a checker tests final object poses against the goal -> on failure, a failure report and a rendered frame go back to the code generator, up to 3 attempts.

## Project goal

Show that Meta's models can reason about physical tasks, measured as task success rate with and without the repair loop over 8 tasks x 5 seeded layouts x 3 conditions (A one-shot, B repair with text report, C repair with text report plus rendered frame).

## Hard rules

- Simulation only.
- The model never controls joints; it only calls primitives pick(obj), place(obj, target), move_above(target), open_gripper().
- Generated code defines run(robot, scene). scene is read-only data (a copy of the starting scene state: name, position, orientation, color, kind), not a primitive; changing it does not affect the simulator.
- Success is judged from simulator state, never by asking a model.
- Every model response is cached to disk by request hash.
- All randomness is seeded.
- Generated code runs only in a sandboxed subprocess with a timeout.

## Models

Via Meta Model API (OpenAI-compatible, base URL https://api.meta.ai/v1, key in env var MODEL_API_KEY):

- muse-spark-1.3 for planning and code
- sam-3.1 for segmentation

## Deadline

Submission deadline: Nov 4.
