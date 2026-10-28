"""Prompt freeze: a hash of everything the models are told, and the git commit, for run logs.

prompt_hash() renders the planner prompt, the code-generation prompt and the API reference on
fixed canonical inputs and hashes the result. It changes when the text sent to a model changes
(not when comments or unrelated code change). The value frozen at the git tag in
prompts_frozen.json is what evaluation runs must match unless explicitly allowed otherwise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

FROZEN_FILE = Path(__file__).with_name("prompts_frozen.json")


def _canonical_perception() -> dict:
    frames = {0: {"box": [100, 100, 130, 130], "mask": None}, 9: {"box": [300, 200, 330, 230], "mask": None}}
    bin_frames = {0: {"box": [280, 180, 400, 280], "mask": None}, 9: {"box": [280, 180, 400, 280], "mask": None}}
    return {
        "naming": {"objects": ["red_bin", "red_block"]},
        "clip": {"n_frames": 10, "fps": 5.0, "size": [640, 480]},
        "tracks": {"red_block": {"frames": frames, "coverage": 1.0}, "red_bin": {"frames": bin_frames, "coverage": 1.0}},
    }


def _canonical_plan():
    from core_robot.planning.schema import Plan

    return Plan.model_validate({
        "objects": [{"name": "red_block", "color": "red", "kind": "block"},
                    {"name": "red_bin", "color": "red", "kind": "bin"}],
        "steps": [{"index": 0, "action": "pick_and_place", "object": "red_block", "destination": "red_bin",
                   "note": "canonical note"}],
        "goal": [{"subject": "red_block", "relation": "in", "target": "red_bin"}],
    })


def _canonical_scene() -> dict:
    return {
        "red_block": {"position": [0.4, 0.1, 0.02], "orientation": [0, 0, 0, 1], "color": "red", "kind": "block"},
        "red_bin": {"position": [0.55, -0.1, 0.0], "orientation": [0, 0, 0, 1], "color": "red", "kind": "bin"},
    }


def prompt_material() -> dict:
    """The rendered prompt texts that the hash covers."""
    from core_robot.codegen.api_doc import primitives_reference
    from core_robot.codegen.muse_generator import CODE_SCHEMA
    from core_robot.codegen.prompt import build_codegen_prompt
    from core_robot.planning.schema import Plan
    from core_robot.planning.video_planner import PLANNER_SYSTEM, planner_prompt

    api = primitives_reference()
    plan, scene = _canonical_plan(), _canonical_scene()
    sys1, first = build_codegen_prompt(plan, scene, api)
    sys2, retry = build_codegen_prompt(plan, scene, api, feedback="<failure report>", with_image=True)
    return {
        "planner": {"system": PLANNER_SYSTEM, "prompt": planner_prompt(_canonical_perception()),
                    "schema": Plan.model_json_schema()},
        "codegen": {"system": [sys1, sys2], "first_attempt": first, "retry": retry, "schema": CODE_SCHEMA},
        "api_reference": api,
    }


@lru_cache(maxsize=1)
def prompt_hash() -> str:
    blob = json.dumps(prompt_material(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def frozen() -> dict:
    return json.loads(FROZEN_FILE.read_text())


def git_commit() -> str:
    """HEAD commit; "+dirty" when tracked files have uncommitted changes; "unknown" outside git."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")
