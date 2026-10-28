"""Run generated robot code in a separate process with a wall-clock timeout.

Contract for generated code: define a top-level function

    def run(robot, scene):
        ...

  robot  exposes only pick(obj), place(obj, target), move_above(target), open_gripper().
         Each returns {"ok": bool, "reason": str, "details": dict}; every call is logged.
  scene  read-only data: a deep copy of the starting scene_state() for this layout,
         name -> {"position", "orientation", "color", "kind"}. Changing it has no effect
         on the simulator. It is information, not a primitive.

Before anything runs, the code is parsed with `ast` and rejected if it imports anything,
touches any attribute starting with "_" (including dunders) or an interpreter-internal
attribute (gi_frame, f_globals, str.format, ...), or names a forbidden builtin
(open, exec, eval, compile, getattr, ...). It then runs in a fresh Python process with only
a small set of safe builtins, a scrubbed environment (no API key), and a timeout.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TIMEOUT_S = 60.0

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir", "input", "breakpoint", "help", "memoryview", "type",
    "object", "super", "classmethod", "staticmethod", "property", "exit", "quit", "id",
}

# Attributes that reach interpreter internals without a leading underscore, plus
# str.format/format_map, whose format strings can walk dunder attributes ("{0.__class__}").
FORBIDDEN_ATTRS = {
    "format", "format_map", "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
    "f_globals", "f_locals", "f_builtins", "f_back", "f_code", "tb_frame", "tb_next",
    "func_globals", "func_code", "with_traceback",
}

SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "print", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "Exception", "ValueError", "KeyError", "IndexError",
    "TypeError", "RuntimeError", "ZeroDivisionError", "AssertionError",
]

SECRET_ENV = ("MODEL_API_KEY",)


def validate_code(code: str) -> list[str]:
    """Static checks. Returns a list of problems; empty means the code may run."""
    try:
        tree = ast.parse(code, filename="<generated>")
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    problems = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append(f"line {line}: imports are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            problems.append(f"line {line}: access to private/dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"line {line}: attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"line {line}: '{node.id}' is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name.startswith("__"):
            problems.append(f"line {line}: dunder definitions are not allowed")
    runs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
    if not runs:
        problems.append("code must define a top-level function run(robot, scene)")
    elif len(runs[0].args.args) != 2:
        problems.append("run must take exactly two parameters: run(robot, scene)")
    return problems


def run_code(
    code: str,
    task_id: str,
    exec_seed: int,
    out_dir: str | Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    record: bool = True,
    calibration_offset_cm: float | None = None,
) -> dict:
    """Validate and execute `code` against a fresh scene for (task_id, exec_seed).

    Returns {ok, exception, call_log, initial_state, final_state, video, frames, stdout,
    timed_out, rejected, sandbox_error}. sandbox_error means the harness itself failed,
    not the generated code. `ok` means the code ran to completion without an exception or
    timeout; it says nothing about task success (use verify.checker for that).
    """
    out = Path(out_dir).resolve()  # the child runs with cwd=out, so paths must be absolute
    out.mkdir(parents=True, exist_ok=True)
    base = {
        "ok": False, "exception": None, "call_log": [], "initial_state": None, "final_state": None,
        "video": None, "frames": {}, "stdout": "", "timed_out": False, "rejected": False,
        "sandbox_error": False,
    }
    problems = validate_code(code)
    if problems:
        return {**base, "rejected": True, "exception": "rejected before running: " + "; ".join(problems)}

    (out / "code.py").write_text(code)
    job = {"code": code, "task_id": task_id, "exec_seed": exec_seed, "out_dir": str(out), "record": record,
           "calibration_offset_cm": calibration_offset_cm}
    (out / "job.json").write_text(json.dumps(job))
    for stale in ("result.json", "calls.jsonl"):
        (out / stale).unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "core_robot.codegen._child", str(out / "job.json")],
            capture_output=True, text=True, timeout=timeout_s, env=child_env(), cwd=str(out),
        )
    except subprocess.TimeoutExpired:
        return {
            **base, "timed_out": True, "exception": f"timeout: run(robot, scene) exceeded {timeout_s:g} s",
            "call_log": _read_calls(out / "calls.jsonl"),
            "video": str(out / "video.mp4") if (out / "video.mp4").exists() else None,
        }
    result_path = out / "result.json"
    if not result_path.exists():
        return {**base, "sandbox_error": True, "exception": f"sandbox process failed (exit {proc.returncode}): {proc.stderr[-2000:]}",
                "call_log": _read_calls(out / "calls.jsonl")}
    return {**base, **json.loads(result_path.read_text())}


def child_env() -> dict:
    """Environment for the sandbox process: the parent's, minus secrets."""
    return {k: v for k, v in os.environ.items() if k not in SECRET_ENV}


def _read_calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
