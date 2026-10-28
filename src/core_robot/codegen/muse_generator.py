"""Live code generator: Muse Spark writes run(robot, scene) for a plan.

Every request goes through MetaClient, so it is cached by request hash (and replayable
with CORE_ROBOT_REPLAY=1). Returns {"code", "tokens_in", "tokens_out"} for the repair loop.
"""

from __future__ import annotations

import re

from core_robot.codegen.api_doc import primitives_reference
from core_robot.codegen.prompt import build_codegen_prompt

CODE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "description": "Python source defining run(robot, scene)"}},
    "required": ["code"],
    "additionalProperties": False,
}
# Room for reasoning plus the code: at 6000 tokens some replies were all reasoning and no code.
DEFAULT_MAX_OUTPUT_TOKENS = 12000
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(result: dict) -> str:
    """Code from a muse() result: the JSON "code" field if present, else the raw text.

    If the chosen text contains a fenced code block, the block (preferring one that defines
    run) is used instead of the surrounding prose.
    """
    data = result.get("json")
    if isinstance(data, dict) and isinstance(data.get("code"), str) and data["code"].strip():
        text = data["code"]
    else:
        text = result.get("text") or ""
    blocks = _FENCE.findall(text)
    if blocks:
        text = next((b for b in blocks if "def run" in b), blocks[0])
    return text.strip("\n").rstrip() + "\n"


class MuseGenerator:
    """generate_fn for run_with_repair, backed by muse-spark-1.3."""

    name = "muse"
    run_source = "live"

    def __init__(self, client=None, max_output_tokens: int | None = None):
        if client is None:
            from core_robot.meta_client import MetaClient
            client = MetaClient()
        self.client = client
        self.max_output_tokens = max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS
        self._api = primitives_reference()

    def _request(self, plan, scene, feedback=None, image_path=None) -> tuple[str, dict]:
        system, text = build_codegen_prompt(plan, scene, self._api, feedback, with_image=image_path is not None)
        return text, {"images": [image_path] if image_path else None, "schema": CODE_SCHEMA, "system": system,
                      "max_output_tokens": self.max_output_tokens}

    def request_key(self, *, plan, scene, feedback=None, image_path=None) -> str:
        """The cache key of this generation request, computed without sending it."""
        text, kw = self._request(plan, scene, feedback, image_path)
        return self.client.muse_request_key(text, **kw)

    def __call__(self, *, task, plan, attempt, scene, feedback=None, image_path=None) -> dict:
        text, kw = self._request(plan, scene, feedback, image_path)
        r = self.client.muse(text, **kw)
        t_in, t_out = r["usage"]
        return {"code": extract_code(r), "tokens_in": t_in, "tokens_out": t_out, "request_key": r["request_key"],
                "status": r.get("status"), "incomplete_reason": r.get("incomplete_reason")}
