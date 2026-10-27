"""Offline stand-ins for the Meta Model API: recorded-looking Responses payloads and a
scripted transport that records every request it receives."""

from __future__ import annotations

import itertools
import json

_ids = itertools.count(1)


def responses_payload(text: str, model: str = "muse-spark-1.3", input_tokens: int = 4200,
                      output_tokens: int = 350, extra_usage: dict | None = None) -> dict:
    """A Responses API body shaped like a real one (reasoning item, message, usage)."""
    n = next(_ids)
    return {
        "id": f"resp_{n:024x}", "object": "response", "created_at": 1761600000 + n, "status": "completed",
        "model": model,
        "output": [
            {"type": "reasoning", "id": f"rs_{n:020x}", "summary": []},
            {"type": "message", "id": f"msg_{n:020x}", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text, "annotations": []}]},
        ],
        "usage": {"input_tokens": input_tokens, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": output_tokens // 3},
                  "total_tokens": input_tokens + output_tokens, **(extra_usage or {})},
    }


def user_parts(body: dict) -> list[dict]:
    return [p for m in body.get("input", []) for p in m.get("content", [])]


def user_text(body: dict) -> str:
    return "\n".join(p["text"] for p in user_parts(body) if p["type"] == "input_text")


def all_text(body: dict) -> str:
    """Every piece of text sent: instructions plus input_text parts."""
    return (body.get("instructions") or "") + "\n" + user_text(body)


class ScriptedTransport:
    """Transport whose reply is computed by handler(method, path, body) -> dict. Records calls."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, str, dict | None]] = []

    def __call__(self, method, path, json_body=None, upload=None):
        self.calls.append((method, path, json_body))
        if path == "/files":
            return {"id": "file-842549258569145", "object": "file", "purpose": "user_data", "status": "uploaded"}
        return self.handler(method, path, json_body)

    def bodies(self, model: str | None = None) -> list[dict]:
        return [b for _, p, b in self.calls if p == "/responses" and (model is None or b["model"] == model)]


def code_reply(code: str, **kw) -> dict:
    return responses_payload(json.dumps({"code": code}), **kw)
