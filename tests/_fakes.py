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


# ---- SAM 3.1 wire format ----

def _mask_payload(h: int, w: int) -> str:
    """A real one_bit payload for a filled h x w raster (decodable by meta_sam_parser)."""
    from meta_sam_parser._mask_codec import _encode_raster  # test-only use of the reference encoder

    return _encode_raster([1] * (h * w), w, h)


def sam_record(oid: int, box, size=(640, 480)) -> str:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = max(1, y2 - y1 + 1), max(1, x2 - x1 + 1)
    return (f"{oid}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={size[0]};h={size[1]}|>"
            f"<|mask;x=0;y=0;data={h},{w},{_mask_payload(h, w)}|>")


def sam_text(frames: dict[int, list[tuple[int, tuple]]]) -> str:
    """{frame: [(object_id, box), ...]} -> SAM output text, one line per frame with objects."""
    return "\n".join(f"<{f}f>" + ",".join(sam_record(o, b) for o, b in recs)
                     for f, recs in sorted(frames.items()) if recs)


def sam_payload(text: str, frames_processed: int | None = None) -> dict:
    usage = {"video_frames_processed": frames_processed} if frames_processed is not None else {}
    return responses_payload(text, model="sam-3.1", input_tokens=0, output_tokens=0, extra_usage=usage)


def media_sha(body: dict) -> str | None:
    """sha256 of the first inline image/video in a request (to tell which file was sent)."""
    import base64
    import hashlib

    for p in user_parts(body):
        url = p.get("video_url") or p.get("image_url")
        if url and url.startswith("data:"):
            return hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
    return None
