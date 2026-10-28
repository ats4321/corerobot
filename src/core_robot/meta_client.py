"""Thin client for the Meta Model API (Muse Spark and SAM 3.1).

Every call goes through `MetaClient._call`, which:
  * computes a cache key from the full logical request (local media hashed by content),
  * returns the cached response on a hit (free reruns),
  * raises `ReplayMiss` on a miss when replay mode is on (CORE_ROBOT_REPLAY=1),
  * otherwise sends the request with retries on 429 / 5xx and caches the result,
  * appends one JSONL line per call: model, latency, token usage, cache hit.

The network layer is a `Transport` (a callable), so tests can inject a fake one.
Request shapes follow https://dev.meta.ai/docs (Responses API, input_text /
input_image / input_video parts, base64 data URLs or Files API `file_id`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import httpx

from core_robot.config import Settings, load_settings

RETRYABLE_STATUS = {429, 500, 502, 503}

# Requests currently being sent, process-wide: (cache_dir, request key) -> Event set when done.
_INFLIGHT: dict[tuple[str, str], threading.Event] = {}
_INFLIGHT_LOCK = threading.Lock()  # 504 is not retried: docs say it recurs unchanged.


class ApiError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.retry_after = retry_after


class ReplayMiss(Exception):
    """Raised in replay mode when a request is not in the cache."""


class Transport(Protocol):
    def __call__(
        self, method: str, path: str, json_body: dict | None = None, upload: Path | None = None
    ) -> dict: ...


class HttpxTransport:
    """Real network transport. Raises ApiError for non-2xx responses."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise RuntimeError("MODEL_API_KEY is not set (put it in .env at the repo root)")
        self._http = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout_s,
        )

    def __call__(self, method, path, json_body=None, upload=None):
        try:
            if upload is not None:
                with open(upload, "rb") as f:
                    mime = mimetypes.guess_type(upload.name)[0] or "application/octet-stream"
                    resp = self._http.post(
                        path, data={"purpose": "user_data"}, files={"file": (upload.name, f, mime)}
                    )
            else:
                resp = self._http.request(method, path, json=json_body)
        except httpx.TransportError as e:
            raise ApiError(599, f"network error: {e!r}") from e
        if resp.status_code >= 400:
            retry_after = resp.headers.get("retry-after")
            raise ApiError(
                resp.status_code,
                resp.text[:2000],
                float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None,
            )
        return resp.json()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def request_key(logical_request: dict) -> str:
    """Stable hash of a logical request (sorted-key JSON)."""
    blob = json.dumps(logical_request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def usage_tokens(response: dict) -> tuple[int, int]:
    """(input, output) tokens from a Responses or Chat Completions usage block; (0, 0) if absent."""
    u = (response or {}).get("usage") or {}
    t_in = u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
    t_out = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
    return int(t_in), int(t_out)


def output_text(response: dict) -> str:
    """Concatenate output_text parts from a Responses API response."""
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)


# --- SAM output parsing (grammar: https://dev.meta.ai/docs/sam/reading-segmentation) ---

_FRAME_RE = re.compile(r"<(\d+)f>")
_RECORD_RE = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*?;data=[^|]*)\|>")


def _kv(s: str) -> dict:
    return dict(p.split("=", 1) for p in s.split(";") if "=" in p)


def parse_sam(text: str) -> dict[int, dict[int, dict]]:
    """Parse SAM output into {frame: {object_id: {"box": (x1,y1,x2,y2), "w", "h", "conf", "mask"}}}.

    The mask is returned as its raw `H,W,<enc>payload` string; decode it with
    meta_sam_parser.decode_mask_to_raster when pixel masks are needed.
    """
    frames: dict[int, dict[int, dict]] = {}
    for line in text.splitlines():
        m = _FRAME_RE.match(line.strip())
        if not m:
            continue
        objs = frames.setdefault(int(m.group(1)), {})
        for rec in _RECORD_RE.finditer(line):
            box = _kv(rec.group(2))
            mask_fields = rec.group(3)
            objs[int(rec.group(1))] = {
                "box": tuple(int(box[k]) for k in ("x1", "y1", "x2", "y2")),
                "w": int(box["w"]),
                "h": int(box["h"]),
                "conf": float(box["c"]) if "c" in box else None,
                "mask": mask_fields.split("data=", 1)[1],
            }
    return frames


@dataclass
class MetaClient:
    settings: Settings = field(default_factory=load_settings)
    transport: Transport | None = None
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self):
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_ids: dict[str, str] = {}
        self._lock = threading.Lock()  # the client is shared by evaluation worker threads

    # ---- public API ----

    def list_models(self) -> list[str]:
        resp = self._call("models", {"method": "GET", "path": "/models"}, lambda: ("GET", "/models", None))
        return [m["id"] for m in resp.get("data", [])]

    def muse(
        self,
        text: str,
        images: list[Path] | None = None,
        video: Path | None = None,
        video_mode: str = "inline",
        schema: dict | None = None,
        system: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        """Call Muse Spark via the Responses API. Returns {"text", "raw", "usage": (in, out)}.

        With a schema, also "json" (parsed, or None) and "json_error" (None, or why parsing failed).
        Also "status" and "incomplete_reason" (e.g. "max_output_tokens" when reasoning used the
        whole output budget). reasoning_effort, if given, is sent as reasoning.effort
        ("minimal" ... "max"); omitted, the API default applies.

        images: local image files, sent inline as base64 data URLs (input_image).
        video: a local .mp4, sent inline as a data URL (input_video, video_mode="inline")
               or uploaded via the Files API and referenced by file_id (video_mode="file").
        schema: optional JSON schema; sets text.format to json_schema and parses the reply.
        """
        parts: list[dict] = [{"type": "input_text", "text": text}]
        for img in images or []:
            parts.append({"type": "input_image", "image_url": _MediaRef(Path(img), "url")})
        if video is not None:
            if video_mode == "file":
                parts.append({"type": "input_video", "file_id": _MediaRef(Path(video), "file")})
            else:
                parts.append({"type": "input_video", "video_url": _MediaRef(Path(video), "url")})
        body: dict[str, Any] = {"model": self.settings.muse_model, "input": []}
        if system:
            body["instructions"] = system
        body["input"].append({"type": "message", "role": "user", "content": parts})
        if schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "output", "schema": schema}}
        if max_output_tokens:
            body["max_output_tokens"] = max_output_tokens
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        resp = self._responses(body)
        out = {"text": output_text(resp), "raw": resp, "usage": usage_tokens(resp), "status": resp.get("status"),
               "request_key": request_key(_logical(body)),
               "incomplete_reason": (resp.get("incomplete_details") or {}).get("reason")}
        if schema is not None:
            try:
                out["json"], out["json_error"] = json.loads(out["text"]), None
            except json.JSONDecodeError as e:
                out["json"], out["json_error"] = None, f"invalid JSON: {e}"
        return out

    def sam_image(self, image: Path, prompt: str, metadata: dict | None = None) -> dict:
        """Segment an image with a concept prompt (a short noun phrase)."""
        return self._sam({"type": "input_image", "image_url": _MediaRef(Path(image), "url")}, prompt, metadata)

    def sam_video(self, video: Path, prompt: str, video_mode: str = "inline", metadata: dict | None = None) -> dict:
        """Segment and track a concept across a video. video_mode: "inline" or "file"."""
        part = (
            {"type": "input_video", "file_id": _MediaRef(Path(video), "file")}
            if video_mode == "file"
            else {"type": "input_video", "video_url": _MediaRef(Path(video), "url")}
        )
        return self._sam(part, prompt, metadata)

    # ---- internals ----

    def _sam(self, media_part: dict, prompt: str, metadata: dict | None) -> dict:
        body: dict[str, Any] = {
            "model": self.settings.sam_model,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}, media_part]}
            ],
        }
        if metadata:
            body["metadata"] = metadata
        resp = self._responses(body)
        text = output_text(resp)
        return {"text": text, "frames": parse_sam(text), "raw": resp}

    def _responses(self, body: dict) -> dict:
        logical = _logical(body)
        return self._call(body["model"], logical, lambda: ("POST", "/responses", _materialize(body, self._upload)))

    def _upload(self, path: Path) -> str:
        digest = _sha256_file(path)
        with self._lock:
            known = self._file_ids.get(digest)
        if known is None:
            resp = self._send_with_retries(lambda: self._transport()("POST", "/files", None, path))
            with self._lock:
                self._file_ids[digest] = resp["id"]
        return self._file_ids[digest]

    def _transport(self) -> Transport:
        if self.transport is None:
            self.transport = HttpxTransport(self.settings)
        return self.transport

    def _call(self, model: str, logical: dict, build: Callable[[], tuple]) -> dict:
        """Cache lookup, then (once per key) the network call.

        Identical requests issued concurrently (e.g. attempt 1 of conditions A, B and C running
        in parallel) are coalesced: the first caller sends the request, the others wait for it
        and read the cached response, so they all get the same answer and only one is billed.
        """
        key = request_key(logical)
        kind = _media_kind(logical)
        path = self.settings.cache_dir / f"{key}.json"
        flight_key = (str(self.settings.cache_dir), key)
        while True:
            if path.exists():
                resp = json.loads(path.read_text())["response"]
                self._log(model, key, kind, cache_hit=True, latency_s=0.0, resp=resp)
                return resp
            if self.settings.replay:
                self._log(model, key, kind, cache_hit=False, latency_s=0.0, resp=None, error="replay_miss")
                raise ReplayMiss(f"request {key[:12]} for {model} not in cache (CORE_ROBOT_REPLAY=1)")
            with _INFLIGHT_LOCK:
                waiting_on = _INFLIGHT.get(flight_key)
                if waiting_on is None:
                    _INFLIGHT[flight_key] = threading.Event()
            if waiting_on is None:
                break  # this caller sends the request
            waiting_on.wait()  # another caller is sending it; then re-check the cache
        try:
            t0 = time.monotonic()
            try:
                method, api_path, body = build()
                resp = self._send_with_retries(lambda: self._transport()(method, api_path, body, None))
            except Exception as e:
                self._log(model, key, kind, cache_hit=False, latency_s=time.monotonic() - t0, resp=None,
                          error=str(e)[:500])
                raise
            latency = time.monotonic() - t0
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({"request": logical, "response": resp}))
            tmp.replace(path)
            self._log(model, key, kind, cache_hit=False, latency_s=latency, resp=resp)
            return resp
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.pop(flight_key).set()

    def _send_with_retries(self, send: Callable[[], dict]) -> dict:
        delay = 0.5
        for attempt in range(1, self.settings.max_attempts + 1):
            try:
                return send()
            except ApiError as e:
                retryable = e.status in RETRYABLE_STATUS or e.status == 599
                if not retryable or attempt == self.settings.max_attempts:
                    raise
                wait = max(e.retry_after or 0.0, delay) + random.uniform(0, delay)
                self.sleep(wait)
                delay *= 2
        raise AssertionError("unreachable")

    def _log(self, model, key, kind, cache_hit, latency_s, resp, error=None):
        usage = (resp or {}).get("usage") if isinstance(resp, dict) else None
        rec = {
            "ts": time.time(),
            "model": model,
            "kind": kind,
            "key": key,
            "cache_hit": cache_hit,
            "latency_s": round(latency_s, 3),
            "usage": usage,
        }
        if error:
            rec["error"] = error
        with self._lock, open(self.settings.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


@dataclass(frozen=True)
class _MediaRef:
    """Placeholder for local media inside a request body.

    In the cache key it becomes the file's sha256; when sending it becomes a base64
    data URL ("url") or an uploaded file_id ("file").
    """

    path: Path
    mode: str


def _media_kind(logical: dict) -> str:
    """"video", "image" or "text": the richest media part in a request (for spend accounting)."""
    blob = json.dumps(logical)
    return "video" if '"input_video"' in blob else "image" if '"input_image"' in blob else "text"


def _logical(obj):
    if isinstance(obj, _MediaRef):
        return {"media_sha256": _sha256_file(obj.path), "mode": obj.mode}
    if isinstance(obj, dict):
        return {k: _logical(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_logical(v) for v in obj]
    return obj


def _materialize(obj, upload: Callable[[Path], str]):
    if isinstance(obj, _MediaRef):
        return upload(obj.path) if obj.mode == "file" else data_url(obj.path)
    if isinstance(obj, dict):
        return {k: _materialize(v, upload) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_materialize(v, upload) for v in obj]
    return obj
