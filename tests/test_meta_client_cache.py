"""Cache, replay, retry and logging behavior of MetaClient, with a fake transport (no network)."""

import json
from dataclasses import replace

import pytest

from core_robot.config import Settings
from core_robot.meta_client import ApiError, MetaClient, ReplayMiss, parse_sam


class FakeTransport:
    def __init__(self, failures=None):
        self.calls = []
        self.failures = list(failures or [])

    def __call__(self, method, path, json_body=None, upload=None):
        self.calls.append((method, path, json_body, upload))
        if self.failures:
            raise self.failures.pop(0)
        if path == "/files":
            return {"id": "file-123"}
        if path == "/models":
            return {"data": [{"id": "muse-spark-1.3"}, {"id": "sam-3.1"}]}
        text = json_body["input"][0]["content"][0]["text"]
        return {"output_text": f"echo:{text}", "usage": {"input_tokens": 3, "output_tokens": 2}}


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")


def make(settings, transport):
    return MetaClient(settings=settings, transport=transport, sleep=lambda s: None)


def log_lines(settings):
    return [json.loads(l) for l in settings.log_path.read_text().splitlines()]


def test_miss_then_hit_calls_transport_once(settings):
    t = FakeTransport()
    c = make(settings, t)
    assert c.muse("hello")["text"] == "echo:hello"
    assert c.muse("hello")["text"] == "echo:hello"
    assert len(t.calls) == 1
    assert [l["cache_hit"] for l in log_lines(settings)] == [False, True]


def test_cache_survives_new_client(settings):
    make(settings, FakeTransport()).muse("hello")
    t2 = FakeTransport()
    assert make(settings, t2).muse("hello")["text"] == "echo:hello"
    assert t2.calls == []


def test_different_requests_different_keys(settings):
    t = FakeTransport()
    c = make(settings, t)
    c.muse("a")
    c.muse("b")
    c.muse("a", max_output_tokens=10)
    assert len(t.calls) == 3
    assert len(set(l["key"] for l in log_lines(settings))) == 3


def test_media_keyed_by_content_not_path(settings, tmp_path):
    a, b, c_ = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c_.write_bytes(b"different")
    t = FakeTransport()
    c = make(settings, t)
    c.muse("look", images=[a])
    c.muse("look", images=[b])
    assert len(t.calls) == 1
    c.muse("look", images=[c_])
    assert len(t.calls) == 2


def test_inline_image_sent_as_data_url(settings, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG")
    t = FakeTransport()
    make(settings, t).muse("look", images=[img])
    part = t.calls[0][2]["input"][0]["content"][1]
    assert part["type"] == "input_image"
    assert part["image_url"].startswith("data:image/png;base64,")


def test_file_mode_uploads_once_and_cache_hit_skips_upload(settings, tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"video")
    t = FakeTransport()
    c = make(settings, t)
    c.sam_video(vid, "red block", video_mode="file")
    assert [call[1] for call in t.calls] == ["/files", "/responses"]
    assert t.calls[1][2]["input"][0]["content"][1] == {"type": "input_video", "file_id": "file-123"}
    c.sam_video(vid, "red block", video_mode="file")
    assert len(t.calls) == 2


def test_replay_miss_raises_without_network(settings):
    t = FakeTransport()
    c = make(replace(settings, replay=True), t)
    with pytest.raises(ReplayMiss):
        c.muse("never cached")
    assert t.calls == []
    assert log_lines(settings)[-1]["error"] == "replay_miss"


def test_replay_hit_returns_cached(settings):
    make(settings, FakeTransport()).muse("cached")
    t = FakeTransport()
    assert make(replace(settings, replay=True), t).muse("cached")["text"] == "echo:cached"
    assert t.calls == []


def test_retries_429_and_5xx_then_succeeds(settings):
    t = FakeTransport(failures=[ApiError(429, "slow down", retry_after=1), ApiError(503, "busy")])
    sleeps = []
    c = MetaClient(settings=settings, transport=t, sleep=sleeps.append)
    assert c.muse("x")["text"] == "echo:x"
    assert len(t.calls) == 3
    assert len(sleeps) == 2 and sleeps[0] >= 1


def test_no_retry_on_400_and_nothing_cached(settings):
    t = FakeTransport(failures=[ApiError(400, "bad")])
    c = make(settings, t)
    with pytest.raises(ApiError):
        c.muse("x")
    assert len(t.calls) == 1
    assert list(settings.cache_dir.glob("*.json")) == []


def test_gives_up_after_max_attempts(settings):
    t = FakeTransport(failures=[ApiError(500, "boom")] * 10)
    c = make(replace(settings, max_attempts=3), t)
    with pytest.raises(ApiError):
        c.muse("x")
    assert len(t.calls) == 3


def test_log_records_usage_and_model(settings):
    make(settings, FakeTransport()).muse("x")
    rec = log_lines(settings)[0]
    assert rec["model"] == "muse-spark-1.3"
    assert rec["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert "latency_s" in rec


def test_list_models_cached(settings):
    t = FakeTransport()
    c = make(settings, t)
    assert c.list_models() == ["muse-spark-1.3", "sam-3.1"]
    c.list_models()
    assert len(t.calls) == 1


def test_parse_sam_sparse_frames_and_ids():
    text = (
        "<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=8|><|mask;x=0;y=0;data=2,2,~ab|>,"
        "2<|box;x1=5;y1=5;x2=6;y2=6;w=10;h=8;c=0.9|><|mask;x=0;y=0;c=0.9;data=1,1,!z|>\n"
        "<3f>2<|box;x1=6;y1=6;x2=7;y2=7;w=10;h=8|><|mask;x=0;y=0;data=1,1,~q|>\n"
    )
    frames = parse_sam(text)
    assert sorted(frames) == [0, 3]
    assert sorted(frames[0]) == [0, 2]
    assert frames[0][0]["box"] == (1, 2, 3, 4)
    assert frames[0][2]["conf"] == 0.9
    assert frames[3][2]["mask"] == "1,1,~q"
