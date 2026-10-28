"""UI live path, offline: confirmation gate, stage streaming, errors on the page."""

import pytest
from test_real_pipeline import FakeRealWorld, _client, _clip

from core_robot import prompt_freeze
from core_robot.ui import live
from core_robot.ui.app import MAX_ATTEMPTS_SHOWN, build_app, live_view


def test_no_live_call_without_confirmation(tmp_path):
    called = []
    evs = list(live.events(str(_clip(tmp_path)), replay=False, confirmed=False,
                           client_factory=lambda *a: called.append(a)))
    assert evs == [("error", {"message": "Tick the confirmation box to allow live API spend (up to $0.50)."})]
    assert called == []


def test_missing_video_is_an_error(tmp_path):
    assert list(live.events(None, replay=False, confirmed=True))[0][0] == "error"


def test_replay_needs_no_confirmation(tmp_path):
    seen = []

    def factory(replay, budget):
        seen.append(replay)
        return _client(tmp_path, FakeRealWorld())

    evs = list(live.events(str(_clip(tmp_path)), replay=True, confirmed=False, client_factory=factory,
                           out_root=tmp_path / "real"))
    assert seen == [True] and evs[-1][0] == "outcome"


def test_stages_stream_in_order_and_render(tmp_path):
    evs = list(live.events(str(_clip(tmp_path)), replay=False, confirmed=True,
                           client_factory=lambda r, b: _client(tmp_path, FakeRealWorld()), out_root=tmp_path / "real"))
    assert [n for n, _ in evs] == ["normalized", "perception", "plan", "attempt", "attempt", "outcome"]
    frames = list(live_view(iter(evs)))
    assert len(frames) == len(evs)
    last = frames[-1]
    assert len(last) == 4 + 3 * MAX_ATTEMPTS_SHOWN + 1
    status, norm, overlay, plan = last[:4]
    assert "Normalized" in status and "Attempt 1: failed" in status and "Attempt 2: success" in status
    assert norm.endswith("normalized.mp4") and overlay.endswith("overlay.mp4") and plan["steps"]
    assert last[-1].startswith("**SUCCESS**") and "plan's own goal" in last[-1]


def test_changed_prompts_are_refused_on_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_freeze, "prompt_hash", lambda: "000000000000")
    evs = list(live.events(str(_clip(tmp_path)), replay=False, confirmed=True,
                           client_factory=lambda r, b: pytest.fail("no client before the freeze check")))
    assert evs[0][0] == "error" and "prompts changed" in evs[0][1]["message"]


def test_pipeline_errors_shown_not_raised(tmp_path):
    def broken_factory(replay, budget):
        raise RuntimeError("MODEL_API_KEY is not set (add it to .env)")

    evs = list(live.events(str(_clip(tmp_path)), replay=False, confirmed=True, client_factory=broken_factory))
    assert evs == [("error", {"message": "MODEL_API_KEY is not set (add it to .env)"})]
    frame = list(live_view(iter(evs)))[-1]
    assert "MODEL_API_KEY" in frame[0]


def test_estimate_text_for_upload(tmp_path):
    text = live.estimate_text(str(_clip(tmp_path)))
    assert "expected" in text and "worst case" in text and "$" in text
    assert live.estimate_text(None).startswith("Upload")


def test_app_builds_with_live_section():
    assert build_app() is not None
