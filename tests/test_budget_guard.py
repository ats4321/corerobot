"""The budget guard reserves each call's worst-case cost before sending, so calls already in
flight can never push spending past the budget."""

import threading
import time

import pytest
from _fakes import ScriptedTransport, code_reply, responses_payload

from core_robot.config import Settings
from core_robot.meta_client import BudgetExceeded, BudgetGuard, MetaClient, _muse_cost


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key=None, cache_dir=tmp_path / "cache", log_path=tmp_path / "log.jsonl")


def slow_reply(method, path, body):
    time.sleep(0.3)  # keep calls in flight together
    return code_reply("def run(robot, scene):\n    pass\n", input_tokens=2000, output_tokens=12000)


def test_reserve_settle_bookkeeping():
    g = BudgetGuard(0.10)
    g.reserve(0.06)
    with pytest.raises(BudgetExceeded):
        g.reserve(0.05)  # 0.06 in flight + 0.05 > 0.10
    g.settle(0.06, 0.02)
    assert g.spent == pytest.approx(0.02) and g.reserved == 0
    g.reserve(0.05)
    assert g.remaining() == pytest.approx(0.03)


def test_concurrent_calls_never_exceed_budget(settings):
    """Eight distinct calls at once, each worst case ~$0.052; budget $0.12 admits only two."""
    t = ScriptedTransport(slow_reply)
    guard = BudgetGuard(0.12)
    client = MetaClient(settings=settings, transport=t, guard=guard)
    results, errors = [], []

    def call(i):
        try:
            results.append(client.muse(f"prompt {i}", max_output_tokens=12000))
        except BudgetExceeded as e:
            errors.append(e)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(t.calls) == len(results) == 2 and len(errors) == 6
    assert guard.spent <= guard.budget and guard.reserved == pytest.approx(0)
    assert guard.spent == pytest.approx(2 * _muse_cost(2000, 12000))


def test_refused_call_sends_nothing_and_is_logged(settings):
    import json

    t = ScriptedTransport(slow_reply)
    client = MetaClient(settings=settings, transport=t, guard=BudgetGuard(0.01))
    with pytest.raises(BudgetExceeded, match="would exceed"):
        client.muse("x", max_output_tokens=12000)
    assert t.calls == []
    last = json.loads(settings.log_path.read_text().splitlines()[-1])
    assert last["error"].startswith("budget:")


def test_cache_hits_cost_nothing(settings):
    t = ScriptedTransport(lambda m, p, b: responses_payload("ok", input_tokens=10, output_tokens=10))
    MetaClient(settings=settings, transport=t).muse("cached", max_output_tokens=100)
    guard = BudgetGuard(0.0)  # no money at all
    out = MetaClient(settings=settings, transport=t, guard=guard).muse("cached", max_output_tokens=100)
    assert out["text"] == "ok" and guard.spent == 0 and len(t.calls) == 1


def test_failed_call_keeps_its_reservation_as_spent(settings):
    from core_robot.meta_client import ApiError

    def boom(m, p, b):
        raise ApiError(400, "bad request")

    guard = BudgetGuard(1.0)
    client = MetaClient(settings=settings, transport=ScriptedTransport(boom), guard=guard)
    with pytest.raises(ApiError):
        client.muse("x", max_output_tokens=1000)
    assert guard.reserved == pytest.approx(0) and guard.spent > 0


def test_sam_video_reserves_billed_frames(settings, tmp_path):
    from _fakes import sam_payload

    from core_robot.config import DEMOS_DIR, SAM_PRICE_PER_FRAME
    from core_robot.perception.frames import resample_video

    clip = resample_video(DEMOS_DIR / "t1_red_in_red_bin/seed_0/video.mp4", tmp_path / "c.mp4")
    guard = BudgetGuard(1.0)
    seen = []

    def reply(m, p, b):
        seen.append(guard.reserved)
        return sam_payload("", frames_processed=50)

    MetaClient(settings=settings, transport=ScriptedTransport(reply), guard=guard).sam_video(clip["path"], "red cube")
    assert seen[0] == pytest.approx(max(50, clip["n_frames"]) * SAM_PRICE_PER_FRAME)
    assert guard.spent == pytest.approx(50 * SAM_PRICE_PER_FRAME)
