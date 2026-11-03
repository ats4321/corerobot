"""Real-footage perception mode: fragment merging, class-based instance selection, image-relative
plausibility (recorded SAM outputs of the synthetic clip, $0), and the plan_matches_clip check."""

import json
from pathlib import Path

import imageio.v2 as iio
import pytest

from core_robot.meta_client import parse_sam
from core_robot.perception import real_mode
from core_robot.perception.perceive import perceive
from core_robot.perception.real_mode import assign, instances, merge_fragments, plausible_real
from core_robot.perception.sam_tracking import select_object
from core_robot.real.expected import compare_plan, expected_plan_path

FIX = Path(__file__).parent / "fixtures" / "synthetic_sort_red_blue"


@pytest.fixture(scope="module")
def recorded():
    d = json.loads((FIX / "sam_video.json").read_text())
    calls = {ph: {"color": c["color"], "frames": parse_sam(c["text"])} for ph, c in d["calls"].items()}
    frames = {int(p.stem.split("_")[1]): iio.imread(p) for p in FIX.glob("frame_*.png")}
    return d["named"], calls, frames, d["n_frames"]


def _rec(box):
    return {"box": tuple(box), "mask": None}


def test_recorded_red_cube_ghost_fragment_merges_but_box_does_not(recorded):
    _, calls, _, _ = recorded
    merged, merges = merge_fragments(instances(calls["red cube"]["frames"]))
    assert sorted(merged) == ["0", "1+2"]  # the red box stays separate
    (m,) = merges
    assert m["merged"] == ["1", "2"] and m["frames"] == [[0, 15], [7, 15]]
    assert m["gap"] == 0 and m["overlap"] >= real_mode.MERGE_OVERLAP and m["size_ratio"] <= real_mode.MERGE_SIZE_RATIO
    assert sorted(merged["1+2"]["frames"]) == list(range(16))


def test_recorded_assignment_by_class_and_size(recorded):
    named, calls, frames, n = recorded
    r = assign(named, calls, frames, n)
    chosen = {k: (t["concept"], t["object_id"]) for k, t in r["tracks"].items()}
    assert chosen == {"red_bin": ("red cube", "0"),  # "red bin" found nothing; the box came back for "red cube"
                      "red_block": ("red cube", "1+2"),  # the cube (with its ghost fragment), not the box
                      "blue_bin": ("blue bin", "0"), "blue_block": ("blue cube", "0")}
    assert r["merges"]["red cube"][0]["merged"] == ["1", "2"]
    assert "smallest" in r["tracks"]["red_block"]["selection"] and "largest" in r["tracks"]["red_bin"]["selection"]


def test_recorded_longest_lived_track_is_the_box_and_is_not_used_for_the_block(recorded):
    _, calls, _, _ = recorded
    # the simulator path keeps ids seen on >= half the longest track; for "red cube" only the red box (id 0)
    assert select_object(calls["red cube"]["frames"], "block") == 0
    named, calls, frames, n = recorded
    assert assign(named, calls, frames, n)["tracks"]["red_block"]["object_id"] != "0"


def test_recorded_plausibility_is_image_relative(recorded):
    _, calls, frames, _ = recorded
    box = calls["red cube"]["frames"][0][0]
    assert plausible_real("red_bin", box["box"], box["mask"], frames[0])[0]
    # ~11% of the frame: under the loose block ceiling; it is kept from being a block by being a bin
    assert plausible_real("red_block", box["box"], box["mask"], frames[0])[0]
    cube = calls["red cube"]["frames"][0][1]
    assert plausible_real("red_block", cube["box"], cube["mask"], frames[0])[0]
    assert not plausible_real("blue_block", cube["box"], cube["mask"], frames[0])[0]  # colour


def test_merge_disjoint_ranges_within_gap():
    a = {f: _rec((100, 100, 130, 130)) for f in range(0, 10)}
    b = {f: _rec((102, 101, 131, 131)) for f in range(12, 20)}  # frames 10-11 missing, same place
    merged, merges = merge_fragments({"0": {"frames": a}, "1": {"frames": b}})
    assert list(merged) == ["0+1"] and merges[0]["gap"] == 3 and merges[0]["shared_frames"] == 0
    assert sorted(merged["0+1"]["frames"]) == list(range(10)) + list(range(12, 20))


def test_no_merge_beyond_gap_elsewhere_or_different_size():
    a = {f: _rec((100, 100, 130, 130)) for f in range(0, 10)}
    late = {f: _rec((100, 100, 130, 130)) for f in range(9 + real_mode.MERGE_GAP + 1, 30)}
    far = {f: _rec((300, 300, 330, 330)) for f in range(5, 15)}
    big = {f: _rec((60, 60, 200, 200)) for f in range(0, 10)}  # a box around the cube
    for other in (late, far, big):
        merged, merges = merge_fragments({"0": {"frames": a}, "1": {"frames": other}})
        assert sorted(merged) == ["0", "1"] and merges == []


def test_assign_bin_takes_largest_block_smallest_without_colour_frames():
    import numpy as np

    frame = np.zeros((480, 640, 3), np.uint8)
    box_bin, box_block = (200, 150, 400, 330), (280, 230, 305, 255)
    calls = {"red cube": {"color": "red", "frames": {f: {0: _rec(box_bin), 1: _rec(box_block)} for f in range(5)}}}
    r = assign(["red_bin", "red_block"], calls, {0: frame}, 5)  # masks absent: no colour check
    assert r["tracks"]["red_bin"]["object_id"] == "0" and r["tracks"]["red_block"]["object_id"] == "1"


def test_perceive_rejects_unknown_footage(tmp_path):
    with pytest.raises(ValueError):
        perceive(tmp_path / "x.mp4", None, tmp_path, footage="phone")


EXPECTED = json.loads((Path(__file__).parents[1] / "data/real/synthetic_sort_red_blue.expected_plan.json").read_text())


def _plan(steps, goal):
    return {"steps": [{"index": i, "action": "pick_and_place", "object": o, "destination": d}
                      for i, (o, d) in enumerate(steps)],
            "goal": [{"subject": s, "relation": "in", "target": t} for s, t in goal]}


def test_plan_matches_clip():
    good = _plan([("red_block", "red_bin"), ("blue_block", "blue_bin")],
                 [("blue_block", "blue_bin"), ("red_block", "red_bin")])  # goal order does not matter
    assert compare_plan(good, EXPECTED) == {"plan_matches_clip": True, "differences": []}
    swapped = _plan([("red_block", "blue_bin"), ("blue_block", "red_bin")],
                    [("red_block", "blue_bin"), ("blue_block", "red_bin")])
    r = compare_plan(swapped, EXPECTED)
    assert not r["plan_matches_clip"] and len(r["differences"]) == 4
    reordered = _plan([("blue_block", "blue_bin"), ("red_block", "red_bin")],
                      [("red_block", "red_bin"), ("blue_block", "blue_bin")])
    r = compare_plan(reordered, EXPECTED)
    assert not r["plan_matches_clip"] and "different order" in r["differences"][0]
    assert compare_plan(None, EXPECTED)["plan_matches_clip"] is False


def test_expected_plan_path_sits_next_to_clip():
    assert expected_plan_path("data/real/a b.mp4") == Path("data/real/a b.expected_plan.json")


def test_summary_text_survives_perception_json_round_trip(recorded, tmp_path):
    from core_robot.perception.perceive import load_perception, summary_text

    named, calls, frames, n = recorded
    perception = {"clip": {"n_frames": n, "fps": 5.0, "size": [960, 540]}, "naming": {"objects": named},
                  "tracks": assign(named, calls, frames, n)["tracks"]}
    in_memory = summary_text(perception)
    assert "(frame 53)" in in_memory and "(frame 9)" not in in_memory  # string-sorted keys would end at "9"
    (tmp_path / "perception.json").write_text(json.dumps(perception, indent=1))
    assert summary_text(json.loads((tmp_path / "perception.json").read_text())) == in_memory
    assert summary_text(load_perception(tmp_path / "perception.json")) == in_memory


def _calls(boxes_by_phrase, n=5):
    return {ph: {"color": color, "frames": {f: {i: _rec(b) for i, b in enumerate(boxes)} for f in range(n)}}
            for ph, (color, boxes) in boxes_by_phrase.items()}


def test_block_must_be_smaller_than_smallest_assigned_bin():
    import numpy as np

    frame = {0: np.zeros((480, 640, 3), np.uint8)}  # masks absent: no colour check
    bin_box, big_red = (100, 100, 199, 199), (300, 100, 409, 209)  # 100x100 bin; 110x110 "block"
    calls = _calls({"blue bin": ("blue", [bin_box]), "red cube": ("red", [big_red])})
    r = assign(["blue_bin", "red_block"], calls, frame, 5)
    assert r["tracks"]["blue_bin"]["object_id"] == "0" and r["tracks"]["red_block"] is None
    assert "not smaller than the smallest assigned bin" in r["candidates"]["red_block"][0]["why_not"][-1]
    # without a bin in the clip, only the absolute ceiling applies
    assert assign(["red_block"], _calls({"red cube": ("red", [big_red])}), frame, 5)["tracks"]["red_block"]


STACK = Path(__file__).parent / "fixtures" / "synthetic_stack_blue_on_red"


@pytest.fixture(scope="module")
def stack():
    d = json.loads((STACK / "sam_video.json").read_text())
    calls = {ph: {"color": c["color"], "frames": parse_sam(c["text"])} for ph, c in d["calls"].items()}
    frames = {int(p.stem.split("_")[1]): iio.imread(p) for p in STACK.glob("frame_*.png")}
    return d["named"], calls, frames, d["n_frames"]


def test_recorded_stack_close_up_cubes_are_accepted(stack):
    named, calls, frames, n = stack
    r = assign(named, calls, frames, n)
    assert {k: (t["concept"], t["object_id"]) for k, t in r["tracks"].items()} == {
        "red_block": ("red cube", "0"), "blue_block": ("blue cube", "0")}
    assert all(t["coverage"] == 1.0 for t in r["tracks"].values())
    for c in calls.values():  # the close-up cubes cover more than the old fixed 4% limit
        box = c["frames"][0][0]["box"]
        assert _box_fraction(box, frames[0]) > 0.04


def test_track_covering_most_of_the_frame_is_still_rejected(stack):
    _, calls, frames, n = stack
    h, w = frames[0].shape[:2]
    side_w, side_h = int(w * 0.6 ** 0.5), int(h * 0.6 ** 0.5)  # 0.6 of the frame, frame-shaped
    huge = (10, 10, 10 + side_w - 1, 10 + side_h - 1)
    assert abs(_box_fraction(huge, frames[0]) - 0.6) < 0.01
    big = {"red cube": {"color": "red", "frames": {f: {0: _rec(huge)} for f in range(n)}}}
    r = assign(["red_block"], big, frames, n)
    assert r["tracks"]["red_block"] is None
    assert "of the frame" in r["candidates"]["red_block"][0]["why_not"][0]


def _box_fraction(box, frame):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1) / (frame.shape[0] * frame.shape[1])
