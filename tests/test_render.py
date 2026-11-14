"""WP2: padding, pixelation, hold-over, backfill, watchdog and censor(). Synthetic frames only."""

from __future__ import annotations

import numpy as np
import pytest

import config
from contracts import FACE, BufferedFrame, Policy, PolicySnapshot, TrackedBox
from vision.render import HoldOver, Watchdog, backfill, censor, pad_box, pixelate_boxes, pixelate_full

WIDTH, HEIGHT = config.FRAME_WIDTH, config.FRAME_HEIGHT


def noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def is_pixelated(image: np.ndarray) -> bool:
    """Every 16x16 block is one flat color."""
    block = config.PIXEL_BLOCK
    height, width = image.shape[:2]
    blocks = image[: height - height % block, : width - width % block].reshape(
        height // block, block, width // block, block, 3
    )
    return bool((blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))).all())


def box(track_id: str, xyxy: tuple[int, int, int, int] = (100, 100, 180, 180), *, frame_index: int = 0,
        last_seen: float = 0.0) -> TrackedBox:
    return TrackedBox(id=track_id, label=FACE, xyxy=xyxy, conf=0.9, frame_index=frame_index, last_seen=last_seen)


def always_blur(_box: TrackedBox, _policy: PolicySnapshot) -> str:
    return "blur"


def show_only(track_id: str):
    return lambda b, _policy: "show" if b.id == track_id else "blur"


def calm_policy() -> PolicySnapshot:
    return Policy().snapshot()


def panic_policy() -> PolicySnapshot:
    policy = Policy()
    policy.set_panic(True)
    return policy.snapshot()


# --- padding and pixelation ---


def test_pad_box_grows_20_percent_per_side() -> None:
    assert pad_box((100, 100, 200, 150), WIDTH, HEIGHT) == (80, 90, 220, 160)


def test_pad_box_is_clipped_to_the_frame() -> None:
    assert pad_box((0, 5, 100, 100), WIDTH, HEIGHT) == (0, 0, 120, 119)
    assert pad_box((600, 440, 640, 480), WIDTH, HEIGHT) == (592, 432, 640, 480)


def test_pixelate_full_is_flat_blocks_and_a_copy() -> None:
    raw = noise()
    before = raw.copy()
    pixelated = pixelate_full(raw)
    assert pixelated.shape == raw.shape and pixelated is not raw
    assert is_pixelated(pixelated)
    np.testing.assert_array_equal(raw, before)


def test_pixelate_full_handles_sizes_that_are_not_whole_blocks() -> None:
    raw = np.random.default_rng(1).integers(0, 255, (50, 70, 3), dtype=np.uint8)
    pixelated = pixelate_full(raw)
    assert pixelated.shape == raw.shape
    assert is_pixelated(pixelated)  # the whole blocks; the ragged edge is flat too
    assert (pixelated[48:, 64:] == pixelated[48, 64]).all()


def test_pixelate_boxes_touches_only_the_boxes() -> None:
    raw = noise()
    before = raw.copy()
    out = pixelate_boxes(raw, [(96, 96, 192, 208)])
    np.testing.assert_array_equal(raw, before)
    assert is_pixelated(out[96:208, 96:192])
    outside = np.ones(raw.shape[:2], dtype=bool)
    outside[96:208, 96:192] = False
    np.testing.assert_array_equal(out[outside], raw[outside])


def test_pixelate_boxes_with_no_boxes_is_an_unchanged_copy() -> None:
    raw = noise()
    out = pixelate_boxes(raw, [])
    assert out is not raw
    np.testing.assert_array_equal(out, raw)


# --- censor ---


@pytest.mark.parametrize(
    "faces_done, tripped, policy",
    [(False, False, calm_policy), (True, True, calm_policy), (True, False, panic_policy)],
    ids=["faces-not-done", "watchdog-tripped", "panic"],
)
def test_censor_fails_closed(faces_done: bool, tripped: bool, policy) -> None:
    raw = noise()
    before = raw.copy()
    frame = BufferedFrame(index=7, timestamp=0.0, image=raw, boxes=[], faces_done=faces_done)
    censored = censor(frame, policy(), show_only("anything"), watchdog_tripped=tripped)
    assert censored.fail_closed and censored.index == 7
    assert is_pixelated(censored.image)
    np.testing.assert_array_equal(raw, before)


def test_censor_blurs_padded_boxes_and_leaves_shown_ones_untouched() -> None:
    raw = noise()
    blurred_box, shown_box = (96, 96, 176, 176), (400, 200, 480, 280)
    frame = BufferedFrame(
        index=3, timestamp=0.0, image=raw, faces_done=True,
        boxes=[box("face#1", blurred_box), box("face#2", shown_box)],
    )
    censored = censor(frame, calm_policy(), show_only("face#2"), watchdog_tripped=False)

    assert not censored.fail_closed and censored.index == 3
    x1, y1, x2, y2 = pad_box(blurred_box, WIDTH, HEIGHT)
    assert (x1, y1, x2, y2) == (80, 80, 192, 192)
    assert is_pixelated(censored.image[y1:y2, x1:x2])
    np.testing.assert_array_equal(censored.image[y2:, :], raw[y2:, :])  # incl. the shown box
    sx1, sy1, sx2, sy2 = pad_box(shown_box, WIDTH, HEIGHT)
    np.testing.assert_array_equal(censored.image[sy1:sy2, sx1:sx2], raw[sy1:sy2, sx1:sx2])


def test_censor_with_no_boxes_passes_the_frame_through() -> None:
    raw = noise()
    frame = BufferedFrame(index=0, timestamp=0.0, image=raw, faces_done=True)
    censored = censor(frame, calm_policy(), always_blur, watchdog_tripped=False)
    assert not censored.fail_closed and censored.image is not raw
    np.testing.assert_array_equal(censored.image, raw)


# --- hold-over ---


def test_hold_over_keeps_a_lost_track_for_half_a_second_then_drops_it() -> None:
    hold = HoldOver()
    assert [b.id for b in hold.apply([box("face#1", last_seen=10.0)], now=10.0)] == ["face#1"]
    held = hold.apply([], now=10.3, frame_index=5)
    assert [b.id for b in held] == ["face#1"] and held[0].frame_index == 5
    assert [b.id for b in hold.apply([], now=10.5)] == ["face#1"]
    assert hold.apply([], now=10.51) == []
    assert hold.apply([], now=10.4) == []  # dropped for good


def test_hold_over_does_not_duplicate_a_visible_track() -> None:
    hold = HoldOver()
    hold.apply([box("face#1", last_seen=1.0), box("face#2", last_seen=1.0)], now=1.0)
    out = hold.apply([box("face#1", (110, 100, 190, 180), last_seen=1.1)], now=1.1)
    assert sorted(b.id for b in out) == ["face#1", "face#2"]
    assert out[0].xyxy == (110, 100, 190, 180)


# --- backfill ---


def test_backfill_reaches_exactly_the_frames_given() -> None:
    frames = [BufferedFrame(index=i, timestamp=i / 15, image=noise(), faces_done=True) for i in range(15)]
    new = box("face#9", frame_index=14)
    backfill(frames[4:14], [new])
    for frame in frames:
        ids = [b.id for b in frame.boxes]
        if 4 <= frame.index < 14:
            assert ids == ["face#9"] and frame.boxes[0].frame_index == frame.index
            assert frame.boxes[0].xyxy == new.xyxy
        else:
            assert ids == []
    assert new.frame_index == 14  # the original is not mutated


def test_backfill_does_not_duplicate_a_track_already_on_the_frame() -> None:
    frame = BufferedFrame(index=0, timestamp=0.0, image=noise(), boxes=[box("face#1")])
    backfill([frame], [box("face#1", frame_index=3), box("face#2", frame_index=3)])
    assert [b.id for b in frame.boxes] == ["face#1", "face#2"]


# --- watchdog ---


def test_watchdog_is_tripped_before_the_first_beat() -> None:
    assert Watchdog().tripped(now=0.0)
    assert Watchdog(watch_objects=False).tripped(now=0.0)


def test_watchdog_faces() -> None:
    dog = Watchdog(watch_objects=False)
    dog.beat_faces(now=1.0)
    assert not dog.tripped(now=1.29)
    assert dog.tripped(now=1.31)


def test_watchdog_objects() -> None:
    dog = Watchdog()
    dog.beat_faces(now=1.0)
    assert dog.tripped(now=1.0)  # no object beat yet
    dog.beat_objects(now=1.0)
    assert not dog.tripped(now=1.0)
    dog.beat_faces(now=2.5)
    assert not dog.tripped(now=2.49)
    assert dog.tripped(now=2.51)


def test_watchdog_can_ignore_objects() -> None:
    dog = Watchdog(watch_objects=False)
    dog.beat_faces(now=100.0)
    assert not dog.tripped(now=100.0)
