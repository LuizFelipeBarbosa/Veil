"""WP20/WP22: ad replacement and its per-sign smoothing. Synthetic frames and ad images only, made in memory or in tmp_path."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import TrackedBox
from vision.replace import AdReplacer, list_assets

HEIGHT, WIDTH = config.FRAME_HEIGHT, config.FRAME_WIDTH
AD_COLOR = (0, 200, 255)  # BGR; nothing in the synthetic scenes has it
ASSETS = {"poster": "ad.png", "billboard": "ad.png"}


def _box(
    xyxy: tuple[int, int, int, int], track_id: str = "obj#1", label: str = "poster", last_seen: float = 0.0
) -> TrackedBox:
    return TrackedBox(id=track_id, label=label, xyxy=xyxy, conf=0.6, frame_index=0, last_seen=last_seen)


@pytest.fixture
def ads_dir(tmp_path: Path) -> Path:
    cv2.imwrite(str(tmp_path / "ad.png"), np.full((150, 200, 3), AD_COLOR, dtype=np.uint8))
    return tmp_path


def _wall() -> np.ndarray:
    """A grey wall with a little texture, so edges come only from the sign."""
    rng = np.random.default_rng(0)
    return (110 + rng.integers(0, 6, (HEIGHT, WIDTH, 3))).astype(np.uint8)


def _is_ad(pixels: np.ndarray) -> np.ndarray:
    return np.all(np.abs(pixels.astype(int) - AD_COLOR) <= 2, axis=-1)


# --- list_assets ---


def test_list_assets_returns_sorted_image_names_only(tmp_path: Path) -> None:
    for name in ("b.jpg", "a.png", "c.JPEG", "notes.txt", ".gitkeep"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "folder.png").mkdir()
    assert list_assets(tmp_path) == ("a.png", "b.jpg", "c.JPEG")
    assert list_assets(tmp_path / "missing") == ()


# --- drawing ---


SKEWED_SIGN = np.int32([[220, 130], [430, 160], [420, 330], [210, 300]])
SKEWED_SIGN_BOX = (200, 120, 440, 340)


def _wall_with_skewed_sign() -> np.ndarray:
    image = _wall()
    cv2.fillConvexPoly(image, SKEWED_SIGN, (240, 240, 240))
    cv2.polylines(image, [SKEWED_SIGN], True, (20, 20, 20), 3)
    return image


def _assert_ad_on_skewed_sign_only(out: np.ndarray) -> None:
    sign_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(sign_mask, SKEWED_SIGN, 255)
    inner = cv2.erode(sign_mask, np.ones((9, 9), np.uint8)) > 0
    outer = cv2.dilate(sign_mask, np.ones((9, 9), np.uint8)) > 0
    ad = _is_ad(out)
    assert ad[inner].mean() > 0.98  # the sign is covered
    x1, y1, x2, y2 = SKEWED_SIGN_BOX
    in_box_off_sign = np.zeros((HEIGHT, WIDTH), bool)
    in_box_off_sign[y1:y2, x1:x2] = True
    in_box_off_sign &= ~outer
    assert not ad[in_box_off_sign].any()  # the wall around it inside the box is not: the quad was used


def test_the_asset_is_warped_onto_a_skewed_sign_inside_its_box(ads_dir: Path) -> None:
    out = AdReplacer(ads_dir).draw(_wall_with_skewed_sign(), [_box(SKEWED_SIGN_BOX)], ASSETS, now=0.0)
    _assert_ad_on_skewed_sign_only(out)


def test_the_sign_keeps_its_shape_when_the_quad_search_fails(ads_dir: Path) -> None:
    replacer = AdReplacer(ads_dir)
    replacer.draw(_wall_with_skewed_sign(), [_box(SKEWED_SIGN_BOX)], ASSETS, now=0.0)
    # Next frames: the outline is lost (motion blur, a passer-by), the box is a fresh detection then a held-over one.
    out = replacer.draw(_wall(), [_box(SKEWED_SIGN_BOX, last_seen=0.07)], ASSETS, now=0.07)
    _assert_ad_on_skewed_sign_only(out)  # still the quad, not the box rectangle
    out = replacer.draw(_wall(), [_box(SKEWED_SIGN_BOX, last_seen=0.07)], ASSETS, now=0.14)
    _assert_ad_on_skewed_sign_only(out)


def test_overlapping_boxes_on_one_sign_are_drawn_once(ads_dir: Path) -> None:
    other_color = (255, 0, 255)
    cv2.imwrite(str(ads_dir / "other.png"), np.full((150, 200, 3), other_color, dtype=np.uint8))
    boxes = [
        _box((100, 50, 300, 200), "obj#1", "poster", last_seen=0.0),
        _box((104, 46, 296, 205), "obj#2", "billboard", last_seen=0.1),  # the fresher one wins
    ]
    out = AdReplacer(ads_dir).draw(_wall(), boxes, {"poster": "ad.png", "billboard": "other.png"}, now=0.1)
    assert not _is_ad(out).any()
    assert np.all(out[60:190, 110:290] == other_color)


def test_without_a_sign_outline_the_whole_box_is_filled(ads_dir: Path) -> None:
    image = _wall()
    box = (100, 50, 300, 200)
    out = AdReplacer(ads_dir).draw(image, [_box(box)], ASSETS, now=0.0)
    x1, y1, x2, y2 = box
    assert _is_ad(out[y1:y2, x1:x2]).mean() > 0.99


def test_pixels_outside_the_boxes_are_unchanged(ads_dir: Path) -> None:
    image = _wall()
    boxes = [_box((100, 50, 300, 200), "obj#1"), _box((400, 300, 600, 450), "obj#2", "billboard")]
    out = AdReplacer(ads_dir).draw(image, boxes, ASSETS, now=0.0)
    outside = np.ones((HEIGHT, WIDTH), bool)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        outside[y1:y2, x1:x2] = False
    assert np.array_equal(out[outside], image[outside])
    assert not np.array_equal(out, image)


def test_the_input_image_is_not_mutated(ads_dir: Path) -> None:
    image = _wall()
    before = image.copy()
    out = AdReplacer(ads_dir).draw(image, [_box((100, 50, 300, 200))], ASSETS, now=0.0)
    assert np.array_equal(image, before)
    assert out is not image and not np.shares_memory(out, image)


def test_a_missing_or_unsafe_asset_leaves_the_frame_unchanged_and_logs_once(
    ads_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = _wall()
    replacer = AdReplacer(ads_dir)
    (ads_dir / "broken.png").write_bytes(b"not an image")
    with caplog.at_level(logging.WARNING, logger="vision.replace"):
        for name in ("gone.png", "broken.png", "../ad.png", ""):
            for now in (0.0, 0.1):
                out = replacer.draw(image, [_box((100, 50, 300, 200))], {"poster": name}, now=now)
                assert np.array_equal(out, image)
        out = replacer.draw(image, [_box((100, 50, 300, 200), label="billboard")], {"poster": "ad.png"}, now=0.2)
        assert np.array_equal(out, image)  # no asset named for this label
    assert len(caplog.records) == 4  # once per bad name; the unnamed label is the "" one


def test_a_box_cut_by_the_frame_edge_is_clipped(ads_dir: Path) -> None:
    image = _wall()
    out = AdReplacer(ads_dir).draw(image, [_box((-40, -30, 120, 90))], ASSETS, now=0.0)
    assert _is_ad(out[0:90, 0:120]).mean() > 0.99
    assert np.array_equal(out[90:], image[90:])


def _ad_column(out: np.ndarray, row: int) -> float:
    """Where the asset's left-to-right gradient crosses its middle value on this row."""
    values = out[row, :, 1].astype(int)
    return float(np.argmin(np.abs(values - 128)))


@pytest.mark.parametrize("same_track", [True, False], ids=["same track", "new track id and label every frame"])
def test_smoothing_damps_a_jittering_box(tmp_path: Path, same_track: bool) -> None:
    gradient = np.zeros((100, 256, 3), np.uint8)
    gradient[:, :, 1] = np.arange(256, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ramp.png"), gradient)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: the box itself is the quad
    replacer = AdReplacer(tmp_path)
    jitter = 12
    columns = []
    for step in range(10):
        dx = jitter if step % 2 else -jitter
        track_id, label = ("obj#1", "poster") if same_track else (f"obj#{step}", ("poster", "billboard")[step % 2])
        box = _box((200 + dx, 100, 456 + dx, 300), track_id, label, last_seen=step * 0.07)
        out = replacer.draw(image, [box], {"poster": "ramp.png", "billboard": "ramp.png"}, now=step * 0.07)
        columns.append(_ad_column(out, 200))
    # Unsmoothed, or reset on every new id, the middle of the ramp would move 2 * jitter = 24 px every frame.
    assert max(abs(a - b) for a, b in zip(columns[3:], columns[4:])) < jitter


def test_a_track_not_seen_for_longer_than_the_hold_over_starts_fresh(ads_dir: Path) -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    replacer = AdReplacer(ads_dir)
    replacer.draw(image, [_box((0, 0, 100, 100))], ASSETS, now=0.0)
    replacer.draw(image, [], ASSETS, now=config.HOLD_OVER_S + 0.1)
    out = replacer.draw(image, [_box((300, 300, 400, 400))], ASSETS, now=config.HOLD_OVER_S + 0.2)
    assert _is_ad(out[300:400, 300:400]).all()  # not dragged from the old position


def test_a_steadily_moving_sign_is_followed_without_growing_lag(ads_dir: Path) -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: the box itself is the measurement
    replacer = AdReplacer(ads_dir)
    speed = 6  # px per frame at 15 FPS, a steady pan
    lags = []
    for step in range(40):
        now = step / 15
        x1 = 50 + speed * step
        out = replacer.draw(image, [_box((x1, 100, x1 + 200, 250), last_seen=now)], ASSETS, now=now)
        covered = np.flatnonzero(_is_ad(out[175]))
        lags.append(x1 + 199 - covered[-1])  # how far the ad's right edge trails the sign's
    assert max(lags[20:]) <= 3  # caught up: a constant-velocity filter has no steady lag
    assert lags[-1] <= max(lags[5:15])  # and the lag does not grow


def test_a_held_over_box_keeps_its_ad_drawn(ads_dir: Path) -> None:
    replacer = AdReplacer(ads_dir)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    replacer.draw(image, [_box((100, 50, 300, 200), last_seen=0.0)], ASSETS, now=0.0)
    for now in (0.07, 0.14, 0.21, 0.28):  # the detector lost it; the pipeline holds the last box over
        out = replacer.draw(image, [_box((100, 50, 300, 200), last_seen=0.0)], ASSETS, now=now)
        assert _is_ad(out[50:200, 100:300]).mean() > 0.99


def test_draw_is_cheap(ads_dir: Path) -> None:
    image = _wall()
    cv2.rectangle(image, (200, 120), (440, 340), (240, 240, 240), -1)
    replacer = AdReplacer(ads_dir)
    box = [_box((180, 100, 460, 360))]
    replacer.draw(image, box, ASSETS, now=0.0)  # loads the asset
    started = time.perf_counter()
    for i in range(20):
        replacer.draw(image, box, ASSETS, now=i * 0.07)
    per_draw_ms = (time.perf_counter() - started) / 20 * 1000
    assert per_draw_ms < 20  # budget ~5 ms per box on the Mac; generous for a loaded test machine


# --- WP23: scene cuts, jumps, things in front of the sign ---


def _shifted_skewed_sign(dx: int) -> np.ndarray:
    image = _wall()
    sign = SKEWED_SIGN + np.int32([dx, 0])
    cv2.fillConvexPoly(image, sign, (240, 240, 240))
    cv2.polylines(image, [sign], True, (20, 20, 20), 3)
    return image


def test_a_scene_cut_starts_fresh_and_drops_held_over_boxes(ads_dir: Path) -> None:
    replacer = AdReplacer(ads_dir)
    old_box = _box((470, 350, 630, 470), "obj#9", last_seen=0.07)
    for step in range(3):  # the skewed sign's shape is learned; another sign elsewhere
        now = step * 0.07
        replacer.draw(_wall_with_skewed_sign(), [_box(SKEWED_SIGN_BOX, last_seen=now), old_box], ASSETS, now=now)
    cut = np.full((HEIGHT, WIDTH, 3), 230, np.uint8)  # a very different frame, no outlines
    boxes = [_box(SKEWED_SIGN_BOX, "obj#2", last_seen=0.21), old_box]  # same place, fresh; and a held-over box
    out = replacer.draw(cut, boxes, ASSETS, now=0.21)
    x1, y1, x2, y2 = SKEWED_SIGN_BOX
    assert _is_ad(out[y1:y2, x1:x2]).mean() > 0.99  # the whole box: not the old scene's skewed shape
    assert not _is_ad(out[350:470, 470:630]).any()  # the held-over box from before the cut is not drawn
    out = replacer.draw(cut, boxes, ASSETS, now=0.28)
    assert not _is_ad(out[350:470, 470:630]).any()  # nor later


def test_a_box_that_jumps_restarts_at_its_new_place(ads_dir: Path) -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: the box itself is the measurement
    replacer = AdReplacer(ads_dir)
    for step in range(5):
        replacer.draw(image, [_box((100, 50, 300, 200), last_seen=step * 0.07)], ASSETS, now=step * 0.07)
    out = replacer.draw(image, [_box((170, 50, 370, 200), last_seen=0.35)], ASSETS, now=0.35)
    assert _is_ad(out[50:200, 170:370]).mean() > 0.99  # no sliding in from the old place


def test_a_sign_whose_outline_moves_far_restarts_within_a_few_frames(ads_dir: Path) -> None:
    replacer = AdReplacer(ads_dir)
    for step in range(3):
        now = step * 0.07
        replacer.draw(_wall_with_skewed_sign(), [_box(SKEWED_SIGN_BOX, last_seen=now)], ASSETS, now=now)
    dx = 50  # beyond the 12 % jump the filter ignores as another outline
    x1, y1, x2, y2 = SKEWED_SIGN_BOX
    for step in range(3, 6):
        now = step * 0.07
        out = replacer.draw(_shifted_skewed_sign(dx), [_box((x1 + dx, y1, x2 + dx, y2), last_seen=now)], ASSETS, now=now)
    sign_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(sign_mask, SKEWED_SIGN + np.int32([dx, 0]), 255)
    inner = cv2.erode(sign_mask, np.ones((9, 9), np.uint8)) > 0
    assert _is_ad(out)[inner].mean() > 0.98


def _luminance(pixels: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).ravel().astype(float)


def test_a_dark_post_in_front_of_the_sign_stays_visible_through_the_ad(ads_dir: Path) -> None:
    image = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
    image[50:200, 190:205] = 30  # a dark post across a bright sign
    out = AdReplacer(ads_dir).draw(image, [_box((100, 50, 300, 200))], ASSETS, now=0.0)
    post = _luminance(out[60:190, 192:203]).mean()
    ad = _luminance(out[60:190, 110:170]).mean()
    assert _is_ad(out[60:190, 110:170]).mean() > 0.99
    assert post < 0.5 * ad


def test_a_flat_sign_under_uneven_light_shows_the_ad_colours(ads_dir: Path) -> None:
    image = _wall()
    image[50:200, 100:300] = np.linspace(170, 240, 200, dtype=np.uint8)[None, :, None]  # lit from the right
    out = AdReplacer(ads_dir).draw(image, [_box((100, 50, 300, 200))], ASSETS, now=0.0)
    assert _is_ad(out[50:200, 100:300]).mean() > 0.98
