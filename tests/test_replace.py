"""WP20: ad replacement. Synthetic frames and ad images only, made in memory or in tmp_path."""

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


def _box(xyxy: tuple[int, int, int, int], track_id: str = "obj#1", label: str = "poster") -> TrackedBox:
    return TrackedBox(id=track_id, label=label, xyxy=xyxy, conf=0.6, frame_index=0, last_seen=0.0)


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


def test_the_asset_is_warped_onto_a_skewed_sign_inside_its_box(ads_dir: Path) -> None:
    image = _wall()
    sign = np.int32([[220, 130], [430, 160], [420, 330], [210, 300]])
    cv2.fillConvexPoly(image, sign, (240, 240, 240))
    cv2.polylines(image, [sign], True, (20, 20, 20), 3)
    box = (200, 120, 440, 340)

    out = AdReplacer(ads_dir).draw(image, [_box(box)], ASSETS, now=0.0)

    sign_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(sign_mask, sign, 255)
    inner = cv2.erode(sign_mask, np.ones((9, 9), np.uint8)) > 0
    outer = cv2.dilate(sign_mask, np.ones((9, 9), np.uint8)) > 0
    ad = _is_ad(out)
    assert ad[inner].mean() > 0.98  # the sign is covered
    x1, y1, x2, y2 = box
    in_box_off_sign = np.zeros((HEIGHT, WIDTH), bool)
    in_box_off_sign[y1:y2, x1:x2] = True
    in_box_off_sign &= ~outer
    assert not ad[in_box_off_sign].any()  # the wall around it inside the box is not: the quad was found


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


def test_smoothing_damps_a_jittering_box(tmp_path: Path) -> None:
    gradient = np.zeros((100, 256, 3), np.uint8)
    gradient[:, :, 1] = np.arange(256, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ramp.png"), gradient)
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: the box itself is the quad
    replacer = AdReplacer(tmp_path)
    jitter = 12
    columns = []
    for step in range(10):
        dx = jitter if step % 2 else -jitter
        out = replacer.draw(image, [_box((200 + dx, 100, 456 + dx, 300))], {"poster": "ramp.png"}, now=step * 0.07)
        columns.append(_ad_column(out, 200))
    # Unsmoothed, the middle of the ramp would move 2 * jitter = 24 px every frame.
    assert max(abs(a - b) for a, b in zip(columns[3:], columns[4:])) < jitter


def test_a_track_not_seen_for_longer_than_the_hold_over_starts_fresh(ads_dir: Path) -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    replacer = AdReplacer(ads_dir)
    replacer.draw(image, [_box((0, 0, 100, 100))], ASSETS, now=0.0)
    replacer.draw(image, [], ASSETS, now=config.HOLD_OVER_S + 0.1)
    out = replacer.draw(image, [_box((300, 300, 400, 400))], ASSETS, now=config.HOLD_OVER_S + 0.2)
    assert _is_ad(out[300:400, 300:400]).all()  # not dragged from the old position


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
