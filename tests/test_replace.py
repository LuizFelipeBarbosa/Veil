"""WP20/WP22/WP25: ad replacement, one drawing per sign, signs followed by optical flow. Synthetic frames and ad images only, made in memory or in tmp_path."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
from contracts import TrackedBox
from vision import replace
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


# The look (WP31) changes every ad pixel: tone, grain. The geometry tests below tell the ad by its exact
# colour, so they run with the look neutral; the look has its own tests at the end, which restore it.
SCENE_LOOK = {name: getattr(replace, name) for name in ("_MAX_BLACK", "_MIN_WHITE", "_MAX_WHITE", "_GRAIN_SIGMA")}
NEUTRAL_LOOK = {"_MAX_BLACK": 0.0, "_MIN_WHITE": 255.0, "_MAX_WHITE": 255.0, "_GRAIN_SIGMA": 0.0}


@pytest.fixture(autouse=True)
def neutral_look(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in NEUTRAL_LOOK.items():
        monkeypatch.setattr(replace, name, value)


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
    now = config.HOLD_OVER_S + 0.2
    out = replacer.draw(image, [_box((300, 300, 400, 400), last_seen=now)], ASSETS, now=now)
    assert _is_ad(out[300:400, 300:400]).all()  # not dragged from the old position


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


def test_a_box_that_jumps_takes_the_sign_to_its_new_place_in_a_glide_not_a_jump(ads_dir: Path) -> None:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: the box itself is the measurement
    replacer = AdReplacer(ads_dir)
    for step in range(5):
        replacer.draw(image, [_box((100, 50, 300, 200), last_seen=step * 0.07)], ASSETS, now=step * 0.07)
    first = replacer.draw(image, [_box((170, 50, 370, 200), last_seen=0.35)], ASSETS, now=0.35)
    assert 0.2 < _is_ad(first[50:200, 300:370]).mean() < 0.5  # WP31: a third of the way, no 70 px jump
    for step in range(6, 18):  # there within a dozen frames, not pulled at the slow box gain
        out = replacer.draw(image, [_box((170, 50, 370, 200), last_seen=step * 0.07)], ASSETS, now=step * 0.07)
    assert _is_ad(out[50:200, 174:370]).mean() > 0.99
    assert not _is_ad(out[50:200, 100:168]).any()


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


# --- WP25: no ad on an ad; signs followed by optical flow ---


@pytest.fixture
def drawn_quads(monkeypatch: pytest.MonkeyPatch) -> list[list[np.ndarray]]:
    """The corners of every quad drawn, per draw() call (a new list is started by the test)."""
    calls: list[list[np.ndarray]] = []
    composite = replace._composite

    def spy(out, image, asset, sign, region, grain):
        if calls:
            calls[-1].append(np.array(sign.corners, float))
        return composite(out, image, asset, sign, region, grain)

    monkeypatch.setattr(replace, "_composite", spy)
    return calls


def _overlapping(quads: list[np.ndarray]) -> bool:
    return any(
        cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))[0] > 0
        for i, a in enumerate(quads) for b in quads[i + 1:]
    )


@pytest.mark.parametrize(
    "boxes",
    [
        [(100, 50, 400, 300), (180, 120, 280, 220)],  # a poster nested in a billboard
        [(100, 50, 300, 200), (220, 60, 400, 210)],  # two partly overlapping boxes on one panel
    ],
    ids=["nested", "partly overlapping"],
)
def test_overlapping_boxes_are_one_drawing(ads_dir: Path, drawn_quads, boxes) -> None:
    replacer = AdReplacer(ads_dir)
    for step in range(3):
        drawn_quads.append([])
        tracked = [_box(xyxy, f"obj#{k}", ("poster", "billboard")[k], last_seen=step * 0.07) for k, xyxy in enumerate(boxes)]
        replacer.draw(_wall(), tracked, ASSETS, now=step * 0.07)
        assert len(drawn_quads[-1]) == 1


def test_two_signs_whose_quads_would_overlap_draw_only_the_established_one(ads_dir: Path, drawn_quads) -> None:
    replacer = AdReplacer(ads_dir)
    first = (100, 50, 300, 200)
    drawn_quads.append([])
    replacer.draw(_wall(), [_box(first, "obj#1")], ASSETS, now=0.0)
    for step in range(1, 4):  # a second box touches the first only a little: its own sign, but its quad overlaps
        now = step * 0.07
        drawn_quads.append([])
        boxes = [_box(first, "obj#1", last_seen=now), _box((280, 60, 450, 210), "obj#2", "billboard", last_seen=now)]
        out = replacer.draw(_wall(), boxes, ASSETS, now=now)
        assert len(drawn_quads[-1]) == 1 and not _overlapping(drawn_quads[-1])
        assert _is_ad(out[60:190, 110:290]).mean() > 0.99  # the established one


def _texture(seed: int = 1) -> np.ndarray:
    """A large blotchy scene with plenty of corners to follow and no long straight outlines."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (1200, 1600)).astype(np.float32)
    blotchy = cv2.GaussianBlur(noise, (0, 0), 3)
    blotchy = (blotchy - blotchy.mean()) * 6 + 128
    return cv2.cvtColor(np.clip(blotchy, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _camera(step: int) -> np.ndarray:
    """Scene -> frame for a camera panning 4 px and zooming in 0.4 % a frame."""
    scale = 1.0 + 0.004 * step
    return np.float32([[scale, 0, -300 - 4 * step], [0, scale, -300 - 1 * step]])


SCENE_SIGN = np.float32([[500, 400], [700, 400], [700, 550], [500, 550]])  # in scene px


def test_drawn_corners_follow_the_scene_not_the_jittering_box(ads_dir: Path, drawn_quads) -> None:
    scene = _texture()
    replacer = AdReplacer(ads_dir)
    errors, corners = [], []
    for step in range(25):
        transform = _camera(step)
        frame = cv2.warpAffine(scene, transform, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
        true = cv2.transform(SCENE_SIGN[None], transform)[0]
        x1, y1 = true.min(axis=0)
        x2, y2 = true.max(axis=0)
        jitter = 0 if step == 0 else (12 if step % 2 else -12)  # the detector's box jumps 24 px a frame
        box = (round(x1) + jitter, round(y1) - jitter, round(x2) + jitter, round(y2) + jitter)
        drawn_quads.append([])
        replacer.draw(frame, [_box(box, last_seen=step / 15)], ASSETS, now=step / 15)
        assert len(drawn_quads[-1]) == 1
        corners.append(drawn_quads[-1][0])
        errors.append(np.abs(drawn_quads[-1][0] - true).max())
    assert max(errors) <= 1.5
    second_difference = np.abs(np.diff(np.stack(corners), n=2, axis=0)).max()
    assert second_difference < 1.0  # the box's 24 px jumps do not reach the drawing


def test_lost_flow_restarts_the_sign_at_its_fresh_box(ads_dir: Path, drawn_quads) -> None:
    frame = cv2.warpAffine(_texture(), _camera(0), (WIDTH, HEIGHT))
    replacer = AdReplacer(ads_dir)
    replacer.draw(frame, [_box((100, 50, 300, 200), last_seen=0.0)], ASSETS, now=0.0)
    changed = frame.copy()
    changed[30:240, 80:340] = cv2.warpAffine(_texture(seed=2), _camera(0), (WIDTH, HEIGHT))[30:240, 80:340]
    drawn_quads.append([])
    for step in range(1, 14):  # WP31: it glides to the box over a few frames instead of jumping
        replacer.draw(changed, [_box((120, 50, 320, 200), last_seen=step * 0.07)], ASSETS, now=step * 0.07)
    assert np.abs(drawn_quads[-1][-1] - [[120, 50], [320, 50], [320, 200], [120, 200]]).max() < 1  # not pulled slowly


def test_a_sign_without_a_fresh_box_for_the_hold_over_is_dropped(ads_dir: Path) -> None:
    image = _wall()
    replacer = AdReplacer(ads_dir)
    held = _box((100, 50, 300, 200), last_seen=0.0)  # the pipeline keeps sending the last box
    for now in (0.0, 0.2, 0.4, config.HOLD_OVER_S):
        assert _is_ad(replacer.draw(image, [held], ASSETS, now=now)[50:200, 100:300]).mean() > 0.99
    for now in (config.HOLD_OVER_S + 0.05, config.HOLD_OVER_S + 0.1):  # dropped, and not restarted from that box
        assert not _is_ad(replacer.draw(image, [held], ASSETS, now=now)).any()


# --- WP27: signs seen together show different images ---


COLOURS = {"a.png": (0, 0, 255), "b.png": (0, 255, 0), "c.png": (255, 0, 0), "d.png": (255, 255, 0)}  # BGR, flat
SIGN_BOXES = [(40, 40, 200, 160), (300, 40, 460, 160), (40, 300, 200, 420)]  # far apart: separate signs


@pytest.fixture
def colour_dir(tmp_path: Path) -> Path:
    for name, colour in COLOURS.items():
        cv2.imwrite(str(tmp_path / name), np.full((150, 200, 3), colour, dtype=np.uint8))
    return tmp_path


def _shown(out: np.ndarray, xyxy: tuple[int, int, int, int]) -> str | None:
    """The image drawn on the box, by the colour at its centre; None when nothing is."""
    pixel = out[(xyxy[1] + xyxy[3]) // 2, (xyxy[0] + xyxy[2]) // 2]
    return next((name for name, colour in COLOURS.items() if np.abs(pixel.astype(int) - colour).max() <= 2), None)


def _draw_signs(replacer: AdReplacer, signs: list[tuple[int, int, int, int]], chosen: str, step: int) -> list[str | None]:
    """One frame with a box on each sign, jittering a few px, with a new track id and label every frame."""
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # no edges: each box is its sign's quad
    jitter = 6 if step % 2 else -6
    boxes = [
        _box((x1 + jitter, y1 - jitter, x2 + jitter, y2 + jitter), f"obj#{step}-{k}",
             ("poster", "billboard")[(step + k) % 2], last_seen=step * 0.07)
        for k, (x1, y1, x2, y2) in enumerate(signs)
    ]
    out = replacer.draw(image, boxes, {"poster": chosen, "billboard": chosen}, now=step * 0.07)
    return [_shown(out, xyxy) for xyxy in signs]


def test_two_signs_together_show_different_images_the_chosen_one_first(colour_dir: Path) -> None:
    assert _draw_signs(AdReplacer(colour_dir), SIGN_BOXES[:2], "b.png", step=0) == ["b.png", "c.png"]


def test_each_sign_keeps_its_image_over_jitter_and_changing_tracks(colour_dir: Path) -> None:
    replacer = AdReplacer(colour_dir)
    for step in range(20):
        assert _draw_signs(replacer, SIGN_BOXES[:2], "a.png", step) == ["a.png", "b.png"]


def test_a_third_sign_takes_the_next_image_and_leaves_the_others(colour_dir: Path) -> None:
    replacer = AdReplacer(colour_dir)
    for step in range(3):
        _draw_signs(replacer, SIGN_BOXES[:2], "c.png", step)
    for step in range(3, 8):
        assert _draw_signs(replacer, SIGN_BOXES, "c.png", step) == ["c.png", "d.png", "a.png"]  # wraps around
    for step in range(8, 10):  # the first one leaves the view: the others keep theirs
        assert _draw_signs(replacer, SIGN_BOXES[1:], "c.png", step) == ["d.png", "a.png"]


def test_a_new_sign_takes_the_first_image_no_living_sign_shows(colour_dir: Path) -> None:
    replacer = AdReplacer(colour_dir)
    black = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    replacer.draw(black, [_box(SIGN_BOXES[0], "obj#1"), _box(SIGN_BOXES[1], "obj#2")], {"poster": "a.png"}, now=0.0)
    for now in (0.5 * config.HOLD_OVER_S, config.HOLD_OVER_S + 0.1):  # the first sign expires, the second is seen
        replacer.draw(black, [_box(SIGN_BOXES[1], "obj#2", last_seen=now)], {"poster": "a.png"}, now=now)
    boxes = [_box(SIGN_BOXES[1], "obj#2", last_seen=now + 0.07), _box(SIGN_BOXES[2], "obj#3", last_seen=now + 0.07)]
    out = replacer.draw(black, boxes, {"poster": "a.png"}, now=now + 0.07)
    assert [_shown(out, SIGN_BOXES[1]), _shown(out, SIGN_BOXES[2])] == ["b.png", "a.png"]


def test_changing_the_chosen_image_picks_again_with_the_new_one_on_the_established_sign(colour_dir: Path) -> None:
    replacer = AdReplacer(colour_dir)
    _draw_signs(replacer, SIGN_BOXES[:1], "a.png", step=0)  # the first sign is the established one
    for step in range(1, 4):
        assert _draw_signs(replacer, SIGN_BOXES[:2], "a.png", step) == ["a.png", "b.png"]
    for step in range(4, 8):
        assert _draw_signs(replacer, SIGN_BOXES[:2], "c.png", step) == ["c.png", "d.png"]


def test_with_one_image_every_sign_shows_it(ads_dir: Path) -> None:
    boxes = [_box(xyxy, f"obj#{k}") for k, xyxy in enumerate(SIGN_BOXES)]
    out = AdReplacer(ads_dir).draw(np.zeros((HEIGHT, WIDTH, 3), np.uint8), boxes, ASSETS, now=0.0)
    for x1, y1, x2, y2 in SIGN_BOXES:
        assert _is_ad(out[y1:y2, x1:x2]).mean() > 0.99


def test_an_unreadable_image_is_skipped_by_the_others_but_never_stands_in_for_the_chosen_one(colour_dir: Path) -> None:
    (colour_dir / "b2.png").write_bytes(b"not an image")  # sorted right after b.png
    assert _draw_signs(AdReplacer(colour_dir), SIGN_BOXES, "b.png", step=0) == ["b.png", "c.png", "d.png"]
    assert _draw_signs(AdReplacer(colour_dir), SIGN_BOXES, "b2.png", step=0) == [None, None, None]


def test_the_directory_is_rescanned_only_for_a_name_not_in_it(colour_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scans: list[Path] = []

    def counted(directory: Path) -> tuple[str, ...]:
        scans.append(directory)
        return list_assets(directory)

    monkeypatch.setattr(replace, "list_assets", counted)
    replacer = AdReplacer(colour_dir)
    for step in range(5):
        _draw_signs(replacer, SIGN_BOXES[:2], "a.png", step)
    assert len(scans) == 1  # at construction
    cv2.imwrite(str(colour_dir / "e.png"), np.full((150, 200, 3), 128, np.uint8))
    for step in range(5, 10):
        _draw_signs(replacer, SIGN_BOXES[:2], "e.png", step)
    assert len(scans) == 2  # once for the new name, not every frame


# --- the look (WP31): the ad is toned, grained and feathered into the scene ---


LOOK_BOX = (200, 100, 360, 380)  # a tall panel, no outline: the whole box is filled


@pytest.fixture
def scene_look(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in SCENE_LOOK.items():
        monkeypatch.setattr(replace, name, value)


def _wall_with_old_poster(paper: int = 200) -> np.ndarray:
    image = _wall()
    x1, y1, x2, y2 = LOOK_BOX
    image[y1:y2, x1:x2] = paper
    return image


def _draw_asset(directory: Path, asset: np.ndarray, image: np.ndarray, calls: int = 1) -> np.ndarray:
    cv2.imwrite(str(directory / "ad.png"), asset)
    replacer = AdReplacer(directory)
    for _ in range(calls):
        out = replacer.draw(image, [_box(LOOK_BOX)], ASSETS, now=0.0)
    return out


def test_a_white_ad_is_as_white_as_the_old_poster_not_whiter(tmp_path: Path, scene_look: None) -> None:
    image = _wall_with_old_poster(paper=200)
    out = _draw_asset(tmp_path, np.full((90, 160, 3), 255, np.uint8), image)
    inside = out[120:360, 220:340].astype(float)
    assert abs(inside.mean() - 200) < 3  # the paper's white, not 255
    assert (out[:100] == image[:100]).all() and (out[:, :200] == image[:, :200]).all()  # only inside the box


def test_a_black_ad_on_a_dark_poster_is_neither_crushed_nor_dimmed(tmp_path: Path, scene_look: None) -> None:
    asset = np.zeros((90, 160, 3), np.uint8)
    asset[:, 80:] = 255
    out = _draw_asset(tmp_path, asset, _wall_with_old_poster(paper=90))
    assert out[240, 240].mean() < 100  # the black half: at the dark poster's level at most
    assert out[240, 320].mean() >= replace._MIN_WHITE - 10  # the white half is not dimmed to the poster's 90


def test_the_ad_carries_grain_that_changes_from_frame_to_frame(tmp_path: Path, scene_look: None) -> None:
    image = _wall_with_old_poster()
    asset = np.full((90, 160, 3), 128, np.uint8)
    first = _draw_asset(tmp_path, asset, image)[120:360, 220:340].astype(float)
    second = _draw_asset(tmp_path, asset, image, calls=2)[120:360, 220:340].astype(float)
    assert 1.0 < first.std() < 2 * replace._GRAIN_SIGMA
    assert (first != second).mean() > 0.3


def test_the_ad_keeps_its_aspect_ratio_on_a_tall_panel(tmp_path: Path) -> None:
    asset = np.full((90, 160, 3), AD_COLOR, np.uint8)
    asset[25:65, 60:100] = (255, 0, 0)  # a square mark in the middle
    out = _draw_asset(tmp_path, asset, _wall_with_old_poster())
    mark = np.all(np.abs(out.astype(int) - (255, 0, 0)) <= 60, axis=-1)
    ys, xs = np.nonzero(mark)
    assert abs((xs.max() - xs.min()) - (ys.max() - ys.min())) <= 3  # still square, not stretched to the panel
    assert _is_ad(out[110:130, 220:340]).all() and _is_ad(out[350:370, 220:340]).all()  # the edge colour fills the rest


def test_the_edge_of_the_ad_fades_into_the_scene(tmp_path: Path) -> None:
    image = _wall_with_skewed_sign()
    cv2.imwrite(str(tmp_path / "ad.png"), np.full((150, 200, 3), AD_COLOR, np.uint8))
    out = AdReplacer(tmp_path).draw(image, [_box(SKEWED_SIGN_BOX)], ASSETS, now=0.0)
    changed = np.any(out != image, axis=-1)
    partly = changed & ~_is_ad(out)
    assert partly.any()  # pixels between the scene and the ad
    edge = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.polylines(edge, [SKEWED_SIGN], True, 255, 13)
    assert not (partly & (edge == 0)).any()  # and only along the outline


def test_small_dark_print_on_the_old_poster_leaves_no_holes_in_the_ad(tmp_path: Path) -> None:
    image = _wall_with_old_poster()
    for y in range(150, 330, 20):
        cv2.line(image, (220, y), (340, y), (20, 20, 20), 2)  # lines of text
    out = _draw_asset(tmp_path, np.full((90, 160, 3), AD_COLOR, np.uint8), image)
    assert _is_ad(out[120:360, 215:345]).all()


def test_a_sign_with_a_found_outline_keeps_its_shape_when_it_restarts_without_one() -> None:
    corners = np.float32([[220, 130], [430, 160], [420, 330], [210, 300]])
    fitted = replace._fitted(corners, (300, 100, 400, 200))
    assert replace._bounds(fitted) == pytest.approx((300, 100, 400, 200))
    assert fitted[1][1] > fitted[0][1] and fitted[3][0] < fitted[0][0]  # still skewed the same way, not a rectangle
