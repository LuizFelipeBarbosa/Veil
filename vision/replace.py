"""Live ad replacement (WP20): draws an image from config.ADS_DIR over each replaced ad.

The result is still a RAW frame. It goes back into render.censor, which pixelates faces
and blurred objects on top of it, so nothing here may skip censoring or keep the frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

import config
from contracts import TrackedBox

log = logging.getLogger(__name__)

# The sign's edge often sits right on the detector's box, so its corners are searched for in
# the box grown by this share of its size per side. The drawing itself stays inside the box.
_SEARCH_MARGIN = 0.1
# A found quad must cover this share of the box (area) to be taken for the sign.
_MIN_QUAD_SHARE = 0.5
_MAX_QUAD_SHARE = 1.3
_MIN_BOUNDS_IOU = 0.7  # between the quad's bounding rectangle and the box
_MIN_HULL_FIT = 0.9  # the 4 corners must keep this share of the hull's area
# Share of the quad's (3 px) outline on edges; the best-supported quad wins. Measured on photo posters
# pasted onto clip frames: true borders found at 0.3 land within 2 px; looser quads were 20-80 px off.
_MIN_EDGE_SUPPORT = 0.3
_APPROX_EPSILONS = (0.02, 0.04, 0.06)  # of the hull's perimeter, for approxPolyDP; the first that gives 4 corners
_CANNY_LOW, _CANNY_HIGH = 50, 150
_MIN_BOX_SIDE = 8  # px; smaller boxes are filled without a corner search
# Per-sign alpha-beta filter on the corners: a constant-velocity prediction corrected by the measurement.
# A found quad is precise (within ~2 px), so it corrects more than the box-derived shape, which jitters
# with the detector's box (up to ~30 px a frame on the ad clip).
_QUAD_GAIN = 0.6
_BOX_GAIN = 0.25
_VELOCITY_GAIN = 0.15  # share of the residual / dt added to the velocity
_SHAPE_GAIN = 0.3  # weight of a newly found quad in the sign's box-relative shape
_COAST_DECAY = 0.7  # velocity kept per frame while the box is a held-over one and no quad is found
# A box overlapping a sign's last box this much is that sign, whatever its track id or label.
_MIN_SAME_SIGN_IOU = 0.3
# A found quad further than this share of the box diagonal (mean corner distance) from the prediction
# is another outline (a poster inside the billboard, a passer-by) and is ignored.
_MAX_QUAD_JUMP = 0.12
# A found quad whose area differs from the sign's by more than this factor is the pane or the frame
# around the panel, not the panel: ignored like a jump.
_MAX_AREA_CHANGE = 1.3
# After this many ignored quads in a row the sign has really moved (a fast pan): it restarts at the quad.
_REJECTS_BEFORE_RESTART = 3
# A box overlapping the sign's last box less than this has jumped: the sign restarts at the measurement.
_MIN_STEADY_IOU = 0.5
# Scene cuts: the frame is shrunk to a 16 x 12 colour thumbnail; when this share of its cells changed by
# more than _CUT_CELL_CHANGE levels since the last call, it is a cut. On the ad clip a hard cut changes
# 62% of cells, a fast pan at most 8%.
_THUMBNAIL_SIZE = (16, 12)
_CUT_CELL_CHANGE = 32
_CUT_SHARE = 0.3
# Shading: each ad pixel is multiplied by how bright the scene is there against the sign's surface
# around it, so posts, glass frames and passers-by darker than the sign stay visible through the ad.
# Pixels at least _SHADE_FULL of the surface keep the ad untouched (the old poster's own content
# is flattened); darker ones fade the ad down to black at _SHADE_DARK.
_SHADE_FULL = 0.6
_SHADE_DARK = 0.2
_SHADE_OFFSET = 16.0  # added to both luminances, so near-black scenes are not read as dark structure
_SURFACE_CELLS = 6  # the surface brightness is the region averaged on this grid, then smoothed back up
_MAX_ASSET_SIDE = 640  # px; assets are shrunk to this on load, which bounds the cache and the warp


def list_assets(directory: Path = config.ADS_DIR) -> tuple[str, ...]:
    """Sorted file names of the ad images in `directory`; empty when it does not exist."""
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in config.AD_IMAGE_SUFFIXES
        )
    )


class _Sign:
    """One physical sign: its filtered corners and, once a quad was found, its shape relative to the box.

    Matched to boxes by overlap, so a new track id or label for the same sign keeps the state.
    """

    def __init__(self, box: tuple[int, int, int, int], now: float) -> None:
        self.box = box  # last box it was matched to (clipped)
        self.box_seen: float | None = None  # its last_seen: an unchanged one is held over, not a new measurement
        self.corners: np.ndarray | None = None  # frame px; set by the first update
        self.velocity = np.zeros((4, 2), np.float32)  # px/s per corner
        self.shape: np.ndarray | None = None  # corners in box-relative units (0..1), once a quad was found
        self.rejected = 0  # found quads ignored in a row
        self.updated = now


class AdReplacer:
    """Warps an ad image onto each replaced sign, with its corners filtered per sign.

    Keeps only the loaded ad images, each sign's corners and a 16 x 12 thumbnail of the last frame
    (576 averaged colours, to notice scene cuts): never a frame.
    """

    def __init__(self, directory: Path = config.ADS_DIR) -> None:
        self._directory = directory
        self._assets: dict[str, np.ndarray | None] = {}  # None: missing or unreadable, logged once
        self._signs: list[_Sign] = []
        self._thumbnail: np.ndarray | None = None
        self._newest_seen = float("-inf")  # newest last_seen of any box so far
        self._stale_until = float("-inf")  # boxes last seen at or before this belong to the scene before a cut

    def draw(
        self, image: np.ndarray, boxes: Sequence[TrackedBox], assets: Mapping[str, str], now: float
    ) -> np.ndarray:
        """A copy of `image` with the asset for each box's label drawn over the sign.

        `boxes` are the boxes whose decision is "replace"; `assets` maps label -> file name.
        A box whose asset is missing or unreadable is left as it is. Overlapping boxes on one sign
        (duplicate tracks, other labels) are drawn once. The drawing stays inside the box.
        """
        out = image.copy()
        height, width = image.shape[:2]
        if self._scene_cut(image):  # nothing from the old scene may carry over, not even held-over boxes
            self._signs = []
            self._stale_until = self._newest_seen
        self._newest_seen = max([self._newest_seen, *(box.last_seen for box in boxes)])
        fresh = [box for box in boxes if box.last_seen > self._stale_until]
        unmatched = list(self._signs)
        for box in _one_per_sign(fresh):
            asset = self._asset(assets.get(box.label, ""))
            xyxy = _clipped(box.xyxy, width, height)
            if asset is None or xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
                continue
            sign = max(unmatched, key=lambda s: _iou(s.box, xyxy), default=None)
            if sign is None or _iou(sign.box, xyxy) < _MIN_SAME_SIGN_IOU:
                sign = _Sign(xyxy, now)
                self._signs.append(sign)
            else:
                unmatched.remove(sign)
            _update(sign, _find_quad(image, xyxy), xyxy, box.last_seen, now)
            _warp_into_box(out, image, asset, sign.corners, xyxy)
        self._signs = [sign for sign in self._signs if now - sign.updated <= config.HOLD_OVER_S]
        return out

    def _scene_cut(self, image: np.ndarray) -> bool:
        """Whether most of the frame changed since the last call; remembers this frame's thumbnail."""
        thumbnail = cv2.resize(image, _THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)
        previous, self._thumbnail = self._thumbnail, thumbnail
        if previous is None:
            return False
        changed = np.abs(thumbnail - previous).max(axis=2) > _CUT_CELL_CHANGE
        return changed.mean() > _CUT_SHARE

    def _asset(self, name: str) -> np.ndarray | None:
        """The cached image for `name`, loaded on first use. Logs a missing one once."""
        if name not in self._assets:
            self._assets[name] = self._load(name)
        return self._assets[name]

    def _load(self, name: str) -> np.ndarray | None:
        # A bare file name only: a path could reach outside the ads directory.
        if not name or Path(name).name != name or Path(name).suffix.lower() not in config.AD_IMAGE_SUFFIXES:
            log.warning("ad image %r is not a file name in %s; leaving its ads as they are", name, self._directory)
            return None
        path = self._directory / name
        asset = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        if asset is None:
            log.warning("cannot read ad image %s; leaving its ads as they are", path)
            return None
        scale = _MAX_ASSET_SIDE / max(asset.shape[:2])
        if scale < 1:
            asset = cv2.resize(asset, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return asset


def _one_per_sign(boxes: Sequence[TrackedBox]) -> list[TrackedBox]:
    """Of boxes overlapping one sign, the freshest (then most confident): held-over copies are stale."""
    kept: list[TrackedBox] = []
    for box in sorted(boxes, key=lambda b: (b.last_seen, b.conf), reverse=True):
        if all(_iou(box.xyxy, other.xyxy) < _MIN_SAME_SIGN_IOU for other in kept):
            kept.append(box)
    return kept


def _update(
    sign: _Sign, found: np.ndarray | None, xyxy: tuple[int, int, int, int], box_seen: float, now: float
) -> None:
    """One alpha-beta step: predict with the velocity, correct with the quad or else the box-relative shape."""
    origin, size = np.float32(xyxy[:2]), np.float32([xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
    if sign.corners is None or _iou(sign.box, xyxy) < _MIN_STEADY_IOU:
        _restart(sign, found, xyxy, box_seen, now)  # new, or the box jumped: no sliding from the old place
        return
    dt = now - sign.updated
    predicted = sign.corners + sign.velocity * dt
    if found is not None and sign.shape is not None and _is_other_outline(found, predicted, size):
        sign.rejected += 1
        if sign.rejected >= _REJECTS_BEFORE_RESTART:  # consistently elsewhere: the sign moved, follow it
            _restart(sign, found, xyxy, box_seen, now)
            return
        found = None  # a sign without a shape yet takes the first quad it finds
    elif found is not None:
        sign.rejected = 0
    if found is not None:
        measured, gain = found, _QUAD_GAIN
        relative = (found - origin) / size
        sign.shape = relative if sign.shape is None else _SHAPE_GAIN * relative + (1 - _SHAPE_GAIN) * sign.shape
    elif box_seen != sign.box_seen:
        shape = sign.shape if sign.shape is not None else _box_corners((0, 0, 1, 1))
        measured, gain = origin + shape * size, _BOX_GAIN
    else:  # a held-over box and no quad: nothing new, coast on the prediction
        measured, gain = predicted, 0.0
        sign.velocity *= _COAST_DECAY

    residual = measured - predicted
    sign.corners = predicted + gain * residual
    if dt > 0:
        sign.velocity += _VELOCITY_GAIN * residual / dt
    sign.box, sign.box_seen, sign.updated = xyxy, box_seen, now


def _restart(
    sign: _Sign, found: np.ndarray | None, xyxy: tuple[int, int, int, int], box_seen: float, now: float
) -> None:
    """The sign starts where it is measured, standing still."""
    origin, size = np.float32(xyxy[:2]), np.float32([xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
    sign.corners = found if found is not None else _box_corners(xyxy)
    sign.shape = None if found is None else (found - origin) / size
    sign.velocity = np.zeros((4, 2), np.float32)
    sign.rejected = 0
    sign.box, sign.box_seen, sign.updated = xyxy, box_seen, now


def _is_other_outline(found: np.ndarray, predicted: np.ndarray, size: np.ndarray) -> bool:
    """A quad far from the prediction or of another size: a poster inside, a passer-by, the glass pane."""
    jump = np.linalg.norm(found - predicted, axis=1).mean()
    area_change = cv2.contourArea(found) / max(1.0, cv2.contourArea(predicted))
    return jump > _MAX_QUAD_JUMP * np.linalg.norm(size) or not 1 / _MAX_AREA_CHANGE <= area_change <= _MAX_AREA_CHANGE


def _grown(xyxy: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    margin_x, margin_y = round(_SEARCH_MARGIN * (x2 - x1)), round(_SEARCH_MARGIN * (y2 - y1))
    return _clipped((x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y), width, height)


def _clipped(xyxy: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _box_corners(xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _find_quad(image: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
    """The sign's 4 corners in frame pixels (top-left, top-right, bottom-right, bottom-left), or None.

    Edges -> contours -> convex hull -> approxPolyDP. A quad counts when it covers a sensible share of
    the box and its own bounding rectangle matches the box (the detector boxed the whole sign);
    of those, the one whose outline lies most on edges wins.
    """
    x1, y1, x2, y2 = xyxy
    box_w, box_h = x2 - x1, y2 - y1
    if min(box_w, box_h) < _MIN_BOX_SIDE:
        return None
    left, top, right, bottom = _grown(xyxy, image.shape[1], image.shape[0])

    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, _CANNY_LOW, _CANNY_HIGH), None)  # close small gaps in the outline
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    box_area = box_w * box_h
    box_in_region = (x1 - left, y1 - top, x2 - left, y2 - top)
    best, best_support = None, _MIN_EDGE_SUPPORT
    for contour in contours:
        rx, ry, rw, rh = cv2.boundingRect(contour)  # the hull's too; cheap, so it filters first
        if _iou((rx, ry, rx + rw, ry + rh), box_in_region) < _MIN_BOUNDS_IOU:
            continue
        # The outline's edges merge with the poster's own and the wall's, so the contour is ragged; its hull is not.
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if not _MIN_QUAD_SHARE * box_area <= area <= _MAX_QUAD_SHARE * box_area:
            continue
        quad = _four_corners(hull)
        if quad is None:
            continue
        support = _edge_support(edges, quad)
        if support > best_support:
            best, best_support = quad, support
    if best is None:
        return None
    return _ordered(best.reshape(4, 2).astype(np.float32) + np.float32([left, top]))


def _four_corners(hull: np.ndarray) -> np.ndarray | None:
    """The hull simplified to 4 corners, if a simplification that still fits it has 4."""
    perimeter = cv2.arcLength(hull, True)
    for epsilon in _APPROX_EPSILONS:
        quad = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(quad) == 4:
            return quad if cv2.contourArea(quad) >= _MIN_HULL_FIT * cv2.contourArea(hull) else None
    return None


def _edge_support(edges: np.ndarray, quad: np.ndarray) -> float:
    """Share of the quad's outline that lies on an edge: high for a real sign border."""
    outline = np.zeros_like(edges)
    cv2.polylines(outline, [quad], True, 255, 3)
    return np.count_nonzero(edges[outline > 0]) / max(1, np.count_nonzero(outline))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ordered(points: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left: clockwise on screen, from the corner nearest the origin."""
    offsets = points - points.mean(axis=0)
    clockwise = points[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
    return np.roll(clockwise, -int(np.argmin(clockwise.sum(axis=1))), axis=0)


def _warp_into_box(
    out: np.ndarray, image: np.ndarray, asset: np.ndarray, corners: np.ndarray, xyxy: tuple[int, int, int, int]
) -> None:
    """Warp `asset` onto `corners`, shaded by `image`, writing only inside the box. Works on the box region alone."""
    x1, y1, x2, y2 = xyxy
    asset_h, asset_w = asset.shape[:2]
    source = np.float32([[0, 0], [asset_w, 0], [asset_w, asset_h], [0, asset_h]])
    local = corners - np.float32([x1, y1])
    homography = cv2.getPerspectiveTransform(source, local)
    region = out[y1:y2, x1:x2]
    warped = cv2.warpPerspective(asset, homography, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local).astype(np.int32), 255)
    shading = cv2.cvtColor(_shading(image[y1:y2, x1:x2]), cv2.COLOR_GRAY2BGR)
    shaded = cv2.multiply(warped, shading, dtype=cv2.CV_8U)
    np.copyto(region, shaded, where=mask[..., None] > 0)


def _shading(region: np.ndarray) -> np.ndarray:
    """Per pixel 0..1: 1 on the sign's surface, less where something clearly darker is (a post, a frame)."""
    luminance = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = luminance.shape
    cells = cv2.resize(luminance, (_SURFACE_CELLS, _SURFACE_CELLS), interpolation=cv2.INTER_AREA)
    surface = cv2.resize(cells, (width, height), interpolation=cv2.INTER_LINEAR)
    ratio = (luminance + _SHADE_OFFSET) / (surface + _SHADE_OFFSET)
    return np.clip((ratio - _SHADE_DARK) / (_SHADE_FULL - _SHADE_DARK), 0.0, 1.0)
