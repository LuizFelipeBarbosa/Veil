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
# Weight of the new corners in the per-track exponential smoothing; 1.0 turns smoothing off.
_SMOOTHING = 0.4
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


class AdReplacer:
    """Warps an ad image onto each replaced sign, with its corners smoothed per track.

    Keeps only the loaded ad images and each track's corners: never a frame.
    """

    def __init__(self, directory: Path = config.ADS_DIR) -> None:
        self._directory = directory
        self._assets: dict[str, np.ndarray | None] = {}  # None: missing or unreadable, logged once
        self._corners: dict[str, tuple[np.ndarray, float]] = {}  # track id -> (smoothed corners, last seen)

    def draw(
        self, image: np.ndarray, boxes: Sequence[TrackedBox], assets: Mapping[str, str], now: float
    ) -> np.ndarray:
        """A copy of `image` with the asset for each box's label drawn over the sign.

        `boxes` are the boxes whose decision is "replace"; `assets` maps label -> file name.
        A box whose asset is missing or unreadable is left as it is.
        """
        out = image.copy()
        height, width = image.shape[:2]
        for box in boxes:
            asset = self._asset(assets.get(box.label, ""))
            x1, y1, x2, y2 = _clipped(box.xyxy, width, height)
            if asset is None or x2 - x1 < 2 or y2 - y1 < 2:
                continue
            found = _find_quad(image, (x1, y1, x2, y2))
            measured = found if found is not None else _box_corners((x1, y1, x2, y2))
            corners = self._smoothed(box.id, measured, now)
            _warp_into_box(out, asset, corners, (x1, y1, x2, y2))
        self._corners = {
            track_id: state for track_id, state in self._corners.items() if now - state[1] <= config.HOLD_OVER_S
        }
        return out

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

    def _smoothed(self, track_id: str, measured: np.ndarray, now: float) -> np.ndarray:
        previous = self._corners.get(track_id)
        if previous is None:
            corners = measured
        else:
            corners = _SMOOTHING * measured + (1 - _SMOOTHING) * previous[0]
        self._corners[track_id] = (corners, now)
        return corners


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
    height, width = image.shape[:2]
    margin_x, margin_y = round(_SEARCH_MARGIN * box_w), round(_SEARCH_MARGIN * box_h)
    left, top = max(0, x1 - margin_x), max(0, y1 - margin_y)
    right, bottom = min(width, x2 + margin_x), min(height, y2 + margin_y)

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


def _warp_into_box(out: np.ndarray, asset: np.ndarray, corners: np.ndarray, xyxy: tuple[int, int, int, int]) -> None:
    """Warp `asset` onto `corners`, writing only inside the box. Works on the box-sized region alone."""
    x1, y1, x2, y2 = xyxy
    asset_h, asset_w = asset.shape[:2]
    source = np.float32([[0, 0], [asset_w, 0], [asset_w, asset_h], [0, asset_h]])
    local = corners - np.float32([x1, y1])
    homography = cv2.getPerspectiveTransform(source, local)
    region = out[y1:y2, x1:x2]
    warped = cv2.warpPerspective(asset, homography, (x2 - x1, y2 - y1), flags=cv2.INTER_LINEAR)
    mask = np.zeros(region.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(local).astype(np.int32), 255)
    region[mask > 0] = warped[mask > 0]
