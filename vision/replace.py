"""Live ad replacement (WP20): draws an image from config.ADS_DIR over each replaced ad.

The result is still a RAW frame. It goes back into render.censor, which pixelates faces
and blurred objects on top of it, so nothing here may skip censoring or keep the frame.
"""

from __future__ import annotations

import logging
from collections import Counter
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
# Optical flow: features on the previous frame in and around the sign (the wall and frame edges around
# it move with it; RANSAC drops a passer-by) follow the sign to this frame. Done on a half-size gray.
_FLOW_SCALE = 0.5
_FLOW_SURROUND = 0.3  # share of the sign's size added per side to the feature search area
_MAX_FEATURES = 80
_MIN_INLIERS = 8  # fewer, or under _MIN_INLIER_SHARE of the tracked features: the flow is lost
_MIN_INLIER_SHARE = 0.4
_RANSAC_PX = 1.5  # in the half-size image
# Detections pull the flow back slowly: a found quad near the sign by this share per fresh detection,
# the box only in position and size (it jitters by up to ~30 px), by less.
_QUAD_GAIN = 0.3
_BOX_GAIN = 0.02
# A box overlapping a sign this much (intersection over the smaller area) is that sign, whatever its
# track id or label; boxes overlapping one another this much are one sign.
_MIN_SAME_SIGN_OVERLAP = 0.3
# A sign whose bounds overlap its fresh box's less than this (IoU) disagrees grossly: it restarts there.
# The detector's box jitters by up to ~30 px, which keeps a 200 px sign above 0.7.
_MIN_AGREE_IOU = 0.5
# Two signs whose quads overlap by this share of the smaller one are one panel tracked twice.
_DUPLICATE_SHARE = 0.5
# A found quad further than this share of the box diagonal (mean corner distance) from the followed sign
# is another outline (a poster inside the billboard, a passer-by, texture) and is ignored: the flow is
# more reliable than a far quad, and a sign that really moved away disagrees with its box and restarts.
_MAX_QUAD_JUMP = 0.05
_MAX_AREA_CHANGE = 1.15  # nor one whose area differs more than this factor: the pane or the frame around the panel
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
    """One physical sign: its corners, followed from frame to frame by optical flow.

    Detections only start a sign, keep it alive and pull it slowly back when the flow drifts.
    """

    def __init__(self, found: np.ndarray | None, box: TrackedBox, xyxy: tuple[int, int, int, int], now: float) -> None:
        # Frame px, top-left, top-right, bottom-right, bottom-left: the quad found in the box, else the box.
        self.corners = found if found is not None else _box_corners(xyxy)
        self.outlined = found is not None  # the corners come from a found quad, not only from a box
        self.candidate: np.ndarray | None = None  # a box-started sign's last found quad, moved with the flow
        # Where it may be drawn: its last fresh box, moved with the flow since (see AdReplacer.draw).
        self.region = np.float32(xyxy)
        self.label = box.label  # of the newest box supporting it: whose chosen image it starts its rotation from
        self.asset_name = ""  # its own image, kept while it lives (see AdReplacer._pick)
        self.seen = box.last_seen  # last_seen of the newest box supporting it (the detector's clock)
        self.updated = now  # when that box first reached draw() (draw's clock): the hold-over counts from here
        self.support = 0  # frames with a supporting box: the established sign wins an overlap


class AdReplacer:
    """Warps an ad image onto each replaced sign, following each sign with optical flow.

    Signs seen together show different images: the most established one shows the image chosen for its
    label, the others the next images in the ads directory after it (sorted, wrapping around).

    Keeps the loaded ad images, each sign's corners, a 16 x 12 thumbnail of the last frame (to notice
    scene cuts) and the last frame as a downscaled grayscale image for the optical flow. That grayscale
    copy is raw image content: it lives only in this object, in the vision thread's memory like the ring
    buffer, and is never written, logged, returned or exposed.
    """

    def __init__(self, directory: Path = config.ADS_DIR) -> None:
        self._directory = directory
        self._names = list_assets(directory)  # rescanned only when the operator names a file not in it
        self._chosen: dict[str, str] = {}  # the label -> file name mapping the signs' images were picked for
        self._assets: dict[str, np.ndarray | None] = {}  # None: missing or unreadable, logged once
        self._signs: list[_Sign] = []
        self._thumbnail: np.ndarray | None = None
        self._gray: np.ndarray | None = None  # the last frame, gray, scaled by _FLOW_SCALE; never leaves
        self._newest_seen = float("-inf")  # newest last_seen of any box so far
        # Boxes last seen at or before this belong to the scene before a cut, or to a sign dropped after the hold-over.
        self._stale_until = float("-inf")

    def draw(
        self, image: np.ndarray, boxes: Sequence[TrackedBox], assets: Mapping[str, str], now: float
    ) -> np.ndarray:
        """A copy of `image` with the asset for each sign's label drawn over the sign.

        `boxes` are the boxes whose decision is "replace"; `assets` maps label -> file name.
        A box whose asset is missing or unreadable is left as it is. Boxes overlapping one another
        (duplicate tracks, other labels, a box nested in another) are one sign, drawn once, and no two
        drawn signs overlap. A sign is drawn only in a frame where a box supports it, and only inside
        its region: its last fresh box (clipped to the frame) moved with the flow since; no margin.
        """
        out = image.copy()
        height, width = image.shape[:2]
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, fx=_FLOW_SCALE, fy=_FLOW_SCALE,
                          interpolation=cv2.INTER_AREA)
        previous_gray, self._gray = self._gray, gray
        if self._scene_cut(image):  # nothing from the old scene may carry over, not even held-over boxes
            self._signs = []
            self._stale_until = self._newest_seen
        self._newest_seen = max([self._newest_seen, *(box.last_seen for box in boxes)])
        expired = [sign for sign in self._signs if now - sign.updated > config.HOLD_OVER_S]
        self._stale_until = max([self._stale_until, *(sign.seen for sign in expired)])  # its held-over boxes too
        self._signs = [sign for sign in self._signs if sign not in expired]
        if assets != self._chosen:
            self._choose(assets)

        lost = {id(sign) for sign in self._signs if previous_gray is None or not _follow(sign, previous_gray, gray)}
        supported: list[_Sign] = []
        for box in _one_per_sign([box for box in boxes if box.last_seen > self._stale_until]):
            xyxy = _clipped(box.xyxy, width, height)
            if self._asset(assets.get(box.label, "")) is None or xyxy[2] - xyxy[0] < 2 or xyxy[3] - xyxy[1] < 2:
                continue
            candidates = [sign for sign in self._signs if sign not in supported]
            sign = max(candidates, key=lambda s: _overlap(_bounds(s.corners), xyxy), default=None)
            if sign is None or _overlap(_bounds(sign.corners), xyxy) < _MIN_SAME_SIGN_OVERLAP:
                sign = _Sign(_find_quad(image, xyxy), box, xyxy, now)
                sign.asset_name = self._pick(assets[box.label], self._signs)
                self._signs.append(sign)
            elif box.last_seen > sign.seen or id(sign) in lost:  # correct the flow; a lost sign restarts at its box
                _correct(sign, image, xyxy, restart=id(sign) in lost)
            sign.label = box.label
            if box.last_seen > sign.seen:
                sign.seen, sign.updated = box.last_seen, now
            sign.support += 1
            supported.append(sign)

        drawn: list[_Sign] = []
        for sign in sorted(supported, key=lambda s: s.support, reverse=True):
            clashing = [other for other in drawn if _quad_overlap(sign.corners, other.corners) > 0]
            if clashing:  # another sign is drawn there: never an ad on an ad
                if any(_quad_overlap(sign.corners, other.corners) >= _DUPLICATE_SHARE * min(
                        cv2.contourArea(sign.corners), cv2.contourArea(other.corners)) for other in clashing):
                    self._signs.remove(sign)  # the same panel tracked twice: the established one stays
                continue
            drawn.append(sign)
            region = _clipped(_rounded(sign.region), width, height)
            asset = self._asset(sign.asset_name)
            if asset is not None and region[2] - region[0] >= 2 and region[3] - region[1] >= 2:
                _warp_into_box(out, image, asset, sign.corners, region)
        return out

    def _choose(self, assets: Mapping[str, str]) -> None:
        """The operator's mapping changed: every living sign picks again, the most established first,
        so that one shows the new choice and the others follow it."""
        self._chosen = dict(assets)
        if any(name not in self._names for name in assets.values()):
            self._names = list_assets(self._directory)
        picked: list[_Sign] = []
        for sign in sorted(self._signs, key=lambda s: s.support, reverse=True):
            sign.asset_name = self._pick(assets.get(sign.label, ""), picked)
            picked.append(sign)

    def _pick(self, chosen: str, others: Sequence[_Sign]) -> str:
        """A new sign's image: the first in the rotation from `chosen` that none of `others` shows;
        when all are shown, the least shown, again in rotation order. The rotation is `chosen` (even
        when unreadable: it is never silently replaced) then the readable images after it, wrapping."""
        start = self._names.index(chosen) + 1 if chosen in self._names else 0
        following = self._names[start:] + self._names[:start]
        rotation = [chosen, *(name for name in following if name != chosen and self._asset(name) is not None)]
        shown = Counter(sign.asset_name for sign in others)
        return min(rotation, key=lambda name: shown[name])  # the first of the least shown

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
    """One box per group of overlapping boxes (nested or partly overlapping count, by the smaller box's
    share): the freshest, then the largest. Held-over copies are stale; the largest is the whole panel."""
    kept: list[TrackedBox] = []
    for box in sorted(boxes, key=lambda b: (b.last_seen, _area(b.xyxy)), reverse=True):
        if all(_overlap(box.xyxy, other.xyxy) < _MIN_SAME_SIGN_OVERLAP for other in kept):
            kept.append(box)
    return kept


def _follow(sign: _Sign, previous_gray: np.ndarray, gray: np.ndarray) -> bool:
    """Moves the sign's corners and region by the motion of the scene in and around it.

    False when the flow is lost (too few consistent features); a flat area with no features at all
    gives no motion to follow, so the sign stays where it is and detections correct it.
    """
    x1, y1, x2, y2 = _bounds(sign.corners)
    grow_x, grow_y = _FLOW_SURROUND * (x2 - x1), _FLOW_SURROUND * (y2 - y1)
    height, width = gray.shape
    left, top = max(0, int((x1 - grow_x) * _FLOW_SCALE)), max(0, int((y1 - grow_y) * _FLOW_SCALE))
    right, bottom = min(width, int((x2 + grow_x) * _FLOW_SCALE) + 1), min(height, int((y2 + grow_y) * _FLOW_SCALE) + 1)
    if right - left < 4 or bottom - top < 4:
        return False
    mask = np.zeros_like(gray)
    mask[top:bottom, left:right] = 255
    points = cv2.goodFeaturesToTrack(previous_gray, _MAX_FEATURES, 0.01, 4, mask=mask)
    if points is None:
        return True
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None, winSize=(15, 15), maxLevel=2)
    tracked = status.ravel() == 1
    if tracked.sum() < _MIN_INLIERS:
        return False
    transform, inliers = cv2.estimateAffinePartial2D(
        points[tracked], moved[tracked], method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_PX
    )
    if transform is None or inliers.sum() < max(_MIN_INLIERS, _MIN_INLIER_SHARE * tracked.sum()):
        return False
    transform[:, 2] /= _FLOW_SCALE  # the same motion in full-size pixels
    sign.corners = cv2.transform(sign.corners[None], transform)[0]
    if sign.candidate is not None:
        sign.candidate = cv2.transform(sign.candidate[None], transform)[0]
    region = np.float32([sign.region[:2], sign.region[2:]])
    moved_region = cv2.transform(region[None], transform)[0]
    sign.region = np.float32([*moved_region.min(axis=0), *moved_region.max(axis=0)])
    return True


def _correct(sign: _Sign, image: np.ndarray, xyxy: tuple[int, int, int, int], restart: bool) -> None:
    """A fresh detection: restart the sign there when its flow was lost or it disagrees grossly with the
    box, else pull it a little towards the quad found in the box, or towards the box's position and size. A far quad is ignored."""
    found = _find_quad(image, xyxy)
    sign.region = np.float32(xyxy)
    # A sign started from a box takes a quad found twice in a row at one place (moved with the flow):
    # a single one may be texture that happens to fit the box.
    confirmed = found is not None and sign.candidate is not None and not _is_other_outline(found, sign.candidate, xyxy)
    on_screen = _clipped(_bounds(sign.corners), image.shape[1], image.shape[0])  # the box is clipped too
    if restart or _iou(on_screen, xyxy) < _MIN_AGREE_IOU or (confirmed and not sign.outlined):
        sign.corners = found if found is not None else _box_corners(xyxy)
        sign.outlined, sign.candidate = found is not None, None
        return
    if not sign.outlined:
        sign.candidate = found
    if found is not None and not _is_other_outline(found, sign.corners, xyxy):
        sign.corners = sign.corners + _QUAD_GAIN * (found - sign.corners)
        return
    x1, y1, x2, y2 = _bounds(sign.corners)
    centre, size = np.float32([(x1 + x2) / 2, (y1 + y2) / 2]), np.float32([x2 - x1, y2 - y1])
    box_centre = np.float32([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])
    box_size = np.float32([xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
    scale = 1 + _BOX_GAIN * (box_size / np.maximum(size, 1) - 1)
    sign.corners = centre + _BOX_GAIN * (box_centre - centre) + (sign.corners - centre) * scale


def _is_other_outline(found: np.ndarray, corners: np.ndarray, xyxy: tuple[int, int, int, int]) -> bool:
    """A quad far from the sign or of another size: a poster inside, a passer-by, the glass pane."""
    jump = np.linalg.norm(found - corners, axis=1).mean()
    area_change = cv2.contourArea(found) / max(1.0, cv2.contourArea(corners))
    diagonal = np.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    return jump > _MAX_QUAD_JUMP * diagonal or not 1 / _MAX_AREA_CHANGE <= area_change <= _MAX_AREA_CHANGE


def _quad_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Area shared by two quads (their convex hulls), px."""
    area, _ = cv2.intersectConvexConvex(cv2.convexHull(a.astype(np.float32)), cv2.convexHull(b.astype(np.float32)))
    return area


def _rounded(region: np.ndarray) -> tuple[int, int, int, int]:
    """Whole pixels inside the region: a box that has not moved is drawn in exactly its own pixels."""
    x1, y1, x2, y2 = (float(v) for v in region)
    return int(np.ceil(x1 - 1e-3)), int(np.ceil(y1 - 1e-3)), int(np.floor(x2 + 1e-3)), int(np.floor(y2 + 1e-3))


def _bounds(corners: np.ndarray) -> tuple[float, float, float, float]:
    return (*corners.min(axis=0), *corners.max(axis=0))


def _area(xyxy) -> float:
    return max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])


def _overlap(a, b) -> float:
    """Intersection over the smaller box's area: 1 for a box nested in another."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


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
