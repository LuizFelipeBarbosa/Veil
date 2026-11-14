"""pygame touch app (WP3, WP13). Runs on the main thread.

Censored preview on the left, a column of thumb-sized buttons on the right
(Record, hold-to-Talk, Panic, Owner), a status dot, the agent transcript and the
recording timer. It only ever shows a CensoredFrame. Works with mouse, touch and
the Mac keys; the layout scales to any landscape window from 480x320 up.

Tapping a face or object on the preview reports the frame pixel under the tap
(on_point). Outlines show what can be pointed at and whether it is blurred, shown
or a replaced ad; they are drawn on the screen only, never into the frame (Never-leaks rule 9).
The Ad button and config.KEY_CYCLE_AD step through the ad images, then off.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Hashable

import cv2
import pygame

import config
from contracts import CensoredFrame, Highlight

log = logging.getLogger(__name__)

_KEY_OWNER = "e"

# Layout, as fractions of the window. The button column is as wide as the
# smaller of these two, so the preview keeps most of the screen.
_COLUMN_WIDTH_OF_WIDTH = 0.30
_COLUMN_WIDTH_OF_HEIGHT = 0.45
_GAP_OF_SHORT_SIDE = 0.02
_BUTTON_FONT_OF_SHORT_SIDE = 0.075
_TEXT_FONT_OF_SHORT_SIDE = 0.06
_TAG_FONT_OF_SHORT_SIDE = 0.042
_MIN_FONT_PX = 18
_MIN_TAG_FONT_PX = 13
_TAP_SLOP_OF_SHORT_SIDE = 0.03  # a pointer that moves further than this is a drag, not a tap
_MIN_TAP_SLOP_PX = 8

_BACKGROUND = (12, 12, 14)
_PLACEHOLDER_BG = (28, 28, 32)
_TEXT = (240, 240, 240)
_DIM_TEXT = (150, 150, 155)
_BUTTON_BG = (52, 54, 60)
_BUTTON_PRESSED_BG = (80, 84, 92)
_BUTTON_BORDER = (110, 112, 120)
_TALK_ON_BG = (30, 120, 220)
_REC_COLOR = (220, 0, 0)
_REC_BUTTON_BG = (120, 20, 20)
_PANIC_BG = (150, 30, 30)
_PANIC_ON_BG = (90, 20, 20)
_PANIC_FILL = (230, 60, 60)
_OWNER_SET_BG = (30, 110, 60)
_STATUS_LIVE = (40, 200, 80)
_STATUS_BLURRED = (230, 40, 40)
_STRIP_RGBA = (0, 0, 0, 160)
_BANNER_RGBA = (200, 0, 0, 220)
# Highlight outlines: Okabe-Ito sky blue and orange stay apart under every common
# colour-vision deficiency, and the tag text says the state in words as well.
_HIGHLIGHT_BLURRED = (86, 180, 233)
_HIGHLIGHT_SHOWN = (230, 159, 0)
_HIGHLIGHT_REPLACED = (204, 121, 167)  # Okabe-Ito reddish purple: an ad with an image drawn over it
_STATE_STYLES = {  # Highlight.state -> outline colour, tag word
    "blur": (_HIGHLIGHT_BLURRED, "blurred"),
    "show": (_HIGHLIGHT_SHOWN, "shown"),
    "replace": (_HIGHLIGHT_REPLACED, "ad"),
}
_TAG_RGBA = (0, 0, 0, 170)
_OUTLINE_PX = 2
_POINT_HINT = "tap a face or sign to blur or unblur"

_BUTTON_NAMES = ("record", "talk", "panic", "owner", "ad")
_MOUSE = "mouse"  # pointer id for the mouse; fingers use ("finger", touch_id, finger_id)


@dataclass(frozen=True)
class UiStatus:
    """What the app shows besides the preview. Read once per UI tick."""

    recording: bool = False
    recording_s: float = 0.0
    fail_closed: bool = False  # red status dot
    panic: bool = False
    talking: bool = False
    transcript: tuple[str, ...] = ()  # newest last, "you: ..." / "agent: ..."
    transcript_opacity: float = 1.0  # 1 fully visible, towards 0 as the chat fades out
    ad: str = ""  # the ad image replacing ads right now, without its extension; empty when off
    owner_enrolled: bool = False


@dataclass(frozen=True)
class UiCallbacks:
    """What the buttons and keys trigger. Called on the main thread; must return fast."""

    on_record_toggle: Callable[[], None] = lambda: None
    on_talk: Callable[[bool], None] = lambda pressed: None  # True on press, False on release
    on_panic: Callable[[bool], None] = lambda on: None
    on_enroll_owner: Callable[[], None] = lambda: None
    on_point: Callable[[int, int], None] = lambda x, y: None  # a tap on the preview, in FRAME pixels
    on_cycle_ad: Callable[[], None] = lambda: None  # config.KEY_CYCLE_AD: the next ad image, or off


@dataclass(frozen=True)
class _Layout:
    """Every rect and font for one window size. Built by _make_layout only."""

    size: tuple[int, int]
    preview_area: pygame.Rect  # left of the buttons; the frame is letterboxed inside
    frame_rect: pygame.Rect  # where the frame is drawn inside preview_area
    buttons: dict[str, pygame.Rect]
    gap: int
    tap_slop: int
    button_font: pygame.font.Font
    text_font: pygame.font.Font
    tag_font: pygame.font.Font

    def button_at(self, pos: tuple[float, float]) -> str | None:
        for name, rect in self.buttons.items():
            if rect.collidepoint(int(pos[0]), int(pos[1])):
                return name
        return None

    def frame_point(self, pos: tuple[float, float]) -> tuple[int, int] | None:
        """The frame pixel under a window position, or None outside the drawn frame."""
        rect = self.frame_rect
        if not (rect.left <= pos[0] < rect.right and rect.top <= pos[1] < rect.bottom):
            return None
        x = math.floor((pos[0] - rect.x) * config.FRAME_WIDTH / rect.width)
        y = math.floor((pos[1] - rect.y) * config.FRAME_HEIGHT / rect.height)
        return min(max(x, 0), config.FRAME_WIDTH - 1), min(max(y, 0), config.FRAME_HEIGHT - 1)

    def window_rect(self, xyxy: tuple[int, int, int, int]) -> pygame.Rect:
        """A box in frame pixels, scaled into the window and clipped to the drawn frame."""
        rect = self.frame_rect
        sx, sy = rect.width / config.FRAME_WIDTH, rect.height / config.FRAME_HEIGHT
        x0, y0, x1, y1 = xyxy
        left, top = rect.x + round(x0 * sx), rect.y + round(y0 * sy)
        box = pygame.Rect(left, top, rect.x + round(x1 * sx) - left, rect.y + round(y1 * sy) - top)
        return box.clip(rect)


def _make_layout(size: tuple[int, int]) -> _Layout:
    """The one place that decides where everything goes."""
    width, height = size
    short = min(width, height)
    gap = max(4, round(short * _GAP_OF_SHORT_SIDE))
    column_width = round(min(width * _COLUMN_WIDTH_OF_WIDTH, height * _COLUMN_WIDTH_OF_HEIGHT))
    column_x = width - column_width

    button_height = (height - gap * (len(_BUTTON_NAMES) + 1)) // len(_BUTTON_NAMES)
    buttons = {
        name: pygame.Rect(column_x + gap, gap + i * (button_height + gap), column_width - 2 * gap, button_height)
        for i, name in enumerate(_BUTTON_NAMES)
    }
    preview_area = pygame.Rect(0, 0, column_x, height)
    return _Layout(
        size=size,
        preview_area=preview_area,
        frame_rect=_letterbox((config.FRAME_WIDTH, config.FRAME_HEIGHT), preview_area),
        buttons=buttons,
        gap=gap,
        tap_slop=max(_MIN_TAP_SLOP_PX, round(short * _TAP_SLOP_OF_SHORT_SIDE)),
        button_font=pygame.font.Font(None, max(_MIN_FONT_PX, round(short * _BUTTON_FONT_OF_SHORT_SIDE))),
        text_font=pygame.font.Font(None, max(_MIN_FONT_PX, round(short * _TEXT_FONT_OF_SHORT_SIDE))),
        tag_font=pygame.font.Font(None, max(_MIN_TAG_FONT_PX, round(short * _TAG_FONT_OF_SHORT_SIDE))),
    )


def _default_window_size() -> tuple[int, int]:
    """The camera frame plus the button column beside it (Mac window)."""
    column_width = round(config.FRAME_HEIGHT * _COLUMN_WIDTH_OF_HEIGHT)
    return config.FRAME_WIDTH + column_width, config.FRAME_HEIGHT


class _Controls:
    """Turns presses and releases into callbacks. Owns the hold state of Talk and Panic.

    A pointer is the mouse, one finger or one key. Talk stays open while any
    pointer holds it and closes exactly once. Panic turns on at the press; clearing
    it needs a fresh press held for config.PANIC_HOLD_S.

    A tap on the preview calls on_point once, at the release, and only when the
    press also started on the frame and the pointer stayed within the tap slop.
    """

    def __init__(self, callbacks: UiCallbacks) -> None:
        self._callbacks = callbacks
        self._held: dict[Hashable, str] = {}  # pointer -> button it pressed
        self._talk_open = False
        self._panic_hold: tuple[Hashable, float] | None = None  # (pointer, pressed at)
        self._taps: dict[Hashable, tuple[float, float]] = {}  # pointer -> window position it went down at

    def start_tap(self, pointer: Hashable, pos: tuple[float, float]) -> None:
        self._taps[pointer] = pos

    def move(self, pointer: Hashable, pos: tuple[float, float], layout: _Layout) -> None:
        """A pointer that wanders off its starting point is a drag; it will not tap."""
        start = self._taps.get(pointer)
        if start is not None and math.dist(start, pos) > layout.tap_slop:
            del self._taps[pointer]

    def end_tap(self, pointer: Hashable, pos: tuple[float, float], layout: _Layout) -> None:
        self.move(pointer, pos, layout)
        if self._taps.pop(pointer, None) is None:
            return
        point = layout.frame_point(pos)
        if point is not None:
            _safe_call(self._callbacks.on_point, *point)

    def is_held(self, button: str) -> bool:
        return button in self._held.values()

    def panic_progress(self, now: float) -> float:
        """0..1 of the clearing hold, 0 when nobody is holding."""
        if self._panic_hold is None:
            return 0.0
        return min(1.0, (now - self._panic_hold[1]) / config.PANIC_HOLD_S)

    def press(self, pointer: Hashable, button: str, status: UiStatus, now: float) -> None:
        if pointer in self._held:
            return  # already down (a key without its KEYUP); one press per pointer
        self._held[pointer] = button
        if button == "record":
            _safe_call(self._callbacks.on_record_toggle)
        elif button == "owner":
            _safe_call(self._callbacks.on_enroll_owner)
        elif button == "ad":
            _safe_call(self._callbacks.on_cycle_ad)
        elif button == "talk" and not self._talk_open:
            self._talk_open = True  # set first: if on_talk raises, the release still closes the mic
            _safe_call(self._callbacks.on_talk, True)
        elif button == "panic":
            if not status.panic:
                _safe_call(self._callbacks.on_panic, True)  # fail closed: instant, no hold
            elif self._panic_hold is None:
                self._panic_hold = (pointer, now)

    def release(self, pointer: Hashable) -> None:
        """Wherever the pointer is now: releasing outside the button still releases it."""
        button = self._held.pop(pointer, None)
        if button == "talk" and self._talk_open and not self.is_held("talk"):
            self._talk_open = False
            _safe_call(self._callbacks.on_talk, False)
        if self._panic_hold is not None and self._panic_hold[0] == pointer:
            self._panic_hold = None  # released early: panic stays on

    def release_all(self) -> None:
        self._taps.clear()  # a tap cut short by focus loss or quitting points at nothing
        for pointer in list(self._held):
            self.release(pointer)

    def tick(self, status: UiStatus, now: float) -> None:
        """Finish a clearing hold that has lasted long enough."""
        if self._panic_hold is None:
            return
        if not status.panic:
            self._panic_hold = None  # cleared elsewhere
        elif now - self._panic_hold[1] >= config.PANIC_HOLD_S:
            self._panic_hold = None
            _safe_call(self._callbacks.on_panic, False)


def _safe_call(callback: Callable[..., None], *args: object) -> None:
    """A raising callback is logged, never allowed to kill the UI."""
    try:
        callback(*args)
    except Exception:
        log.exception("UI callback %r failed", callback)


def run(
    get_frame: Callable[[], CensoredFrame | None],
    get_status: Callable[[], UiStatus] = UiStatus,
    callbacks: UiCallbacks = UiCallbacks(),
    max_frames: int | None = None,
    *,
    now: Callable[[], float] = time.monotonic,
    window_size: tuple[int, int] | None = None,
) -> None:
    """Show the censored preview until the window closes, Esc is pressed or max_frames were drawn.

    `now` and `window_size` exist for tests. config.FULLSCREEN uses the display's size.
    """
    pygame.init()
    if config.FULLSCREEN and window_size is None:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(window_size or _default_window_size(), pygame.RESIZABLE)
    pygame.display.set_caption("privacy-cam")
    clock = pygame.time.Clock()
    keys = {
        pygame.key.key_code(config.KEY_RECORD): "record",
        pygame.key.key_code(config.KEY_TALK): "talk",
        pygame.key.key_code(config.KEY_PANIC): "panic",
        pygame.key.key_code(_KEY_OWNER): "owner",
        pygame.key.key_code(config.KEY_CYCLE_AD): "ad",
    }
    controls = _Controls(callbacks)
    layout = _make_layout(screen.get_size())

    drawn = 0
    try:
        while max_frames is None or drawn < max_frames:
            if screen.get_size() != layout.size:
                layout = _make_layout(screen.get_size())
            status = get_status()
            if _handle_events(controls, layout, keys, status, now()):
                break
            controls.tick(status, now())
            _draw(screen, layout, get_frame(), status, controls, now())
            pygame.display.flip()
            drawn += 1
            clock.tick(config.FPS)
    finally:
        controls.release_all()  # quitting while Talk is held still closes the mic
        pygame.quit()


def _handle_events(
    controls: _Controls, layout: _Layout, keys: dict[int, str], status: UiStatus, now: float
) -> bool:
    """Feed mouse, touch and key events to the controls; True when the app should quit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
        if event.type == pygame.WINDOWFOCUSLOST:
            controls.release_all()
        elif event.type == pygame.KEYDOWN and event.key in keys:
            controls.press(("key", event.key), keys[event.key], status, now)
        elif event.type == pygame.KEYUP:
            controls.release(("key", event.key))
        elif event.type == pygame.MOUSEBUTTONDOWN and _is_real_left_click(event):
            _pointer_down(controls, layout, _MOUSE, event.pos, status, now)
        elif event.type == pygame.MOUSEMOTION and not _is_from_touch(event):
            controls.move(_MOUSE, event.pos, layout)
        elif event.type == pygame.MOUSEBUTTONUP and _is_real_left_click(event):
            controls.release(_MOUSE)
            controls.end_tap(_MOUSE, event.pos, layout)
        elif event.type == pygame.FINGERDOWN:
            _pointer_down(controls, layout, _finger(event), _finger_pos(event, layout), status, now)
        elif event.type == pygame.FINGERMOTION:
            controls.move(_finger(event), _finger_pos(event, layout), layout)
        elif event.type == pygame.FINGERUP:
            controls.release(_finger(event))
            controls.end_tap(_finger(event), _finger_pos(event, layout), layout)
    return False


def _pointer_down(
    controls: _Controls, layout: _Layout, pointer: Hashable, pos: tuple[float, float], status: UiStatus, now: float
) -> None:
    """A press lands on a button, on the frame (a possible tap) or on nothing (the letterbox bars)."""
    button = layout.button_at(pos)
    if button is not None:
        controls.press(pointer, button, status, now)
    elif layout.frame_point(pos) is not None:
        controls.start_tap(pointer, pos)


def _is_real_left_click(event: pygame.event.Event) -> bool:
    return event.button == 1 and not _is_from_touch(event)


def _is_from_touch(event: pygame.event.Event) -> bool:
    """SDL also emits mouse events for touches; the FINGER events already handle those."""
    return bool(getattr(event, "touch", False))


def _finger(event: pygame.event.Event) -> Hashable:
    return ("finger", event.touch_id, event.finger_id)


def _finger_pos(event: pygame.event.Event, layout: _Layout) -> tuple[float, float]:
    """Finger events come normalised to 0..1; kept fractional so the frame mapping stays exact."""
    width, height = layout.size
    return event.x * width, event.y * height


# --- drawing ---------------------------------------------------------------


def _draw(
    screen: pygame.Surface,
    layout: _Layout,
    frame: CensoredFrame | None,
    status: UiStatus,
    controls: _Controls,
    now: float,
) -> None:
    screen.fill(_BACKGROUND)
    _draw_preview(screen, layout, frame)
    highlights = () if frame is None or frame.fail_closed else frame.highlights
    for highlight in highlights:
        _draw_highlight(screen, layout, highlight)
    _draw_status(screen, layout, status)
    if status.panic:
        _draw_panic_banner(screen, layout)
    if status.transcript or not highlights:
        _draw_transcript(screen, layout, status.transcript, status.transcript_opacity)
    else:
        _draw_point_hint(screen, layout)
    _draw_record_button(screen, layout, status, controls)
    _draw_talk_button(screen, layout, status, controls)
    _draw_panic_button(screen, layout, status, controls, now)
    _draw_owner_button(screen, layout, status, controls)
    _draw_ad_button(screen, layout, status, controls)


def _draw_preview(screen: pygame.Surface, layout: _Layout, frame: CensoredFrame | None) -> None:
    """The censored frame, letterboxed. Nothing at all but a dark panel until the first one."""
    area = layout.preview_area
    if frame is None:
        pygame.draw.rect(screen, _PLACEHOLDER_BG, area)
        _blit_centered(screen, layout.text_font.render("starting…", True, _DIM_TEXT), area.center)
        return
    target = layout.frame_rect
    screen.blit(pygame.transform.scale(_to_surface(frame), target.size), target)  # nearest: keeps pixel blocks crisp


def _draw_highlight(screen: pygame.Surface, layout: _Layout, highlight: Highlight) -> None:
    """A thin outline around one pointable track, with a small tag naming it and its state.

    Drawn on the screen only: the frame's image is never touched (Never-leaks rule 9).
    """
    box = layout.window_rect(highlight.xyxy)
    if box.width < 2 or box.height < 2:
        return
    color, state_word = _STATE_STYLES.get(highlight.state, _STATE_STYLES["blur"])
    pygame.draw.rect(screen, color, box, width=_OUTLINE_PX)

    words = [highlight.label, state_word]
    if highlight.is_owner:
        words.append("owner")
    pad = max(2, layout.gap // 2)
    text = _fit(layout.tag_font, " · ".join(words), max(box.width, layout.frame_rect.width // 3) - pad * 2, color)
    tag = pygame.Rect(box.x, box.y - text.get_height() - pad * 2, text.get_width() + pad * 2, text.get_height() + pad * 2)
    if tag.top < layout.frame_rect.top:
        tag.top = box.top  # no room above: tuck it inside the box
    tag.clamp_ip(layout.frame_rect)
    _fill_translucent(screen, tag, _TAG_RGBA)
    screen.blit(text, (tag.x + pad, tag.y + pad))


def _letterbox(source_size: tuple[int, int], area: pygame.Rect) -> pygame.Rect:
    source_width, source_height = source_size
    scale = min(area.width / source_width, area.height / source_height)
    target = pygame.Rect(0, 0, round(source_width * scale), round(source_height * scale))
    target.center = area.center
    return target


def _draw_status(screen: pygame.Surface, layout: _Layout, status: UiStatus) -> None:
    """Green dot "live", or red "blurred" while fail-closed or in panic, top-left of the preview."""
    if status.panic:
        color, label = _STATUS_BLURRED, "blurred: panic"
    elif status.fail_closed:
        color, label = _STATUS_BLURRED, "blurred: fail-safe"
    else:
        color, label = _STATUS_LIVE, "live"
    text = layout.text_font.render(label, True, _TEXT)
    radius = text.get_height() // 3
    strip = pygame.Rect(layout.preview_area.x, layout.preview_area.y, 0, text.get_height() + layout.gap * 2)
    strip.width = layout.gap * 3 + radius * 2 + text.get_width() + layout.gap
    _fill_translucent(screen, strip, _STRIP_RGBA)
    dot_center = (strip.x + layout.gap * 2 + radius, strip.centery)
    pygame.draw.circle(screen, color, dot_center, radius)
    screen.blit(text, (dot_center[0] + radius + layout.gap, strip.centery - text.get_height() // 2))


def _draw_panic_banner(screen: pygame.Surface, layout: _Layout) -> None:
    area = layout.preview_area
    font = layout.button_font
    banner = pygame.Rect(area.x, 0, area.width, font.get_linesize() + layout.gap * 2)
    banner.centery = area.y + area.height // 3
    _fill_translucent(screen, banner, _BANNER_RGBA)
    _blit_centered(screen, _fit(font, "PANIC — everything blurred", banner.width - layout.gap * 2), banner.center)


def _draw_transcript(
    screen: pygame.Surface, layout: _Layout, transcript: tuple[str, ...], opacity: float = 1.0
) -> None:
    """The last few lines, newest last, ellipsised to the preview width, over a translucent strip.

    `opacity` fades the strip and its text together.
    """
    lines = transcript[-config.TRANSCRIPT_LINES:]
    if not lines or opacity <= 0:
        return
    area = layout.preview_area
    font = layout.text_font
    line_height = font.get_linesize()
    strip = pygame.Rect(area.x, 0, area.width, line_height * len(lines) + layout.gap * 2)
    strip.bottom = area.bottom

    chat = pygame.Surface(strip.size, pygame.SRCALPHA)
    chat.fill(_STRIP_RGBA)
    max_width = strip.width - layout.gap * 2
    for i, line in enumerate(lines):
        chat.blit(_fit(font, line, max_width), (layout.gap, layout.gap + i * line_height))
    chat.set_alpha(round(255 * min(1.0, opacity)))
    screen.blit(chat, strip.topleft)


def _draw_point_hint(screen: pygame.Surface, layout: _Layout) -> None:
    """One dim line where the transcript goes, until there is a transcript."""
    area = layout.preview_area
    text = _fit(layout.text_font, _POINT_HINT, area.width - layout.gap * 2, _DIM_TEXT)
    strip = pygame.Rect(area.x, 0, area.width, text.get_height() + layout.gap * 2)
    strip.bottom = area.bottom
    _fill_translucent(screen, strip, _STRIP_RGBA)
    screen.blit(text, (strip.x + layout.gap, strip.y + layout.gap))


def _draw_record_button(screen: pygame.Surface, layout: _Layout, status: UiStatus, controls: _Controls) -> None:
    """Red with the REC dot and mm:ss while recording."""
    rect = layout.buttons["record"]
    if not status.recording:
        _draw_button(screen, layout, rect, "Record", _BUTTON_BG, controls.is_held("record"))
        return
    _draw_button(screen, layout, rect, "", _REC_BUTTON_BG, controls.is_held("record"))
    text = layout.button_font.render(_format_mm_ss(status.recording_s), True, _TEXT)
    radius = text.get_height() // 3
    content_width = radius * 2 + layout.gap + text.get_width()
    dot_center = (rect.centerx - content_width // 2 + radius, rect.centery)
    pygame.draw.circle(screen, _REC_COLOR, dot_center, radius)
    screen.blit(text, (dot_center[0] + radius + layout.gap, rect.centery - text.get_height() // 2))


def _draw_talk_button(screen: pygame.Surface, layout: _Layout, status: UiStatus, controls: _Controls) -> None:
    rect = layout.buttons["talk"]
    if status.talking:
        _draw_button(screen, layout, rect, "Talking…", _TALK_ON_BG, controls.is_held("talk"))
    else:
        _draw_button(screen, layout, rect, "Hold to talk", _BUTTON_BG, controls.is_held("talk"))


def _draw_panic_button(
    screen: pygame.Surface, layout: _Layout, status: UiStatus, controls: _Controls, now: float
) -> None:
    """Off: a red "Panic" button. On: "Hold to clear", filling up while held."""
    rect = layout.buttons["panic"]
    if not status.panic:
        _draw_button(screen, layout, rect, "Panic", _PANIC_BG, controls.is_held("panic"))
        return
    _draw_button(screen, layout, rect, "", _PANIC_ON_BG, controls.is_held("panic"))
    progress = controls.panic_progress(now)
    if progress > 0:
        fill = rect.inflate(-4, -4)
        fill.width = round(fill.width * progress)
        pygame.draw.rect(screen, _PANIC_FILL, fill, border_radius=layout.gap)
    _blit_centered(screen, _fit(layout.button_font, "Hold to clear", rect.width - layout.gap * 2), rect.center)


def _draw_owner_button(screen: pygame.Surface, layout: _Layout, status: UiStatus, controls: _Controls) -> None:
    rect = layout.buttons["owner"]
    if status.owner_enrolled:
        _draw_button(screen, layout, rect, "Owner set", _OWNER_SET_BG, controls.is_held("owner"))
    else:
        _draw_button(screen, layout, rect, "Set owner", _BUTTON_BG, controls.is_held("owner"))


def _draw_ad_button(screen: pygame.Surface, layout: _Layout, status: UiStatus, controls: _Controls) -> None:
    """Each press goes to the next ad image, then off. Lit, with the image's name, while ads are replaced."""
    rect = layout.buttons["ad"]
    if status.ad:
        _draw_button(screen, layout, rect, f"Ad: {status.ad}", _OWNER_SET_BG, controls.is_held("ad"))
    else:
        _draw_button(screen, layout, rect, "Ads", _BUTTON_BG, controls.is_held("ad"))


def _draw_button(
    screen: pygame.Surface, layout: _Layout, rect: pygame.Rect, label: str, color: tuple[int, int, int], pressed: bool
) -> None:
    pygame.draw.rect(screen, _BUTTON_PRESSED_BG if pressed else color, rect, border_radius=layout.gap)
    pygame.draw.rect(screen, _BUTTON_BORDER, rect, width=2, border_radius=layout.gap)
    if label:
        _blit_centered(screen, _fit(layout.button_font, label, rect.width - layout.gap * 2), rect.center)


def _fit(font: pygame.font.Font, text: str, max_width: int, color: tuple[int, int, int] = _TEXT) -> pygame.Surface:
    """Render text, cutting it with an ellipsis when it is wider than max_width."""
    if font.size(text)[0] <= max_width:
        return font.render(text, True, color)
    while text and font.size(text + "…")[0] > max_width:
        text = text[:-1]
    return font.render(text.rstrip() + "…", True, color)


def _blit_centered(screen: pygame.Surface, surface: pygame.Surface, center: tuple[int, int]) -> None:
    screen.blit(surface, surface.get_rect(center=center))


def _fill_translucent(screen: pygame.Surface, rect: pygame.Rect, rgba: tuple[int, int, int, int]) -> None:
    overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
    overlay.fill(rgba)
    screen.blit(overlay, rect)


def _format_mm_ss(seconds: float) -> str:
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _to_surface(frame: CensoredFrame) -> pygame.Surface:
    if not isinstance(frame, CensoredFrame):
        raise TypeError("the preview only accepts CensoredFrame")
    rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB")
