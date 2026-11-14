"""WP1 + WP3 + WP13: the touch UI. Headless pygame, no real screen; time comes from a fake clock."""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # before pygame opens a display

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pygame
import pytest

import config
import ui
from contracts import CensoredFrame, Highlight
from ui import UiCallbacks, UiStatus

SIZES = [(640, 480), (480, 320), (1280, 800), (800, 480), ui._default_window_size()]
POINT_SIZES = [ui._default_window_size(), (480, 320), (1280, 800)]  # 856x480: the Mac window, no bars
SCREENSHOT_DIR_ENV = "PRIVACY_CAM_UI_SCREENSHOTS"  # set to a directory to write the PNGs

HIGHLIGHTS = (
    Highlight(id="t1", label="face", xyxy=(80, 140, 200, 300), state="blur"),
    Highlight(id="t2", label="face", xyxy=(300, 120, 420, 290), state="show", is_owner=True),
    Highlight(id="t3", label="street sign", xyxy=(470, 220, 600, 300), state="blur"),
)


def _black_frame() -> CensoredFrame:
    image = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    return CensoredFrame(image=image, index=0, fail_closed=True)


def _button_rects(size: tuple[int, int]) -> dict[str, pygame.Rect]:
    pygame.font.init()
    return dict(ui._make_layout(size).buttons)


def _frame_rect(size: tuple[int, int]) -> pygame.Rect:
    pygame.font.init()
    return ui._make_layout(size).frame_rect


# --- event builders ---------------------------------------------------------


def _key_down(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _key_up(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYUP, key=key)


def _mouse_down(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1, touch=False)


def _mouse_up(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1, touch=False)


def _mouse_move(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(1, 0, 0), touch=False)


def _finger(kind: int, pos: tuple[float, float], size: tuple[int, int], finger_id: int = 0) -> pygame.event.Event:
    x, y = pos[0] / size[0], pos[1] / size[1]
    return pygame.event.Event(kind, x=x, y=y, dx=0.0, dy=0.0, touch_id=1, finger_id=finger_id, pressure=1.0)


def _press_release(kind: str, pos: tuple[int, int], size: tuple[int, int]) -> list[pygame.event.Event]:
    if kind == "mouse":
        return [_mouse_down(pos), _mouse_up(pos)]
    return [_finger(pygame.FINGERDOWN, pos, size), _finger(pygame.FINGERUP, pos, size)]


def _tap(kind: str, pixel: tuple[int, int], size: tuple[int, int]) -> list[pygame.event.Event]:
    """A tap on one window pixel. A finger lands on the pixel's centre, as a real touch would."""
    if kind == "mouse":
        return _press_release(kind, pixel, size)
    return _press_release(kind, (pixel[0] + 0.5, pixel[1] + 0.5), size)  # type: ignore[arg-type]


def _drag(kind: str, start: tuple[int, int], end: tuple[int, int], size: tuple[int, int]) -> list[pygame.event.Event]:
    """Press at start, move to end and back, release at start: still a drag, never a tap."""
    if kind == "mouse":
        return [_mouse_down(start), _mouse_move(end), _mouse_move(start), _mouse_up(start)]
    return [
        _finger(pygame.FINGERDOWN, start, size),
        _finger(pygame.FINGERMOTION, end, size),
        _finger(pygame.FINGERMOTION, start, size),
        _finger(pygame.FINGERUP, start, size),
    ]


# --- a fake app around ui.run ------------------------------------------------


class _FakeApp:
    """Records every callback and mirrors panic into the status, like app.py does with Policy."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.panic = False
        self.now = 0.0
        self.ticks = 0

    def callbacks(self, **overrides: Callable) -> UiCallbacks:
        callbacks = {
            "on_record_toggle": lambda: self.calls.append(("record",)),
            "on_talk": lambda pressed: self.calls.append(("talk", pressed)),
            "on_panic": self._set_panic,
            "on_enroll_owner": lambda: self.calls.append(("owner",)),
            "on_point": lambda x, y: self.calls.append(("point", x, y)),
        }
        return UiCallbacks(**{**callbacks, **overrides})

    def _set_panic(self, on: bool) -> None:
        self.calls.append(("panic", on))
        self.panic = on

    def run(self, script: dict[int, list], size: tuple[int, int] = (640, 480), frames: int | None = None, **overrides) -> None:
        """script: tick -> steps. A float sets the clock, an Event is posted; both apply on the next tick."""

        def get_frame() -> CensoredFrame:
            self.ticks += 1
            for step in script.get(self.ticks, ()):
                if isinstance(step, float):
                    self.now = step
                else:
                    pygame.event.post(step)
            return _black_frame()

        ui.run(
            get_frame,
            get_status=lambda: UiStatus(panic=self.panic),
            callbacks=self.callbacks(**overrides),
            max_frames=frames or max(script, default=0) + 2,
            now=lambda: self.now,
            window_size=size,
        )


# --- WP1 behaviour, still there ---------------------------------------------


def test_r_key_toggles_recording() -> None:
    toggles = []
    ticks = 0

    def get_frame() -> CensoredFrame:
        nonlocal ticks
        ticks += 1
        if ticks in (1, 3):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_r))
        if ticks == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x))  # not the record key
        return _black_frame()

    ui.run(get_frame, callbacks=UiCallbacks(on_record_toggle=lambda: toggles.append(1)), max_frames=5)
    assert len(toggles) == 2


def test_rec_dot_and_timer_show_only_while_recording() -> None:
    rect = _button_rects(ui._default_window_size())["record"]
    red_pixels = []

    def get_frame() -> CensoredFrame:
        surface = pygame.display.get_surface()  # still holds the previous tick's drawing
        array = pygame.surfarray.array3d(surface.subsurface(rect))
        red_pixels.append(int(np.all(array == ui._REC_COLOR, axis=-1).sum()))
        return _black_frame()

    ui.run(get_frame, get_status=lambda: UiStatus(recording=True, recording_s=75.0), max_frames=3)
    assert all(count > 20 for count in red_pixels[1:])

    red_pixels.clear()
    ui.run(get_frame, get_status=UiStatus, max_frames=3)
    assert red_pixels[1:] == [0, 0]


def test_mm_ss() -> None:
    assert ui._format_mm_ss(0) == "00:00"
    assert ui._format_mm_ss(75.9) == "01:15"
    assert ui._format_mm_ss(3600) == "60:00"


# --- keys ---------------------------------------------------------------------


def test_keys_trigger_every_button() -> None:
    app = _FakeApp()
    app.run({
        1: [_key_down(pygame.K_r), _key_up(pygame.K_r)],
        2: [_key_down(pygame.K_t)],
        3: [_key_up(pygame.K_t)],
        4: [_key_down(pygame.K_p), _key_up(pygame.K_p)],
        5: [_key_down(pygame.K_e), _key_up(pygame.K_e)],
    })
    assert app.calls == [("record",), ("talk", True), ("talk", False), ("panic", True), ("owner",)]


def test_ad_key_cycles_once_per_key_down() -> None:
    ad_key = ord(config.KEY_CYCLE_AD)  # a letter key's pygame code is its ASCII code
    app = _FakeApp()
    app.run({
        1: [_key_down(ad_key)],
        2: [_key_down(ad_key)],  # no KEYUP yet: a repeat, not a new press
        3: [_key_up(ad_key)],
        4: [_key_down(ad_key), _key_up(ad_key)],
    }, on_cycle_ad=lambda: app.calls.append(("ad",)))
    assert app.calls == [("ad",), ("ad",)]


def test_ad_key_works_with_the_default_callbacks() -> None:
    ad_key = ord(config.KEY_CYCLE_AD)  # a letter key's pygame code is its ASCII code
    app = _FakeApp()
    app.run({1: [_key_down(ad_key), _key_up(ad_key)], 2: [_key_down(pygame.K_e)]})  # no on_cycle_ad given
    assert app.calls == [("owner",)]
    assert UiCallbacks().on_cycle_ad() is None


def test_escape_quits() -> None:
    app = _FakeApp()
    app.run({1: [_key_down(pygame.K_ESCAPE)]}, frames=50)
    assert app.ticks == 1


# --- layout -------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_layout_fits_and_buttons_are_thumb_sized(size: tuple[int, int]) -> None:
    pygame.font.init()
    layout = ui._make_layout(size)
    window = pygame.Rect((0, 0), size)
    rects = list(layout.buttons.values())
    min_side = 0.15 * min(size)
    for i, rect in enumerate(rects):
        assert window.contains(rect)
        assert rect.width >= min_side and rect.height >= min_side
        assert not rect.colliderect(layout.preview_area)
        assert all(not rect.colliderect(other) for other in rects[i + 1:])
    assert window.contains(layout.preview_area)
    assert layout.preview_area.width >= size[0] // 2


def test_default_window_shows_the_frame_unscaled() -> None:
    pygame.font.init()
    layout = ui._make_layout(ui._default_window_size())
    assert layout.preview_area.size == (config.FRAME_WIDTH, config.FRAME_HEIGHT)


# --- pointers on every button at every size -------------------------------------


@pytest.mark.parametrize("kind", ["mouse", "finger"])
@pytest.mark.parametrize("size", [(640, 480), (480, 320), (1280, 800)])
def test_pointer_presses_every_button(kind: str, size: tuple[int, int]) -> None:
    rects = _button_rects(size)
    app = _FakeApp()
    app.run(
        {i + 1: _press_release(kind, rects[name].center, size) for i, name in enumerate(["record", "talk", "panic", "owner"])},
        size=size,
    )
    assert app.calls == [("record",), ("talk", True), ("talk", False), ("panic", True), ("owner",)]


@pytest.mark.parametrize("kind", ["mouse", "finger"])
def test_press_on_preview_presses_no_button(kind: str) -> None:
    app = _FakeApp()
    app.run({1: _press_release(kind, (50, 200), (640, 480))})
    assert [call[0] for call in app.calls] == ["point"]


# --- pointing at the preview (WP13) ---------------------------------------------------


def _assert_points(calls: list[tuple], expected: list[tuple[int, int]]) -> None:
    assert [call[0] for call in calls] == ["point"] * len(expected)
    for (_, x, y), (want_x, want_y) in zip(calls, expected):
        assert abs(x - want_x) <= 1 and abs(y - want_y) <= 1, ((x, y), (want_x, want_y))


@pytest.mark.parametrize("kind", ["mouse", "finger"])
@pytest.mark.parametrize("size", POINT_SIZES)
def test_tap_maps_window_to_frame_pixels(kind: str, size: tuple[int, int]) -> None:
    rect = _frame_rect(size)
    last_x, last_y = config.FRAME_WIDTH - 1, config.FRAME_HEIGHT - 1
    taps = [
        (rect.center, (config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2)),
        (rect.topleft, (0, 0)),
        ((rect.right - 1, rect.top), (last_x, 0)),
        ((rect.left, rect.bottom - 1), (0, last_y)),
        ((rect.right - 1, rect.bottom - 1), (last_x, last_y)),
    ]
    app = _FakeApp()
    app.run({i + 1: _tap(kind, pixel, size) for i, (pixel, _) in enumerate(taps)}, size=size)
    _assert_points(app.calls, [frame_pixel for _, frame_pixel in taps])


def test_mac_window_maps_one_to_one() -> None:
    pygame.font.init()
    layout = ui._make_layout(ui._default_window_size())
    assert layout.frame_rect == pygame.Rect(0, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    assert layout.frame_point((123, 345)) == (123, 345)


@pytest.mark.parametrize("kind", ["mouse", "finger"])
@pytest.mark.parametrize("size", [(480, 320), (1280, 800)])
def test_tap_in_the_letterbox_bars_points_at_nothing(kind: str, size: tuple[int, int]) -> None:
    rect = _frame_rect(size)
    assert rect.top > 0  # these sizes do have bars
    app = _FakeApp()
    app.run({
        1: _tap(kind, (rect.centerx, rect.top - 3), size),
        2: _tap(kind, (rect.centerx, rect.bottom + 2), size),
        3: _tap(kind, (rect.left, 0), size),
    }, size=size)
    assert app.calls == []


@pytest.mark.parametrize("kind", ["mouse", "finger"])
@pytest.mark.parametrize("size", POINT_SIZES)
def test_tap_on_a_button_does_not_point(kind: str, size: tuple[int, int]) -> None:
    app = _FakeApp()
    app.run({1: _tap(kind, _button_rects(size)["owner"].center, size)}, size=size)
    assert app.calls == [("owner",)]


@pytest.mark.parametrize("kind", ["mouse", "finger"])
@pytest.mark.parametrize("size", POINT_SIZES)
def test_drag_on_the_preview_does_not_point(kind: str, size: tuple[int, int]) -> None:
    rect = _frame_rect(size)
    far = (rect.centerx + rect.width // 5, rect.centery)
    app = _FakeApp()
    app.run({1: _drag(kind, rect.center, far, size)}, size=size)
    assert app.calls == []


@pytest.mark.parametrize("kind", ["mouse", "finger"])
def test_small_wobble_is_still_a_tap(kind: str) -> None:
    size = (480, 320)
    rect = _frame_rect(size)
    nudge = (rect.centerx + 3, rect.centery + 2)
    app = _FakeApp()
    if kind == "mouse":
        events = [_mouse_down(rect.center), _mouse_move(nudge), _mouse_up(nudge)]
    else:
        events = [_finger(pygame.FINGERDOWN, rect.center, size), _finger(pygame.FINGERMOTION, nudge, size),
                  _finger(pygame.FINGERUP, nudge, size)]
    app.run({1: events}, size=size)
    assert [call[0] for call in app.calls] == ["point"]


@pytest.mark.parametrize("kind", ["mouse", "finger"])
def test_talk_press_released_over_the_preview_closes_and_does_not_point(kind: str) -> None:
    size = (480, 320)
    talk, centre = _button_rects(size)["talk"].center, _frame_rect(size).center
    if kind == "mouse":
        script = {1: [_mouse_down(talk)], 2: [_mouse_move(centre)], 3: [_mouse_up(centre)]}
    else:
        script = {1: [_finger(pygame.FINGERDOWN, talk, size)], 2: [_finger(pygame.FINGERMOTION, centre, size)],
                  3: [_finger(pygame.FINGERUP, centre, size)]}
    app = _FakeApp()
    app.run(script, size=size)
    assert app.calls == [("talk", True), ("talk", False)]


def test_mouse_events_synthesised_from_a_touch_do_not_point_twice() -> None:
    size = (480, 320)
    centre = _frame_rect(size).center
    synthetic = [pygame.event.Event(kind, pos=centre, button=1, touch=True)
                 for kind in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)]
    app = _FakeApp()
    app.run({1: [_finger(pygame.FINGERDOWN, centre, size), synthetic[0],
                 _finger(pygame.FINGERUP, centre, size), synthetic[1]]}, size=size)
    assert [call[0] for call in app.calls] == ["point"]


def test_a_raising_on_point_does_not_stop_the_loop() -> None:
    def boom(x: int, y: int) -> None:
        raise RuntimeError("toggle failed")

    centre = _frame_rect((640, 480)).center
    app = _FakeApp()
    app.run({1: _tap("mouse", centre, (640, 480)), 2: [_key_down(pygame.K_e)], 3: _tap("finger", centre, (640, 480))},
            frames=6, on_point=boom)
    assert app.ticks == 6
    assert app.calls == [("owner",)]


# --- highlights on the preview (WP13) --------------------------------------------------


def _render(size: tuple[int, int], frame: CensoredFrame, status: UiStatus = UiStatus()) -> pygame.Surface:
    pygame.font.init()
    screen = pygame.Surface(size)
    ui._draw(screen, ui._make_layout(size), frame, status, ui._Controls(UiCallbacks()), now=0.0)
    return screen


def _with_highlights(frame: CensoredFrame, highlights: tuple[Highlight, ...] = HIGHLIGHTS) -> CensoredFrame:
    return CensoredFrame(image=frame.image, index=frame.index, fail_closed=frame.fail_closed, highlights=highlights)


@pytest.mark.parametrize("size", POINT_SIZES)
def test_highlights_are_outlined_at_their_scaled_position(size: tuple[int, int]) -> None:
    plain = _pixelated_frame()
    with_boxes = _render(size, _with_highlights(plain))
    without = _render(size, plain)
    layout = ui._make_layout(size)
    for highlight in HIGHLIGHTS:
        box = layout.window_rect(highlight.xyxy)
        colour = ui._HIGHLIGHT_SHOWN if highlight.state == "show" else ui._HIGHLIGHT_BLURRED
        for edge in [(box.left, box.centery), (box.right - 1, box.centery), (box.centerx, box.bottom - 1)]:
            assert tuple(with_boxes.get_at(edge))[:3] == colour
            assert with_boxes.get_at(edge) != without.get_at(edge)
        inside = (box.centerx, box.centery + box.height // 4)
        assert with_boxes.get_at(inside) == without.get_at(inside)  # outline only, no fill


def test_blurred_shown_and_replaced_outlines_differ() -> None:
    assert len({ui._HIGHLIGHT_BLURRED, ui._HIGHLIGHT_SHOWN, ui._HIGHLIGHT_REPLACED}) == 3


def test_replaced_ads_get_their_own_outline_and_tag() -> None:
    size = ui._default_window_size()
    ad = Highlight(id="obj#4", label="billboard", xyxy=(100, 100, 400, 260), state="replace")
    plain = _pixelated_frame()
    screen = _render(size, _with_highlights(plain, (ad,)))
    box = ui._make_layout(size).window_rect(ad.xyxy)
    assert tuple(screen.get_at((box.left, box.centery)))[:3] == ui._HIGHLIGHT_REPLACED
    tag_pixels = pygame.surfarray.array3d(screen.subsurface(pygame.Rect(box.x, box.y - 30, box.width, 30)))
    assert np.all(tag_pixels == ui._HIGHLIGHT_REPLACED, axis=-1).any()  # the tag text is in the ad colour
    assert ui._STATE_STYLES["replace"][1] == "ad"


def test_fail_closed_frames_show_no_highlights() -> None:
    size = (480, 320)
    blacked_out = _black_frame()
    assert pygame.image.tobytes(_render(size, _with_highlights(blacked_out)), "RGB") == \
        pygame.image.tobytes(_render(size, blacked_out), "RGB")


def test_hint_shows_only_until_there_is_a_transcript() -> None:
    size = (480, 320)
    layout = ui._make_layout(size)
    probe = (layout.preview_area.x + layout.gap + 1, layout.preview_area.bottom - 2)
    frame = _with_highlights(_pixelated_frame())
    plain = _render(size, _pixelated_frame())
    assert _render(size, frame).get_at(probe) != plain.get_at(probe)  # hint strip
    talking = UiStatus(transcript=("agent: hi",))
    assert _render(size, frame, talking).get_at(probe) == _render(size, _pixelated_frame(), talking).get_at(probe)


def test_drawing_highlights_never_touches_the_frame_image() -> None:
    frame = _with_highlights(_pixelated_frame())
    before = frame.image.tobytes()
    ui.run(lambda: frame, max_frames=3, window_size=(480, 320))
    _render((1280, 800), frame)
    assert frame.image.tobytes() == before


# --- Talk: the mic never stays open ---------------------------------------------


def test_talk_release_outside_the_button_still_closes() -> None:
    talk = _button_rects((640, 480))["talk"].center
    app = _FakeApp()
    app.run({1: [_mouse_down(talk)], 3: [_mouse_up((10, 10))]})
    assert app.calls == [("talk", True), ("talk", False)]


def test_talk_finger_release_outside_the_button_still_closes() -> None:
    size = (480, 320)
    talk = _button_rects(size)["talk"].center
    app = _FakeApp()
    app.run({1: [_finger(pygame.FINGERDOWN, talk, size)], 3: [_finger(pygame.FINGERUP, (5, 5), size)]}, size=size)
    assert app.calls == [("talk", True), ("talk", False)]


def test_talk_focus_loss_closes() -> None:
    app = _FakeApp()
    app.run({1: [_key_down(pygame.K_t)], 2: [pygame.event.Event(pygame.WINDOWFOCUSLOST)], 3: [_key_up(pygame.K_t)]})
    assert app.calls == [("talk", True), ("talk", False)]


def test_talk_quit_while_held_closes_once() -> None:
    app = _FakeApp()
    app.run({1: [_mouse_down(_button_rects((640, 480))["talk"].center)], 2: [pygame.event.Event(pygame.QUIT)]})
    assert app.calls == [("talk", True), ("talk", False)]


def test_talk_max_frames_while_held_closes_once() -> None:
    app = _FakeApp()
    app.run({1: [_key_down(pygame.K_t)]}, frames=4)
    assert app.calls == [("talk", True), ("talk", False)]


def test_talk_held_by_two_pointers_opens_and_closes_once() -> None:
    app = _FakeApp()
    talk = _button_rects((640, 480))["talk"].center
    app.run({1: [_key_down(pygame.K_t)], 2: [_mouse_down(talk)], 3: [_key_up(pygame.K_t)], 4: [_mouse_up(talk)]})
    assert app.calls == [("talk", True), ("talk", False)]


# --- Panic: on at the press, cleared only by a fresh full hold --------------------


def test_panic_turns_on_at_the_press_and_that_press_never_clears_it() -> None:
    panic = _button_rects((640, 480))["panic"].center
    app = _FakeApp()
    app.run({1: [0.0, _mouse_down(panic)], 2: [0.5], 3: [5.0], 4: [10.0]})
    assert app.calls == [("panic", True)]
    assert app.panic


def test_panic_short_hold_does_not_clear() -> None:
    panic = _button_rects((640, 480))["panic"].center
    app = _FakeApp()
    app.run({
        1: [_key_down(pygame.K_p), _key_up(pygame.K_p)],
        3: [10.0, _mouse_down(panic)],
        4: [10.0 + config.PANIC_HOLD_S * 0.9],
        5: [_mouse_up(panic)],
        6: [20.0],
    })
    assert app.calls == [("panic", True)]
    assert app.panic


def test_panic_full_hold_clears_once() -> None:
    size = (480, 320)
    panic = _button_rects(size)["panic"].center
    app = _FakeApp()
    app.run({
        1: [_finger(pygame.FINGERDOWN, panic, size), _finger(pygame.FINGERUP, panic, size)],
        3: [10.0, _finger(pygame.FINGERDOWN, panic, size)],
        4: [10.0 + config.PANIC_HOLD_S * 0.5],
        5: [10.0 + config.PANIC_HOLD_S],
        6: [30.0],
    }, size=size)
    assert app.calls == [("panic", True), ("panic", False)]
    assert not app.panic


def test_panic_progress_fills_while_holding() -> None:
    controls = ui._Controls(UiCallbacks())
    on = UiStatus(panic=True)
    controls.press("mouse", "panic", on, now=0.0)
    assert controls.panic_progress(config.PANIC_HOLD_S / 2) == pytest.approx(0.5)
    controls.release("mouse")
    assert controls.panic_progress(config.PANIC_HOLD_S / 2) == 0.0


# --- robustness and the preview contract -------------------------------------------


def test_a_raising_callback_does_not_stop_the_loop() -> None:
    def boom() -> None:
        raise RuntimeError("callback failed")

    app = _FakeApp()
    app.run({1: [_key_down(pygame.K_r), _key_up(pygame.K_r)], 2: [_key_down(pygame.K_e)]}, frames=6, on_record_toggle=boom)
    assert app.ticks == 6
    assert app.calls == [("owner",)]


def test_non_censored_frame_is_refused() -> None:
    raw = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        ui.run(lambda: raw, max_frames=2)  # type: ignore[arg-type, return-value]


def test_no_frame_shows_the_starting_placeholder() -> None:
    seen = []

    def get_frame() -> None:
        surface = pygame.display.get_surface()
        preview = ui._make_layout(surface.get_size()).preview_area
        seen.append(tuple(surface.get_at((preview.centerx, preview.bottom - 5)))[:3])
        return None

    ui.run(get_frame, max_frames=3)
    assert seen[1:] == [ui._PLACEHOLDER_BG, ui._PLACEHOLDER_BG]


def test_long_text_is_ellipsised_to_fit() -> None:
    pygame.font.init()
    font = pygame.font.Font(None, 20)
    surface = ui._fit(font, "agent: " + "very long words " * 30, 200)
    assert surface.get_width() <= 200


def test_draw_handles_every_state_at_every_size() -> None:
    pygame.font.init()
    status = UiStatus(recording=True, recording_s=5, fail_closed=True, panic=True, talking=True,
                      transcript=tuple(f"you: line {i} " * 20 for i in range(6)), owner_enrolled=True)
    edge_cases = HIGHLIGHTS + (
        Highlight(id="edge", label="a very long label for a tiny box", xyxy=(0, 0, 3, 3), state="show"),
        Highlight(id="corner", label="face", xyxy=(600, 440, 640, 480), state="blur", is_owner=True),
        Highlight(id="ad", label="advertisement", xyxy=(0, 400, 640, 480), state="replace"),
    )
    for size in SIZES:
        for frame in (_pixelated_frame(), _with_highlights(_pixelated_frame(), edge_cases)):
            for shown in (status, UiStatus()):
                screen = pygame.Surface(size)
                ui._draw(screen, ui._make_layout(size), frame, shown, ui._Controls(UiCallbacks()), now=0.0)


# --- screenshots, only when asked ---------------------------------------------------


def _pixelated_frame() -> CensoredFrame:
    """A synthetic, already pixelated frame: a colour gradient in config.PIXEL_BLOCK blocks."""
    blocks_y, blocks_x = config.FRAME_HEIGHT // config.PIXEL_BLOCK, config.FRAME_WIDTH // config.PIXEL_BLOCK
    yy, xx = np.mgrid[0:blocks_y, 0:blocks_x]
    noise = np.random.default_rng(1).integers(0, 40, (blocks_y, blocks_x))
    small = np.stack([60 + 3 * xx + noise, 90 + 4 * yy + noise, 140 + noise], axis=-1).clip(0, 255).astype(np.uint8)
    image = cv2.resize(small, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return CensoredFrame(image=image, index=0, fail_closed=False)


@pytest.mark.skipif(not os.environ.get(SCREENSHOT_DIR_ENV), reason=f"set {SCREENSHOT_DIR_ENV} to write screenshots")
def test_write_screenshots() -> None:
    out = Path(os.environ[SCREENSHOT_DIR_ENV])
    out.mkdir(parents=True, exist_ok=True)
    pygame.font.init()
    transcript = (
        "you: blur the license plate on the left",
        "agent: Done, plate blurred.",
        "you: show the person next to me, the one in the red jacket by the door",
    )
    panic_controls = ui._Controls(UiCallbacks())
    panic_controls.press("mouse", "panic", UiStatus(panic=True), now=0.0)
    live = _with_highlights(_pixelated_frame())  # one blurred face, the owner's face shown, a blurred sign
    blacked_out = CensoredFrame(image=live.image, index=0, fail_closed=True, highlights=HIGHLIGHTS)  # drawn without
    shots = {
        "idle": (live, UiStatus(), ui._Controls(UiCallbacks())),
        "recording_talking": (
            live,
            UiStatus(recording=True, recording_s=83, talking=True, transcript=transcript, owner_enrolled=True),
            ui._Controls(UiCallbacks()),
        ),
        "panic": (blacked_out, UiStatus(recording=True, recording_s=125, fail_closed=True, panic=True), panic_controls),
    }
    for size in [(800, 480), (480, 320)]:
        layout = ui._make_layout(size)
        for name, (frame, status, controls) in shots.items():
            screen = pygame.Surface(size)
            ui._draw(screen, layout, frame, status, controls, now=config.PANIC_HOLD_S * 0.4)
            pygame.image.save(screen, str(out / f"ui_{name}_{size[0]}x{size[1]}.png"))
