"""WP1 + WP3: the touch UI. Headless pygame, no real screen; time comes from a fake clock."""

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
from contracts import CensoredFrame
from ui import UiCallbacks, UiStatus

SIZES = [(640, 480), (480, 320), (1280, 800), (800, 480), ui._default_window_size()]
SCREENSHOT_DIR_ENV = "PRIVACY_CAM_UI_SCREENSHOTS"  # set to a directory to write the PNGs


def _black_frame() -> CensoredFrame:
    image = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    return CensoredFrame(image=image, index=0, fail_closed=True)


def _button_rects(size: tuple[int, int]) -> dict[str, pygame.Rect]:
    pygame.font.init()
    return dict(ui._make_layout(size).buttons)


# --- event builders ---------------------------------------------------------


def _key_down(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def _key_up(key: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYUP, key=key)


def _mouse_down(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1, touch=False)


def _mouse_up(pos: tuple[int, int]) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1, touch=False)


def _finger(kind: int, pos: tuple[int, int], size: tuple[int, int], finger_id: int = 0) -> pygame.event.Event:
    x, y = pos[0] / size[0], pos[1] / size[1]
    return pygame.event.Event(kind, x=x, y=y, dx=0.0, dy=0.0, touch_id=1, finger_id=finger_id, pressure=1.0)


def _press_release(kind: str, pos: tuple[int, int], size: tuple[int, int]) -> list[pygame.event.Event]:
    if kind == "mouse":
        return [_mouse_down(pos), _mouse_up(pos)]
    return [_finger(pygame.FINGERDOWN, pos, size), _finger(pygame.FINGERUP, pos, size)]


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
def test_press_on_preview_does_nothing(kind: str) -> None:
    app = _FakeApp()
    app.run({1: _press_release(kind, (50, 200), (640, 480))})
    assert app.calls == []


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
    for size in SIZES:
        screen = pygame.Surface(size)
        ui._draw(screen, ui._make_layout(size), _pixelated_frame(), status, ui._Controls(UiCallbacks()), now=0.0)


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
    size = (800, 480)
    layout = ui._make_layout(size)
    transcript = (
        "you: blur the license plate on the left",
        "agent: Done, plate blurred.",
        "you: show the person next to me, the one in the red jacket by the door",
    )
    panic_controls = ui._Controls(UiCallbacks())
    panic_controls.press("mouse", "panic", UiStatus(panic=True), now=0.0)
    shots = {
        "idle": (UiStatus(), ui._Controls(UiCallbacks())),
        "recording_talking": (
            UiStatus(recording=True, recording_s=83, talking=True, transcript=transcript, owner_enrolled=True),
            ui._Controls(UiCallbacks()),
        ),
        "panic": (UiStatus(recording=True, recording_s=125, fail_closed=True, panic=True), panic_controls),
    }
    for name, (status, controls) in shots.items():
        screen = pygame.Surface(size)
        ui._draw(screen, layout, _pixelated_frame(), status, controls, now=config.PANIC_HOLD_S * 0.4)
        pygame.image.save(screen, str(out / f"ui_{name}.png"))
