"""WP1: R toggles recording and the REC overlay shows. Headless pygame, no real screen."""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"  # before pygame opens a display

import numpy as np
import pygame

import config
import ui
from contracts import CensoredFrame
from ui import UiCallbacks, UiStatus


def _black_frame() -> CensoredFrame:
    image = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    return CensoredFrame(image=image, index=0, fail_closed=True)


def test_r_key_toggles_recording() -> None:
    toggles = []
    ticks = 0

    def get_frame() -> CensoredFrame:
        nonlocal ticks
        ticks += 1
        if ticks in (1, 3):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        if ticks == 2:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t))  # not the record key
        return _black_frame()

    ui.run(get_frame, callbacks=UiCallbacks(on_record_toggle=lambda: toggles.append(1)), max_frames=5)
    assert len(toggles) == 2


def test_rec_dot_shows_only_while_recording() -> None:
    dot_colors = []

    def get_frame() -> CensoredFrame:
        surface = pygame.display.get_surface()  # still holds the previous tick's drawing
        dot_colors.append(tuple(surface.get_at(ui._REC_DOT_CENTER))[:3])
        return _black_frame()

    recording = UiStatus(recording=True, recording_s=75.0)
    ui.run(get_frame, get_status=lambda: recording, max_frames=3)
    assert dot_colors[1:] == [ui._REC_COLOR, ui._REC_COLOR]

    dot_colors.clear()
    ui.run(get_frame, get_status=UiStatus, max_frames=3)
    assert dot_colors[1:] == [(0, 0, 0), (0, 0, 0)]


def test_mm_ss() -> None:
    assert ui._format_mm_ss(0) == "00:00"
    assert ui._format_mm_ss(75.9) == "01:15"
    assert ui._format_mm_ss(3600) == "60:00"
