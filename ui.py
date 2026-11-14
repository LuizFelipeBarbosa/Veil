"""pygame app (WP1 basic preview, WP3 touch UI). Runs on the main thread.

It only ever receives CensoredFrame. WP0 ships the bare window; buttons, keys,
status dot, transcript and recording timer come with WP1 and WP3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import pygame

import config
from contracts import CensoredFrame


@dataclass(frozen=True)
class UiStatus:
    """What the app shows besides the preview. Read once per UI tick."""

    recording: bool = False
    recording_s: float = 0.0
    fail_closed: bool = False  # red status dot
    panic: bool = False
    talking: bool = False
    transcript: tuple[str, ...] = ()  # newest last, "you: ..." / "agent: ..."


@dataclass(frozen=True)
class UiCallbacks:
    """What the buttons and keys trigger. Called on the main thread; must return fast."""

    on_record_toggle: Callable[[], None] = lambda: None
    on_talk: Callable[[bool], None] = lambda pressed: None  # True on press, False on release
    on_panic: Callable[[bool], None] = lambda on: None


def run(
    get_frame: Callable[[], CensoredFrame | None],
    get_status: Callable[[], UiStatus] = UiStatus,
    callbacks: UiCallbacks = UiCallbacks(),
    max_frames: int | None = None,
) -> None:
    """Show the censored preview until the window closes, Esc is pressed or max_frames were drawn."""
    pygame.init()
    flags = pygame.FULLSCREEN if config.FULLSCREEN else 0
    screen = pygame.display.set_mode((config.FRAME_WIDTH, config.FRAME_HEIGHT), flags)
    pygame.display.set_caption("privacy-cam")
    clock = pygame.time.Clock()

    drawn = 0
    try:
        while max_frames is None or drawn < max_frames:
            if _quit_requested():
                break
            frame = get_frame()
            if frame is not None:
                screen.blit(_to_surface(frame), (0, 0))
            pygame.display.flip()
            drawn += 1
            clock.tick(config.FPS)
    finally:
        pygame.quit()


def _quit_requested() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _to_surface(frame: CensoredFrame) -> pygame.Surface:
    if not isinstance(frame, CensoredFrame):
        raise TypeError("the preview only accepts CensoredFrame")
    rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return pygame.image.frombuffer(rgb.tobytes(), (width, height), "RGB")
