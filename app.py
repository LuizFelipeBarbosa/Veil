"""Entry point: starts the threads and runs the pygame UI on the main thread.

    python app.py                      # live camera
    python app.py --source clip.mp4    # play a clip through the same pipeline
    python app.py --say "blur all faces"   # send one typed command to the agent at startup
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import capture
import config
import ui
from contracts import AD_CLASSES, Policy
from pipeline import Pipeline
from recorder import Recorder
from vision.replace import list_assets
from voice.dg_agent import VoiceAgent
from voice.tools import ToolHandlers

log = logging.getLogger(__name__)

_ENV_FILE = Path(__file__).parent / ".env"
_VOICE_READY_TIMEOUT_S = 10.0


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    _load_env_file()

    policy = Policy()
    recorder = Recorder()
    pipeline = Pipeline(capture.open_source(args.source), policy, recorder)

    transcript: deque[str] = deque(maxlen=config.TRANSCRIPT_LINES)
    last_line_at = [float("-inf")]  # time.monotonic() of the newest transcript line
    talk_held = threading.Event()

    def on_transcript(role: str, text: str) -> None:
        log.info("%s: %s", role, text)
        transcript.append(f"{role}: {text}")
        last_line_at[0] = time.monotonic()

    def transcript_opacity() -> float:
        """1 while Talk is held and for a while after the last line, then down to 0."""
        if talk_held.is_set():
            return 1.0
        fading_for = time.monotonic() - last_line_at[0] - config.TRANSCRIPT_VISIBLE_S
        return min(1.0, max(0.0, 1.0 - fading_for / config.TRANSCRIPT_FADE_S))

    ad_assets = list_assets()
    voice = VoiceAgent(
        ToolHandlers(policy, pipeline.scene, ad_assets=ad_assets), on_transcript, use_audio=not args.no_audio
    )

    def cycle_ad() -> None:
        """Key A: off -> first ad image -> next -> ... -> off, for every ad class at once."""
        current = policy.snapshot().replacements.get(AD_CLASSES[0])
        following = ad_assets[ad_assets.index(current) + 1 :] if current in ad_assets else ad_assets
        for label in AD_CLASSES:
            if following:
                policy.set_replacement(label, following[0])
            else:
                policy.set_class_rule(label, "show")
        log.info("ads: %s", following[0] if following else "not replaced")

    def toggle_recording() -> None:
        if recorder.is_recording:
            log.info("saved %s", recorder.stop())
        else:
            log.info("recording to %s", recorder.start())

    def talk(pressed: bool) -> None:
        if pressed:
            talk_held.set()
        else:
            talk_held.clear()
        voice.set_talking(pressed)

    def status() -> ui.UiStatus:
        if args.source is None and not pipeline.is_alive():
            # Fail closed: a frozen preview looks live. Exit so the service manager reopens the camera.
            raise CameraLost("the camera stopped delivering frames")
        latest = pipeline.latest()
        opacity = transcript_opacity()
        return ui.UiStatus(
            recording=recorder.is_recording,
            recording_s=recorder.elapsed_s,
            fail_closed=latest is None or latest.fail_closed,
            panic=policy.snapshot().panic,
            talking=talk_held.is_set(),
            transcript=tuple(transcript) if opacity > 0 else (),  # gone: the pointing hint comes back
            transcript_opacity=opacity,
            owner_enrolled=pipeline.owner_enrolled,
        )

    pipeline.start()
    voice.start()
    if args.say:
        threading.Thread(target=_say_when_ready, args=(voice, args.say), name="say", daemon=True).start()
    try:
        ui.run(
            get_frame=pipeline.latest,
            get_status=status,
            callbacks=ui.UiCallbacks(
                on_record_toggle=toggle_recording,
                on_talk=talk,
                on_panic=policy.set_panic,
                on_enroll_owner=pipeline.enroll_owner,
                on_point=pipeline.toggle_at,
                on_cycle_ad=cycle_ad,
            ),
            max_frames=args.frames,
        )
    finally:
        voice.stop()
        pipeline.stop()
        pipeline.join(timeout=2.0)
        recorder.stop()


class CameraLost(RuntimeError):
    """The live camera went away (unplugged, USB reset) while the app was running."""


def _load_env_file() -> None:
    """Read KEY=VALUE lines from .env next to this file, without overriding the real environment."""
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("'\""))


def _say_when_ready(voice: VoiceAgent, text: str) -> None:
    if not voice.wait_ready(_VOICE_READY_TIMEOUT_S):
        log.warning("voice agent not ready; sending %r anyway (keyword fallback)", text)
    voice.inject_text(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="privacy-cam")
    parser.add_argument("--source", help="video file to play instead of the camera")
    parser.add_argument("--frames", type=int, help="exit after drawing this many frames (smoke tests)")
    parser.add_argument("--say", help="typed command to send to the agent once it is connected")
    parser.add_argument("--no-audio", action="store_true", help="never open the mic or the speaker")
    return parser.parse_args()


if __name__ == "__main__":
    main()
