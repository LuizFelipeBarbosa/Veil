"""Entry point: starts the threads and runs the pygame UI on the main thread.

    python app.py                      # live camera
    python app.py --source clip.mp4    # play a clip through the same pipeline
"""

from __future__ import annotations

import argparse
import logging

import capture
import ui
from contracts import Policy
from pipeline import Pipeline
from recorder import Recorder

log = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")

    policy = Policy()
    recorder = Recorder()
    pipeline = Pipeline(capture.open_source(args.source), policy, recorder)

    def toggle_recording() -> None:
        if recorder.is_recording:
            log.info("saved %s", recorder.stop())
        else:
            log.info("recording to %s", recorder.start())

    def status() -> ui.UiStatus:
        latest = pipeline.latest()
        return ui.UiStatus(
            recording=recorder.is_recording,
            recording_s=recorder.elapsed_s,
            fail_closed=latest is None or latest.fail_closed,
            panic=policy.snapshot().panic,
        )

    pipeline.start()
    try:
        ui.run(
            get_frame=pipeline.latest,
            get_status=status,
            callbacks=ui.UiCallbacks(on_record_toggle=toggle_recording, on_panic=policy.set_panic),
            max_frames=args.frames,
        )
    finally:
        pipeline.stop()
        pipeline.join(timeout=2.0)
        recorder.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="privacy-cam")
    parser.add_argument("--source", help="video file to play instead of the camera")
    parser.add_argument("--frames", type=int, help="exit after drawing this many frames (smoke tests)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
