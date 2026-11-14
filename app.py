"""Entry point: starts the threads and runs the pygame UI on the main thread.

    python app.py                      # live camera (after WP1)
    python app.py --source clip.mp4    # play a clip through the same pipeline
"""

from __future__ import annotations

import argparse
import logging

import capture
import ui
from contracts import Policy
from fakes.capture import FileSource
from pipeline import Pipeline


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")

    source = FileSource(args.source) if args.source else capture.open_source(None)
    pipeline = Pipeline(source, Policy())
    pipeline.start()
    try:
        ui.run(get_frame=pipeline.latest, max_frames=args.frames)
    finally:
        pipeline.stop()
        pipeline.join(timeout=2.0)
        source.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="privacy-cam")
    parser.add_argument("--source", help="video file to play instead of the camera")
    parser.add_argument("--frames", type=int, help="exit after drawing this many frames (smoke tests)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
