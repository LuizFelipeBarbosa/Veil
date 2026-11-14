"""One-off: YOLOE-26n + the 4 fixed classes -> NCNN at 320 px for the Pi (WP5).

    python scripts/export_ncnn.py

Needs the `ncnn` and `pnnx==20260526` packages (ultralytics' NCNN exporter pins that pnnx).
The model is the detect-only head with vision.objects.PROMPTS baked in as its classes,
distractor prompts included: vision.objects drops their boxes at run time, on both runtimes.
tests/test_objects.py checks that the export's names equal PROMPTS.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from vision.objects import NCNN_IMGSZ, load_prompted_model  # noqa: E402


def export() -> Path:
    """Write config.YOLOE_NCNN_MODEL and return its path."""
    target = Path(config.YOLOE_NCNN_MODEL)
    # ultralytics names the folder after the loaded .pt and writes it next to it.
    exported = Path(load_prompted_model().export(format="ncnn", imgsz=NCNN_IMGSZ, device="cpu"))
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), target)
    return target


if __name__ == "__main__":
    print(export())
