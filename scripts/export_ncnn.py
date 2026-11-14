"""One-off: YOLOE-26n + the 4 fixed classes -> NCNN at 320 px for the Pi (WP5).

    python scripts/export_ncnn.py
"""

from __future__ import annotations

from pathlib import Path


def export() -> Path:
    """Write config.YOLOE_NCNN_MODEL and return its path."""
    raise NotImplementedError("WP5")


if __name__ == "__main__":
    print(export())
