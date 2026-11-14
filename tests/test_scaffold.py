"""WP0: contracts, config, fakes and the fail-closed pipeline. No camera, mic or screen."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import config
import contracts
from contracts import KNOWN_CLASSES, Policy
from fakes import scenes
from fakes.capture import FileSource
from pipeline import Pipeline, fail_closed_frame
from recorder import Recorder

SYNTHETIC_CLIP = Path(__file__).parent / "data" / "synthetic.mp4"

MODULES = [
    "app", "capture", "recorder", "ui", "pipeline",
    "vision.faces", "vision.identity", "vision.objects", "vision.tracking", "vision.policy", "vision.render",
    "voice.dg_agent", "voice.tools", "scripts.export_ncnn", "fakes.capture", "fakes.scenes",
]


@pytest.mark.parametrize("module", MODULES)
def test_every_module_imports(module: str) -> None:
    importlib.import_module(module)


# --- policy ---


def test_policy_starts_with_everything_blurred() -> None:
    assert Policy().to_dict() == {
        "class_rules": {label: "blur" for label in KNOWN_CLASSES},
        "track_overrides": {},
        "panic": False,
    }


def test_showing_the_face_class_is_refused() -> None:
    policy = Policy()
    with pytest.raises(ValueError):
        policy.set_class_rule("face", "show")
    assert policy.snapshot().class_rules["face"] == "blur"


def test_unknown_labels_and_actions_are_refused() -> None:
    policy = Policy()
    with pytest.raises(ValueError):
        policy.set_class_rule("logo", "blur")
    with pytest.raises(ValueError):
        policy.set_class_rule("street sign", "delete")
    with pytest.raises(ValueError):
        policy.set_track_override("face#1", "delete")


def test_blurring_faces_clears_face_overrides_only() -> None:
    policy = Policy()
    policy.set_track_override("face#7", "show")
    policy.set_track_override("obj#3", "show")
    policy.set_class_rule("face", "blur")
    assert policy.snapshot().track_overrides == {"obj#3": "show"}


def test_show_all_unblurs_objects_and_the_given_people_but_not_the_face_class() -> None:
    policy = Policy()
    policy.set_panic(True)
    policy.show_all(["face#2", "face#7"])
    shown = policy.snapshot()
    assert shown.class_rules == {"face": "blur", **{label: "show" for label in contracts.OBJECT_CLASSES}}
    assert shown.track_overrides == {"face#2": "show", "face#7": "show"}
    assert shown.panic is False


def test_snapshot_does_not_follow_later_changes() -> None:
    policy = Policy()
    before = policy.snapshot()
    policy.set_panic(True)
    policy.set_class_rule("street sign", "show")
    assert before.panic is False
    assert before.class_rules["street sign"] == "blur"
    assert policy.snapshot().panic is True


# --- scene and tools ---


def test_scene_dict_matches_the_brief() -> None:
    scene = scenes.OWNER_AND_FRIEND_LEFT.to_dict()
    assert set(scene) == {"owner", "tracks", "class_rules", "panic"}
    assert scene["owner"] == "face#2"
    assert scene["tracks"][1] == {
        "id": "face#7", "label": "face", "side": "owner_left", "distance": "near", "state": "blur",
    }


def test_the_tools() -> None:
    assert contracts.TOOL_NAMES == ("get_scene", "set_class_rule", "set_track", "blur_all", "clear_panic", "show_all")
    by_name = {schema["name"]: schema for schema in contracts.TOOL_SCHEMAS}
    assert by_name["set_class_rule"]["parameters"]["properties"]["label"]["enum"] == list(KNOWN_CLASSES)
    assert by_name["set_track"]["parameters"]["required"] == ["track_id", "action"]
    assert all("endpoint" not in schema for schema in contracts.TOOL_SCHEMAS)  # client-side


# --- config ---


def test_config_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        monkeypatch.setenv("PRIVACY_CAM_PROFILE", "pi")
        pi = importlib.reload(config)
        assert (pi.YUNET_INPUT_SIZE, pi.YOLOE_RUNTIME, pi.YOLOE_IMGSZ, pi.FULLSCREEN) == ((320, 240), "ncnn", 320, True)

        monkeypatch.setenv("PRIVACY_CAM_PROFILE", "mac")
        mac = importlib.reload(config)
        assert (mac.YUNET_INPUT_SIZE, mac.YOLOE_RUNTIME, mac.YOLOE_DEVICE, mac.YOLOE_IMGSZ) == ((640, 480), "pytorch", "mps", 640)
        assert (mac.FPS, mac.DELAY_BUFFER_FRAMES, mac.BACKFILL_FRAMES, mac.PIXEL_BLOCK) == (15, 15, 10, 16)
        assert mac.YOLOE_CLASSES == contracts.OBJECT_CLASSES

        monkeypatch.setenv("PRIVACY_CAM_PROFILE", "laptop")
        with pytest.raises(ValueError):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# --- fakes and the fail-closed pipeline ---


def test_file_source_plays_and_loops() -> None:
    source = FileSource(SYNTHETIC_CLIP)
    frames = [source.read() for _ in range(40)]  # the clip has 30 frames
    source.release()
    assert all(frame is not None and frame.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3) for frame in frames)


def test_file_source_ends_without_loop() -> None:
    source = FileSource(SYNTHETIC_CLIP, loop=False)
    frames = [source.read() for _ in range(40)]
    source.release()
    assert frames[-1] is None


def _is_pixelated(image: np.ndarray) -> bool:
    """Every 16x16 block is one flat color."""
    block = config.PIXEL_BLOCK
    height, width = image.shape[:2]
    blocks = image[: height - height % block, : width - width % block].reshape(
        height // block, block, width // block, block, 3
    )
    return bool((blocks.max(axis=(1, 3)) == blocks.min(axis=(1, 3))).all())


def test_fail_closed_frame_pixelates_everything() -> None:
    raw = np.random.default_rng(0).integers(0, 255, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    censored = fail_closed_frame(raw, index=3)
    assert censored.fail_closed and censored.index == 3
    assert not _is_pixelated(raw)
    assert _is_pixelated(censored.image)


def test_pipeline_fails_closed_without_detection(tmp_path: Path) -> None:
    """Frames whose face detection never finished come out pixelated, in the preview and on disk."""
    recorder = Recorder(tmp_path)
    recorder.start()
    pipeline = Pipeline(FileSource(SYNTHETIC_CLIP), Policy(), recorder)
    pipeline._detect = lambda frame: (_ for _ in ()).throw(RuntimeError("detector down"))
    pipeline.start()
    deadline = time.monotonic() + 5.0
    while pipeline.latest() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    pipeline.stop()
    pipeline.join(timeout=2.0)
    recording = recorder.stop()

    latest = pipeline.latest()
    assert latest is not None and latest.fail_closed
    assert _is_pixelated(latest.image)

    video = cv2.VideoCapture(str(recording))
    ok, first = video.read()
    video.release()
    assert ok
    # MJPG is lossy, so compare block means against the frame instead of exact flatness
    assert np.abs(first.astype(int) - fail_closed_frame(first, 0).image.astype(int)).mean() < 3
