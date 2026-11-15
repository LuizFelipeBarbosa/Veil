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
from contracts import AD_CLASSES, KNOWN_CLASSES, BufferedFrame, Policy, TrackedBox
from fakes import scenes
from fakes.capture import FileSource
from pipeline import Pipeline, fail_closed_frame
from recorder import Recorder
from vision.replace import AdReplacer

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


def test_policy_starts_with_everything_private_blurred_and_ads_shown() -> None:
    assert Policy().to_dict() == {
        "class_rules": {label: ("show" if label in AD_CLASSES else "blur") for label in KNOWN_CLASSES},
        "track_overrides": {},
        "panic": False,
        "replacements": {},
    }


def test_only_ads_can_be_replaced_and_a_class_rule_turns_it_off() -> None:
    policy = Policy()
    for label in ("face", "license plate", "logo"):
        with pytest.raises(ValueError):
            policy.set_replacement(label, "ad.png")
    with pytest.raises(ValueError):
        policy.set_class_rule("poster", "replace")  # replace goes through set_replacement only
    with pytest.raises(ValueError):
        policy.set_track_override("face#1", "replace")

    policy.set_replacement("poster", "ad.png")
    replaced = policy.snapshot()
    assert replaced.class_rules["poster"] == "replace"
    assert replaced.replacements == {"poster": "ad.png"}

    policy.show_all()  # unblurring everything leaves a replaced ad replaced
    assert policy.snapshot().class_rules["poster"] == "replace"

    policy.set_class_rule("poster", "show")
    off = policy.snapshot()
    assert off.class_rules["poster"] == "show"
    assert off.replacements == {}


def test_showing_the_face_class_is_an_explicit_choice_and_blurring_it_resets() -> None:
    policy = Policy()
    assert policy.snapshot().class_rules["face"] == "blur"  # blurred until someone asks

    policy.set_class_rule("face", "show")
    policy.set_track_override("face#3", "blur")  # hide one person while everyone else is shown
    shown = policy.snapshot()
    assert shown.class_rules["face"] == "show"
    assert shown.track_overrides == {"face#3": "blur"}
    assert shown.face_reset_count == 0

    policy.set_class_rule("face", "blur")
    reset = policy.snapshot()
    assert reset.class_rules["face"] == "blur"
    assert reset.track_overrides == {}
    assert reset.face_reset_count == 1


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


def test_show_all_unblurs_every_class_keeps_blur_choices_and_clears_panic() -> None:
    policy = Policy()
    policy.set_panic(True)
    policy.set_track_override("face#7", "blur")
    policy.show_all()
    shown = policy.snapshot()
    assert shown.class_rules == {label: "show" for label in KNOWN_CLASSES}  # nothing was replaced
    assert shown.track_overrides == {"face#7": "blur"}
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
    assert contracts.TOOL_NAMES == (
        "get_scene", "set_class_rule", "set_track", "set_replacement", "blur_all", "clear_panic", "show_all",
        "start_recording", "stop_recording",
    )
    by_name = {schema["name"]: schema for schema in contracts.TOOL_SCHEMAS}
    assert by_name["set_replacement"]["parameters"]["properties"]["label"]["enum"] == list(AD_CLASSES)
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
        # a third of a second of delay (the lead's choice), and backfill never reaches past the buffer
        assert (mac.FPS, mac.DELAY_BUFFER_FRAMES, mac.BACKFILL_FRAMES, mac.PIXEL_BLOCK) == (15, 5, 4, 16)
        assert mac.BACKFILL_FRAMES < mac.DELAY_BUFFER_FRAMES
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


def test_a_face_in_front_of_a_replaced_ad_is_still_pixelated(tmp_path: Path) -> None:
    """Ads are drawn before censoring: the ad image covers the poster, the face box on top of it is pixelated."""
    ad = np.random.default_rng(2).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    ad[..., :2] = 0  # a noisy, pure-red image: easy to tell from the frame, and not flat
    cv2.imwrite(str(tmp_path / "ad.png"), ad)
    policy = Policy()
    policy.set_replacement("poster", "ad.png")
    pipeline = Pipeline(FileSource(SYNTHETIC_CLIP), policy)
    pipeline._ads = AdReplacer(tmp_path)
    pipeline._watchdog.beat_faces(time.monotonic())
    pipeline._watchdog.beat_objects(time.monotonic())

    raw = np.random.default_rng(1).integers(0, 255, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    poster = TrackedBox("obj#1", "poster", (96, 96, 416, 336), 0.8, frame_index=0, last_seen=0.0)
    face = TrackedBox("face#1", "face", (208, 160, 272, 224), 0.9, frame_index=0, last_seen=0.0)
    frame = BufferedFrame(index=0, timestamp=0.0, image=raw, boxes=[poster, face], faces_done=True)

    censored = pipeline._censor(frame)

    assert not censored.fail_closed
    ad_block = censored.image[104:120, 104:120]  # on the poster, away from the face
    assert (ad_block[..., :2] == 0).all() and ad_block[..., 2].max() > ad_block[..., 2].min()  # the ad, unpixelated
    face_block = censored.image[176:192, 224:240]
    assert (face_block == face_block[0, 0]).all()  # one flat block: pixelated on top of the ad
    assert (censored.image[:90] == raw[:90]).all()  # outside both boxes nothing changed
    assert {h.id: h.state for h in censored.highlights} == {"obj#1": "replace", "face#1": "blur"}


def test_a_sign_box_inside_a_replaced_ad_does_not_pixelate_it_but_faces_and_chosen_blurs_do(tmp_path: Path) -> None:
    ad = np.random.default_rng(2).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ad.png"), ad)
    policy = Policy()
    policy.set_replacement("poster", "ad.png")
    pipeline = Pipeline(FileSource(SYNTHETIC_CLIP), policy)
    pipeline._ads = AdReplacer(tmp_path)
    pipeline._watchdog.beat_faces(time.monotonic())
    pipeline._watchdog.beat_objects(time.monotonic())

    raw = np.random.default_rng(1).integers(0, 255, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    poster = TrackedBox("obj#1", "poster", (96, 96, 416, 336), 0.8, frame_index=0, last_seen=0.0)
    same_panel = TrackedBox("obj#2", "storefront sign", (112, 112, 400, 320), 0.5, frame_index=0, last_seen=0.0)
    elsewhere = TrackedBox("obj#3", "street sign", (480, 368, 560, 432), 0.5, frame_index=0, last_seen=0.0)

    def flat(image: np.ndarray, y: int, x: int) -> bool:
        """Pixelated: a 16x16 patch spans at most 4 blocks, so at most 4 colours. The noise has hundreds."""
        patch = image[y : y + 16, x : x + 16].reshape(-1, 3)
        return len(np.unique(patch, axis=0)) <= 4

    frame = BufferedFrame(index=0, timestamp=0.0, image=raw, boxes=[poster, same_panel, elsewhere], faces_done=True)
    censored = pipeline._censor(frame)
    assert not flat(censored.image, 200, 200)  # the ad, not pixelated by the sign box on the same panel
    assert flat(censored.image, 392, 512)  # a sign outside the ad is still pixelated

    policy.set_track_override("obj#2", "blur")  # the operator's own choice wins
    assert flat(pipeline._censor(frame).image, 200, 200)


def test_a_weak_ad_box_is_not_replaced_until_its_track_is_confirmed(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "ad.png"), np.zeros((90, 160, 3), dtype=np.uint8))
    policy = Policy()
    policy.set_replacement("poster", "ad.png")
    pipeline = Pipeline(FileSource(SYNTHETIC_CLIP), policy)
    pipeline._ads = AdReplacer(tmp_path)
    pipeline._watchdog.beat_faces(time.monotonic())
    pipeline._watchdog.beat_objects(time.monotonic())
    raw = np.random.default_rng(1).integers(1, 255, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)

    def censored_with(conf: float) -> np.ndarray:
        box = TrackedBox("obj#1", "poster", (96, 96, 416, 336), conf, frame_index=0, last_seen=0.0)
        return pipeline._censor(BufferedFrame(index=0, timestamp=0.0, image=raw, boxes=[box], faces_done=True)).image

    assert (censored_with(config.AD_CONFIRM_CONF - 0.1) == raw).all()  # weak and never confirmed: left alone
    assert not (censored_with(config.AD_CONFIRM_CONF) == raw).all()  # confirmed: replaced
    assert not (censored_with(config.AD_CONFIRM_CONF - 0.1) == raw).all()  # and stays replaced when it weakens
