"""Every tunable number, per profile. Only the orchestrator edits this file.

Spec: docs/brief.md, section "Config defaults". PROFILE is the only switch
between the MacBook and the Pi; set PRIVACY_CAM_PROFILE=pi on the Pi.
"""

from __future__ import annotations

import os
from pathlib import Path

# Leak review, 15:40: Ultralytics posts usage statistics (frame counts, speeds, device; no frames)
# to Google Analytics unless told it is offline. Nothing but mic audio to Deepgram may leave the
# device, so this is set before anything imports ultralytics. The weights are already on disk.
os.environ.setdefault("YOLO_OFFLINE", "1")

PROFILE = os.environ.get("PRIVACY_CAM_PROFILE", "mac")
if PROFILE not in ("mac", "pi"):
    raise ValueError(f"PRIVACY_CAM_PROFILE must be 'mac' or 'pi', got {PROFILE!r}")
_ON_PI = PROFILE == "pi"

# --- shared paths, outside the repo so every worktree sees the same files ---
WEIGHTS_DIR = Path(os.environ.get("PRIVACY_CAM_WEIGHTS", "~/privacy-cam-weights")).expanduser()
RECORDINGS_DIR = Path(os.environ.get("PRIVACY_CAM_RECORDINGS", "~/privacy-cam-recordings")).expanduser()
YUNET_MODEL = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = WEIGHTS_DIR / "face_recognition_sface_2021dec.onnx"
YOLOE_PT_MODEL = WEIGHTS_DIR / "yoloe-26n-seg.pt"
YOLOE_NCNN_MODEL = WEIGHTS_DIR / "yoloe-26n-seg_ncnn_model"

# --- camera and buffer ---
CAMERA_INDEX = 0  # Mac: built-in camera. Pi: the USB webcam.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# The camera rate on both profiles. A Pi 4 sustains about 12-13 FPS of face work on the live
# camera (YOLOE paced by OBJECT_MIN_PERIOD_S, one rotated YuNet pass a frame); the pipeline skips
# the frames it cannot reach, unseen, so the Pi's preview runs at whatever the scene allows.
FPS = 15
# Brief: 1 s. The lead chose a short delay at 13:25: about a third of a second, so a face swap on a
# shown track is visible until the identity check re-blurs it. That check runs every ID_RECHECK_S
# and allows retries, so the leak review puts the worst case near 0.9 s (it was 0 with the 1 s delay).
DELAY_BUFFER_FRAMES = max(2, round(FPS / 3))  # 5 frames (0.33 s)
BACKFILL_FRAMES = DELAY_BUFFER_FRAMES - 1  # frames before a new face that get blurred too

# True when the camera points at the operator (laptop camera): the operator's
# left is then the right side of the image. False for a camera pointing away.
CAMERA_FACES_OPERATOR = not _ON_PI

# --- faces (YuNet) and owner (SFace) ---
YUNET_INPUT_SIZE = (320, 240) if _ON_PI else (640, 480)  # (width, height)
# Rotated YuNet passes per frame besides the upright one, round-robin over 5 angles. All 5 angles
# must come round within HOLD_OVER_S or a tilted face unblurs between sightings: the Mac takes
# 2.5 frames (0.17 s), the Pi 5 detected frames (0.4 s at 12.5 FPS).
FACE_ROTATIONS_PER_CALL = 1 if _ON_PI else 2
FACE_SCORE = 0.4  # Brief: 0.6. Real faces scored 0.3-0.6 in the leak test and went unblurred
# A face track is blurred only once it has scored this at least once. Equal to FACE_SCORE: gate off.
# 0.5 stops a static false hit (a wall pattern) from blurring, but let 2 confident faces through on crossing.mov.
FACE_CONFIRM_SCORE = FACE_SCORE
SFACE_COSINE_MATCH = 0.363
ID_RECHECK_S = 1.0

# --- objects (YOLOE-26n) ---
YOLOE_RUNTIME = "ncnn" if _ON_PI else "pytorch"
YOLOE_DEVICE = "cpu" if _ON_PI else "mps"
YOLOE_IMGSZ = 320 if _ON_PI else 640
YOLOE_CLASSES = (
    "license plate", "street sign", "house number", "storefront sign",  # censored
    "billboard", "poster", "advertisement",  # ads: shown by default, replaceable
)
YOLOE_CONF = 0.25

# --- censoring ---
BOX_PADDING = 0.25  # per side. Brief: 0.20. 0.35 was too much on the live preview
HOLD_OVER_S = 0.5  # a lost track stays blurred this long
PIXEL_BLOCK = 16  # px
# No finished face detection for this long: whole frame pixelates. Brief: 0.3 s. The Pi 4 value must
# cover one frame period (67 ms) plus its slowest detection (about 350 ms with an SFace check).
FACE_WATCHDOG_S = 0.6 if _ON_PI else 0.300
OBJECT_WATCHDOG_S = 1.5
# Shortest time from one YOLOE run's start to the next. On the Pi a back-to-back YOLOE slows
# face detection from about 56 ms to 91 ms a frame; the Mac runs it on the GPU and needs no pause.
OBJECT_MIN_PERIOD_S = 0.5 if _ON_PI else 0.0

# --- ad replacement ---
ADS_DIR = Path(__file__).parent / "assets" / "ads"  # landscape images; the file name is the asset name
AD_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
KEY_CYCLE_AD = "a"  # off -> first asset -> next asset -> ... -> off

# --- scene ---
SCENE_MIN_CONF = 0.6  # weaker boxes are blurred but not offered to the agent, so they can't be unblurred
NEAR_BOX_HEIGHT_FRACTION = 0.25  # a box at least this tall, relative to the frame, is "near"
CENTER_BAND_FRACTION = 0.2  # middle band of the frame that counts as "center" when no owner is in frame

# --- recording ---
RECORDING_FOURCC = "MJPG"
RECORDING_EXTENSION = ".avi"

# --- UI ---
FULLSCREEN = _ON_PI
PANIC_HOLD_S = 1.0
KEY_RECORD = "r"
KEY_TALK = "t"
KEY_PANIC = "p"
TRANSCRIPT_LINES = 3
TRANSCRIPT_VISIBLE_S = 6.0  # the chat stays this long after its last line (and while Talk is held)...
TRANSCRIPT_FADE_S = 1.5  # ...then fades out over this long

# --- voice (Deepgram Voice Agent) ---
DEEPGRAM_API_KEY_ENV = "DEEPGRAM_API_KEY"
MIC_SAMPLE_RATE = 16_000  # mono, linear16
SPEAKER_SAMPLE_RATE = 24_000
DG_LISTEN_MODEL = "flux-general-en"
DG_THINK_PROVIDER = "open_ai"
DG_THINK_MODEL = "gpt-4.1"  # WP7: 20/20 on every run, ~0.6 s to first tool call; gpt-4o-mini scored 14-16/20
DG_THINK_TEMPERATURE = 0
DG_SPEAK_MODEL = "aura-2-thalia-en"
DEEPGRAM_KEYTERMS = (
    "blur", "unblur", "pixelate", "panic", "faces", "license plate", "street sign", "house number",
)
KEYWORD_FALLBACK_S = 2.0  # no tool call this long after end of turn: run the keyword parser
AGENT_REPLY_MAX_WORDS = 8
