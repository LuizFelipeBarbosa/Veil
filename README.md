# privacy-cam

A handheld privacy camera. A Raspberry Pi with a USB webcam and a touchscreen shows a live
preview and records video in which faces, license plates, street-name signs, house numbers and
storefront signs are already pixelated. The uncensored image is never shown, never saved and
never sent anywhere. You control it with on-screen buttons, by tapping a box on the preview, and
by voice: hold Talk and say "unblur the person on my left", and Deepgram's Voice Agent turns that
into a tool call that runs on the device. It can also replace billboards and posters with your
own ad images, live. Everything was developed and tested on a MacBook first; one setting switches
it to the Pi.

Built at VexoHacks. Team: TODO(lead). Demo video: TODO(lead).

## Principles

- **Blur by default, unblur by consent.** Every face and every private object starts pixelated.
  Something is shown only after someone asks for it (a tap, a voice command) or it is the
  enrolled owner.
- **Fail closed.** When the software cannot be sure, the whole frame is pixelated: before the
  first detection, when a detector falls behind, when anything raises, and in panic.
- **Raw frames never leave the device and are never written to disk.** Only censored frames
  reach the screen and the recording file.
- **Mic audio goes only to Deepgram, and only while Talk is held.**

## What it does

| Feature | How it works |
| --- | --- |
| Faces | Pixelated by default, on every frame (YuNet, plus rotated passes for tilted heads). |
| Owner | Press **Set owner** (key **E**): the largest confident face becomes the owner and is shown. Re-checked every second with SFace. Kept in memory only, so it is set again after each start. |
| Remembered people | Someone you unblurred stays unblurred when their track is lost and comes back (up to 16 people, this session only). "Blur all faces" forgets them all. |
| Objects | License plates, street-name signs, house numbers and storefront signs are pixelated (YOLOE-26n with text prompts). Stop, one-way and other traffic signs are deliberately left alone. |
| Tap to toggle | Outlines on the preview show what you can tap: blue "blurred", orange "shown", purple "ad". A tap flips blur and show for that one track. Outlines are drawn on the screen only, never into the recording. |
| Voice | Hold **Talk** (key **T**). Deepgram Voice Agent with 9 client-side tools (below). The reply is spoken and shown as a 3-line transcript. |
| Panic | **Panic** (key **P**) pixelates the whole frame at the press, with no hold. Clearing it needs a fresh press held for 1 s (or "clear panic"). |
| Recording | **Record** (key **R**) writes only censored frames, as MJPG `.avi` at 640x480, 15 FPS, video only (no audio track), to `~/privacy-cam-recordings/privacy-cam-YYYYMMDD-HHMMSS.avi`. |
| Live ad replacement | **Ads** button (key **A**) or voice. Billboards, posters and advertisements are shown by default; switched on, each one is covered with an image from `assets/ads/` (each press goes to the next image, then off). When several ads are in view, the chosen image goes on the first and the others get the next images in the folder, so no two show the same one. The sign's corners are found and the image is warped onto them, then followed from frame to frame with optical flow. Ads are drawn before censoring, so a face in front of an ad is still pixelated on top. |

### Voice tools and example phrases

The agent can call exactly these tools (`contracts.TOOL_SCHEMAS`): `get_scene`, `set_class_rule`,
`set_track`, `set_replacement`, `blur_all`, `clear_panic`, `show_all`, `start_recording`,
`stop_recording`. Phrases from the prompt and
the live check (`scripts/voice_check.py`):

- "Blur all faces." / "Unblur everyone."
- "Unblur the person on my left, she's with me."
- "Blur my face too."
- "Hide license plates." / "Show the street signs again."
- "Unblur the guy." with two people in view: it asks "Which one, left or right?" and acts on the answer.
- "Unblur everything." / "Blur everything." / "Clear panic." / "Reset."
- "Replace the ads." / "Stop replacing the ads."
- "Start recording." / "Stop recording.": the same recorder as the Record button, censored frames only.
- "Hide the tattoos.": not a known type, so it says what it can hide instead.

If the model has made no tool call 2 s after you let go of Talk, a local keyword parser handles
class-wide, panic, show-everything, ad and recording commands. Anything about one specific person is left to
the model.

## How it stays private

```
 camera ──> capture thread ──> delay ring buffer (5 frames, 0.33 s; RAW, in memory only)
                                    │                         │
                       newest frame │                         │ oldest frame
                                    v                         v
             vision thread: YuNet faces + ByteTrack      draw replaced ads (still raw)
             + SFace owner / ID re-check                      │
             objects thread: YOLOE-26n on the newest frame    v
                                    │                  render.censor: pixelate every box not
             boxes attached to each buffered frame,    explicitly "show"; whole frame if panic,
             backfilled onto frames still waiting ───> watchdog tripped or detection unfinished
                                                              │
                                                              v
                                                        CensoredFrame ──> pygame preview
                                                                     └──> recorder (.avi)
```

- **Threads.** Capture, vision (faces, tracking, identity, censoring), objects (YOLOE), recorder
  (only while recording), voice (an asyncio loop for the Deepgram socket, plus a playback thread),
  and the pygame UI on the main thread.
- **Delay buffer.** Frames wait 5 frames (a third of a second at 15 FPS) before they are shown.
  Detection runs on the newest frame, so a face found late is also pixelated on the frames
  still waiting (backfill). A lost track stays pixelated for 0.5 s (hold-over). When detection
  is slower than the camera, the frames it could not reach are dropped unseen, never shown.
- **Watchdogs.** No finished face detection for 0.3 s (Pi: 0.6 s) or no object result for
  1.5 s pixelates the whole frame. A frame whose detection failed, or a censoring error, does
  the same. If the live camera stops delivering frames, the app exits rather than show a frozen
  picture.
- **One output type.** The preview and the recorder accept only `CensoredFrame`, and only
  `vision/render.py` and `pipeline.fail_closed_frame` build one. The recorder raises on
  anything else.
- **The agent sees no pixels.** `get_scene` returns ids, labels, sides (relative to the owner,
  or left/center/right), near/far and current state. Only faces scoring 0.6 or more are offered,
  so a weak detection can be blurred but never unblurred by voice. `set_track` refuses an id
  that is not in the live scene.
- **Nothing else leaves.** Ultralytics' usage analytics are switched off (`YOLO_OFFLINE=1`, set
  in `config.py` before ultralytics is imported), and the YOLOE text encoder is loaded from the
  weights folder rather than downloaded. The Deepgram settings set `mip_opt_out`. The
  `websockets` logger is kept at WARNING because its debug output would include the API key.

## Models and stack

| Part | What |
| --- | --- |
| Faces | YuNet (`face_detection_yunet_2023mar.onnx`) via OpenCV; 640x480 on the Mac, 320x240 on the Pi |
| Owner and ID re-check | SFace (`face_recognition_sface_2021dec.onnx`) via OpenCV, cosine match 0.363 |
| Objects | YOLOE-26n with fixed text prompts, via `ultralytics`: PyTorch on MPS at 640 px on the Mac, NCNN at 320 px on the Pi |
| Tracking | ByteTrack from the `trackers` package (with `supervision` detections) |
| Voice | Deepgram Voice Agent over its websocket: Flux STT (`flux-general-en`), hosted OpenAI think model (`gpt-4.1`, temperature 0), Aura-2 TTS (`aura-2-thalia-en`) |
| Audio, UI | `sounddevice` (mic 16 kHz, speaker 24 kHz), `pygame` |

## Setup and run (Mac)

Python 3.11.

```bash
python3.11 -m venv ~/privacy-cam-venv
source ~/privacy-cam-venv/bin/activate
pip install -r requirements.txt
```

**Weights.** They live outside the repo in `~/privacy-cam-weights` (override with
`PRIVACY_CAM_WEIGHTS`). No script in the repo fetches them; download them yourself:

| File | Used by |
| --- | --- |
| `face_detection_yunet_2023mar.onnx` | faces (both profiles) |
| `face_recognition_sface_2021dec.onnx` | owner and ID re-check (both profiles) |
| `yoloe-26n-seg.pt` | objects on the Mac, and the NCNN export |
| `mobileclip2_b.ts` | YOLOE text encoder (Mac, and the NCNN export); the app refuses to download it itself |
| `yoloe-26n-seg_ncnn_model/` | objects on the Pi; made by `scripts/export_ncnn.py` |

The commands used during the build (from `docs/claude-code-plan.md`):

```bash
mkdir -p ~/privacy-cam-weights && cd ~/privacy-cam-weights
curl -sLO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sLO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
# ultralytics downloads yoloe-26n-seg.pt and mobileclip2_b.ts into the current folder
python -c "from ultralytics import YOLOE; m=YOLOE('yoloe-26n-seg.pt'); n=['license plate']; m.set_classes(n, m.get_text_pe(n))"
```

**Key.** Put `DEEPGRAM_API_KEY` in the environment, or in a `.env` file next to `app.py`
(`DEEPGRAM_API_KEY=...`). `.env` is in `.gitignore`; never commit it. Without a key the camera
still works and voice reports "Voice off".

**Run.**

```bash
python app.py                          # live camera
python app.py --source clip.mp4        # play a clip through the same pipeline
python app.py --say "blur all faces"   # send one typed command once the agent is connected
python app.py --no-audio               # never open the mic or the speaker
python app.py --frames 100             # exit after drawing 100 frames (smoke test)
```

| Key | Button | Action |
| --- | --- | --- |
| R | Record | start / stop recording |
| T (hold) | Hold to talk | mic open while held |
| P | Panic | press: whole frame pixelated; press and hold 1 s: clear |
| E | Set owner | enrol the largest confident face as the owner |
| A | Ads | next ad image, then off |
| Esc | | quit |

Ad images: any `.png`, `.jpg` or `.jpeg` in `assets/ads/` (the repo ships `deepgram.jpg`,
`openai.jpg`, `vexoai.png`). The file name is the name the agent uses.

**Tests.**

```bash
python -m pytest -q
```

The tests never open the camera, mic or screen. `tests/data/synthetic.mp4` is in the repo. The
leak tests (`tests/test_leak.py`) and several face, identity and object tests need the lead's own
recordings (`faces.mov`, `crossing.mov`, `intersection.mov`, `ad.mp4`), which are not in the repo,
and skip without them; tests that need weights or the NCNN export skip when those are missing.
The live voice check needs the key and the network:
`python scripts/voice_check.py`.

## Raspberry Pi

```bash
PRIVACY_CAM_PROFILE=pi python app.py
```

`PRIVACY_CAM_PROFILE` is the only switch (`mac` by default). What it changes in `config.py`:

| Setting | Mac | Pi |
| --- | --- | --- |
| Camera rate (`FPS`) | 15 | 15 (frames detection cannot reach are skipped unseen) |
| YuNet input | 640x480 | 320x240 |
| Rotated YuNet passes per frame | 2 | 1 |
| YOLOE runtime | PyTorch, `mps`, 640 px | NCNN, `cpu`, 320 px |
| Pause between YOLOE runs | none | 0.5 s (objects held over 0.5 s longer to match) |
| Face watchdog | 0.3 s | 0.6 s |
| Window | resizable window | fullscreen |
| Camera faces the operator (for "left"/"right") | yes (laptop camera) | no (camera points away) |

Make the NCNN model on the Mac and copy the folder to the Pi's weights folder:

```bash
pip install ncnn pnnx==20260526
python scripts/export_ncnn.py      # writes ~/privacy-cam-weights/yoloe-26n-seg_ncnn_model
```

On the Pi: Python 3.11 venv, `pip install -r requirements.txt` plus `ncnn`, and
`sudo apt install libportaudio2`. Copy the two ONNX face models, the NCNN folder and `.env`.

The Pi numbers in `config.py` were measured on a Pi 4, according to its comments (about 12-13 FPS
of face work on the live camera).

TODO(lead): Pi demo status, and which Pi the demo runs on.

## Known limits

- **Short delay.** The delay is a third of a second, by the lead's choice. If someone swaps in
  front of a shown face's track, the new face can be visible for up to about 0.9 s before the
  identity check (every 1 s, with retries) pixelates it again.
- **Leak test bar.** On the two test clips no face scoring 0.6 or more is left unpixelated.
  Weaker faces are left unpixelated on up to about 8% of frames (28 of 347 and 18 of 427; the
  test's limit is 10%), mostly in the first frames before a face is first detected.
- **Small faces.** Faces under 32 px tall are not identity-checked.
- **Rotated faces.** Beside a clearly upright face, faces turned past 90 degrees can be missed:
  those rotated passes are skipped then, to stop false faces on posters.
- **Signs.** Ads default to shown, so a sign the model labels as an ad is shown. A street-name
  sign far away can be missed. Stop, one-way and other traffic signs are deliberately not
  blurred.
- **Ad occlusion.** An ad is drawn over a person or object standing in front of it (a face is
  still pixelated on top). Fixing this needs segmentation masks.
- **Voice ambiguity.** The agent can occasionally act on an ambiguous command ("unblur the guy"
  with two candidates) instead of asking.

## Devpost text

**Inspiration.** Anyone filming in public captures bystanders who never agreed to be filmed,
plus street signs, plates and house numbers that reveal where they are. Blurring afterwards
means the raw footage exists first, and raw footage leaks. We wanted a camera where the raw
footage never exists at all. TODO(lead): add a personal angle if you have one.

**What it does.** privacy-cam is a handheld camera (Raspberry Pi, USB webcam, touchscreen) that
only ever shows and records censored video: faces, license plates, street-name signs, house
numbers and storefront signs are pixelated by default. You unblur people by tapping them or by
voice ("unblur the person on my left, she's with me") through Deepgram's Voice Agent, and a
panic button pixelates everything at once. It can also replace billboards and posters with your
own ad images, live, while faces in front stay pixelated.

**How we built it.** Python with OpenCV (YuNet faces, SFace owner match), YOLOE-26n with text
prompts for plates and signs (PyTorch on the Mac, NCNN on the Pi), ByteTrack, pygame for the
touch UI, and the Deepgram Voice Agent API (Flux STT, a hosted OpenAI model, Aura-2 TTS) with 7
tools that run on the device. Raw frames wait in a short in-memory delay buffer so late
detections can be backfilled, and only a `CensoredFrame` type can reach the screen or the
recorder. We built it with Claude Code: an orchestrator session owned the shared contracts,
config and integration, and handed about 25 work packages to implementer agents, each in its
own git worktree and branch (`BOARD.md` lists them).

**Challenges we ran into.** Real faces often scored below the detector's usual threshold, and
lowering it brought false faces on walls and posters, so we tuned the trade-off against a leak
test on our own clips. Tilted faces needed rotated detection passes, which then found faces in
building facades and printed ads. Keeping replaced ads steady on a moving sign took optical-flow
tracking. The fail-closed rules (watchdogs, whole-frame pixelation) had to hold while the Pi
fell behind the camera.

**What we learned.** "Never leak a frame" is a design rule, not a filter: it shaped the types,
the threads and the order of every step. A read-only leak review late in the day still found
things to fix, such as library analytics and a short tail of mic audio after Talk was released.

**What's next.** Segmentation masks so ads are not drawn over people in front of them, saving
the last seconds as a clip, and testing more on the Pi. TODO(lead).
