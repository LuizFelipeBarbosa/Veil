# Privacy Camera: Project Brief

Updated 2026-11-14, 14:40

A handheld privacy camera on a Raspberry Pi 5 with a webcam and a touchscreen. It records video with faces, street signs and license plates already pixelated, so raw footage never exists on disk, and you control it by voice through Deepgram's Voice Agent API. We build it on the MacBook first, then move it to the Pi. Building starts at 11:20, features freeze at 16:00, and the submission is due at 17:45. **This file is the source of truth. `docs/claude-code-plan.md` covers how we build it.**

## What and why

**The problem.** Anyone filming in public captures bystanders who never agreed to be filmed, plus street signs, plates and house numbers that reveal where they are. Blurring afterwards means the raw footage exists first, and raw footage leaks.

**The product.** A camera app on the Pi's touchscreen. It shows a censored live preview, records only censored video, and takes voice commands: "Blur all faces." "Unblur the person on my left, she's with me." Deepgram's Voice Agent hears the command, its hosted OpenAI model decides, and tool calls change a local policy engine. The agent answers out loud and on screen. It can also swap billboards and posters for your own ad images, live. Say "clip that" and it saves the last 30 seconds as its own file.

**Three principles:**

1. **Blur by default, unblur by consent.** Every face except the enrolled owner starts blurred. People are unblurred one at a time.
2. **Fail closed.** If detection stalls or falls behind, the whole frame is pixelated before it is shown or saved.
3. **Raw frames never leave the camera and are never written to disk.** Mic audio goes to Deepgram only while you hold Talk.

**Scope.** Recordings are video only, with no audio track, and there is no livestreaming for now.

**The demo.** Hold the Pi and tap Record. A teammate walks into frame and is blurred; a plate and a street sign on a phone screen are blurred. Hold Talk and say "unblur the person on my left": only she appears. Say "replace the poster with my ad": the poster on the wall becomes your ad. Hold Panic: everything pixelates. Stop, then play the recording back: only censored footage exists.

## How it works

Two loops meet at one policy engine. The **video loop** runs every frame and decides what gets pixelated. The **voice loop** runs only while you hold Talk, and all it can do is change the policy.

1. **Video loop.** Webcam → 15-frame delay buffer (1 s at 15 FPS) → YuNet faces every frame, YOLOE-26 nano (7 fixed classes) on its own thread → ByteTrack IDs, SFace marks the owner → the policy decides blur or show per box → pixelate → the touchscreen preview and, while recording, the video file.
2. **Voice loop.** Hold Talk (on screen, or T on the Mac) → mic audio streams to Deepgram's Voice Agent → release sends `ForceEndTurn` → Flux transcribes → the hosted OpenAI model calls `get_scene`, then the policy tools, which run on the device → Aura-2 confirms out loud and the reply shows on screen.

One Python 3.11 process. Threads share two objects: the `Policy` (behind a lock) and the latest `Scene`.

| Thread | Loop | Writes |
| --- | --- | --- |
| `capture` | Grab frames at 15 FPS into the ring buffer | Ring buffer |
| `vision` | Faces every frame, track, apply policy, pixelate the oldest frame | Censored frame for preview and recorder, `Scene`, watchdog heartbeat |
| `objects` | YOLOE on the newest frame whenever it's free | Object boxes, tagged with their frame index |
| `recorder` | Write censored frames to disk while recording | `recordings/*.avi` |
| `voice` | asyncio loop: Deepgram socket, mic gate, speaker playback, tool handlers | `Policy`, transcript for the screen |
| main thread | pygame touchscreen app: preview, buttons, status, agent transcript | Record state, `Policy.panic` |

```
privacy-cam/
  app.py               # entry point: starts threads, runs the pygame UI on the main thread
  config.py            # every number in "Config defaults" + PROFILE = "mac" | "pi"
  contracts.py         # tracked box, Policy, Scene, tool schemas
  capture.py           # ring buffer; each frame carries its own box list
  recorder.py          # MJPG writer, censored frames only
  ui.py                # pygame screens and touch buttons
  vision/
    faces.py           # YuNet
    identity.py        # SFace owner match and ID-swap re-checks
    objects.py         # YOLOE-26n thread: PyTorch on the Mac, NCNN on the Pi
    tracking.py        # two ByteTrackTracker instances
    policy.py          # Policy, decide(), scene builder
    render.py          # pixelate, padding, hold-over, backfill, watchdog
  voice/
    dg_agent.py        # Deepgram Voice Agent socket, Settings, mic gate, playback
    tools.py           # tool schemas, handlers, keyword parser
  scripts/
    export_ncnn.py     # one-off: YOLOE-26n + 4 classes -> NCNN for the Pi
  tests/data/          # clips and photos for offline tests
```

## Hardware

| Item | Role |
| --- | --- |
| Raspberry Pi 5 with active cooler | The camera's computer: every vision model, the app, recording |
| USB webcam | The camera; its built-in mic if it has one |
| Touchscreen | Preview and buttons: Record, Talk, Panic |
| Speaker or earbud (USB or Bluetooth) | The agent's spoken replies. The Pi 5 has no headphone jack; without one, replies are text on screen only |
| USB-C power bank that can supply 5 V / 5 A | Handheld power |
| MacBook Pro M1 Max | Development machine and the first place everything runs |

## Tech stack

Every vision model runs on the device. Only voice goes to the cloud (Deepgram, which runs the LLM turn too). Each layer has one fallback, used only if it fails its smoke test. Settings differ by `PROFILE` where marked.

| Layer | Choice | Key settings | Fallback |
| --- | --- | --- | --- |
| Camera in | OpenCV `VideoCapture` | Mac: built-in camera. Pi: USB webcam. 640×480 @ 15 FPS | 10 FPS |
| Faces | YuNet, `face_detection_yunet_2023mar.onnx` | Score 0.6, every frame; input 640×480 on Mac, 320×240 on Pi | Score 0.5 |
| Owner ID | SFace, `face_recognition_sface_2021dec.onnx` | Cosine ≥ 0.363 is a match | Tap a face on screen to mark it as the owner |
| Signs, plates | YOLOE-26 nano, 7 fixed classes | Mac: PyTorch on `mps`, 640 px. Pi: NCNN export, 320 px. Own thread, confidence 0.25 | 256 px, or every other frame |
| Tracking | `trackers` `ByteTrackTracker`, one for faces, one for objects | IDs `face#n`, `obj#n` | Hold last boxes 0.5 s |
| Censoring | OpenCV pixelate | 16×16 px blocks, boxes padded 20% | Solid black box |
| App and preview | pygame, fullscreen on the Pi | Buttons: Record, hold Talk, hold Panic 1 s; status dot; agent transcript. Mac keys: R, T, P | — |
| Recording | OpenCV `VideoWriter`, MJPG `.avi` | 640×480 @ 15 FPS into `~/privacy-cam-recordings/`. The Pi 5 has no hardware H.264 encoder, so MJPG keeps the CPU free | Convert to MP4 on the Mac with ffmpeg for sharing |
| Voice | Deepgram Voice Agent API via `deepgram-sdk` | Flux `flux-general-en` with keyterms; OpenAI model hosted by Deepgram, temperature 0; Aura-2 voice; 7 client-side functions; audio only while Talk is held | Keyword parser on the transcript if no tool call within 2 s; own OpenAI key if the hosted model isn't on our plan |
| Audio I/O | `sounddevice` | 16 kHz mono in; agent audio out to headphones (Mac) or speaker (Pi) | Text-only replies on screen |

**Install:** `ultralytics`, `opencv-python`, `trackers`, `pygame`, `sounddevice`, `deepgram-sdk`, `pytest`; on the Pi also `ncnn` and `sudo apt install libportaudio2`. One key: `DEEPGRAM_API_KEY`. Exporting YOLOE freezes its prompt classes, so voice can blur or show the 4 known classes, faces and individual tracks, but it can't add new object types.

## Interfaces

These shapes live in `contracts.py`. Only the orchestrator changes them.

**Tracked box** (vision → policy → render), one per object per frame:

| Field | Type | Example |
| --- | --- | --- |
| `id` | string | `face#7`, `obj#3` |
| `label` | string | `face`, `license plate`, `street sign` |
| `xyxy` | 4 ints, pixels | `[206, 60, 249, 115]` |
| `conf` | float | `0.82` |
| `frame_index` | int | the buffered frame these boxes belong to |
| `last_seen` | timestamp | drives the 0.5 s hold-over |
| `is_owner` | bool | set by SFace, re-checked every second |

**Policy** (agent and buttons → vision), the only state the agent can change:

```json
{
  "class_rules": {"face": "blur", "license plate": "blur", "street sign": "blur",
                  "house number": "blur", "storefront sign": "blur",
                  "billboard": "replace", "poster": "show", "advertisement": "show"},
  "replacements": {"billboard": "coffee.png"},
  "track_overrides": {"face#7": "show"},
  "panic": false
}
```

Decision order per box: panic, then the track override, then owner, then the class rule. Unknown faces blur.

**Scene** (vision → agent), what `get_scene` returns:

```json
{
  "owner": "face#2",
  "tracks": [
    {"id": "face#7", "label": "face", "side": "owner_left", "distance": "near", "state": "blur"},
    {"id": "obj#3", "label": "license plate", "side": "right", "distance": "far", "state": "blur"}
  ],
  "class_rules": {"face": "blur", "license plate": "blur"},
  "panic": false
}
```

Sides are worked out relative to the owner, with one config flag for camera orientation. When the camera faces you, your left is the right side of the image.

**Agent tools**, all client-side functions in the Voice Agent `Settings`:

| Tool | Arguments | Effect |
| --- | --- | --- |
| `get_scene` | none | Returns the Scene above; no state change |
| `set_class_rule` | `label` (one of the known classes), `action` (`blur` or `show`) | Changes a class rule |
| `set_track` | `track_id` (from the scene), `action` | Overrides one track |
| `blur_all` | none | Panic on |
| `clear_panic` | none | Panic off |
| `set_replacement` | `label` (`billboard`, `poster` or `advertisement`), `asset` (a file in `assets/ads/`) | Paints that image over every sign of that class; `set_class_rule` turns it off |
| `save_clip` | `seconds` (default 30, max 60) | Saves the last N seconds of censored video as a clip; no policy change |

**Agent prompt rules:** call `get_scene` before any command about specific people or objects; use only IDs from that scene; if two people fit, ask out loud instead of guessing; if asked to hide a type it doesn't know, say which types it can hide; keep spoken replies under 8 words.

**Hard limits in code, whatever the model says:** `set_class_rule("face", "show")` is refused; `set_class_rule("face", "blur")` also clears every face override; unknown labels are refused; `set_track` rejects IDs not in the live scene.

| You say | Agent does |
| --- | --- |
| "Blur all faces." | `set_class_rule("face", "blur")`, which clears face overrides |
| "Unblur the person on my left, she's with me." | `get_scene()`, then `set_track("face#7", "show")` |
| "Show the street signs again." | `set_class_rule("street sign", "show")` |
| "Blur everything." | `blur_all()` |
| "Unblur the guy." (two candidates) | `get_scene()`, then asks: "The one on your left or the one in the back?" |
| "Hide the logos." | Says it can hide faces, plates, street signs, house numbers and storefronts |

## Live ad replacement

Added at 14:35. The camera finds billboards and posters and paints your own ad over them, in the preview and the recording. This section adds to the stack, policy and tools above.

- **Detect.** Three more YOLOE classes: `billboard`, `poster`, `advertisement`. That makes 7 in total, all in the NCNN export. Their default rule is show.
- **Replace.** For each ad box whose rule is `replace`, find the sign's four corners inside the box and warp the chosen image onto them. If no clean corners are found, fill the box. Corners are smoothed per track, and the 0.5 s hold-over applies.
- **Order.** Ads are drawn before pixelation, so a face in front of a sign stays censored. A person standing in front of a sign gets painted over by the ad; that is a known limit.
- **Your ads.** Image files in `assets/ads/`. The agent's prompt lists the file names.
- **Control.** "Replace the billboards with the coffee ad", "swap to the sneaker ad", "stop replacing ads". On the Mac, key A cycles through off and each image.
- **Fallback.** If YOLOE-26n can't find the demo poster at 1–3 m, match a reference photo of that poster with ORB features and a homography instead.
- **Schedule.** WP12: one implementer, 60 minutes, alongside integration and the Pi port. If the Pi port slips, ads stay Mac-only for the demo. The leak review is not cut.

## Clips

Added at 14:40. "Clip that" saves the last 30 seconds as its own file, whether or not you are recording. This section adds to the stack and tools above.

- **Buffer.** `clipper.py` keeps the last 60 s of censored frames in memory as JPEGs, a few tens of megabytes at 640×480.
- **Save.** On command, a writer thread copies the last N seconds (default 30, maximum 60) to `~/privacy-cam-recordings/clips/` as MJPG, without stalling the preview.
- **Privacy.** The buffer is fed from the same censored output as the recorder, never from the camera, so blurs and replaced ads are baked into every clip.
- **Control.** Voice: "clip that", "save the last 20 seconds" (tool `save_clip`). A Clip button on the touchscreen; key C on the Mac. The agent replies "Clip saved."
- **Not today.** Auto-marking of good moments and vertical export.
- **Schedule.** WP13: one implementer, 40 minutes. It doesn't depend on detection, so it is the lower-risk of the two late additions. If time gets tight, protect in this order: integration, Pi port, leak review, clips, ad replacement.

## Never leaks

Missing a face for one frame is a privacy leak, and a recording makes it permanent. These rules are the spec, and the leak review checks every one.

1. **Faces every frame.** YuNet runs on every frame in the output path and never waits on YOLOE.
2. **Boxes stay with their frame.** Each buffered frame carries its own box list, and the frame going out is pixelated with its own boxes, never a newer frame's.
3. **Padding and hold-over.** Every box grows 20% per side. A lost track stays blurred for 0.5 s.
4. **Backfill.** When a face is first found, the 10 buffered frames before it are blurred too.
5. **Watchdog.** No finished face detection for 300 ms, or no object detection for 1.5 s: the whole frame pixelates in the preview and the recording, and the status turns red.
6. **ID-swap check.** The owner and every unblurred person have their SFace embedding saved and re-checked about once a second. A mismatch blurs the track. Only one track can be the owner.
7. **New tracks start blurred.** An unblurred person who leaves and comes back is a new track and is blurred again.
8. **Panic is local.** Hold Panic (or P): the whole frame pixelates instantly, with no speech, agent or network involved.
9. **Nothing raw anywhere.** The preview, the recording, thumbnails and debug dumps only ever receive censored frames. Operator highlights appear on the preview only, never in recordings.

## Config defaults

Every tunable number lives in `config.py`. These are starting values for the 16:00 tuning pass.

| Setting | Mac | Pi | Why |
| --- | --- | --- | --- |
| Frame size and rate | 640×480, 15 FPS | 640×480, 15 FPS | Same footage on both; light enough for the Pi |
| Delay buffer | 15 frames (1 s) | 15 frames (1 s) | Room for slow detections and backfill |
| Backfill depth | 10 frames | 10 frames | Covers a face found up to 0.7 s late |
| YuNet input | 640×480 | 320×240 | Speed on the Pi |
| Face score | 0.6 | 0.6 | Favors catching faces; false blurs are fine |
| YOLOE runtime | PyTorch `mps`, 640 px | NCNN, 320 px | NCNN is the fast path on ARM |
| YOLOE classes | `license plate`, `street sign`, `house number`, `storefront sign` | same, baked into the export | Location giveaways, all blurred at startup |
| YOLOE confidence | 0.25 | 0.25 | Over-blurring is the safe failure |
| Box padding | 20% per side | 20% per side | Hair, edges, motion |
| Hold-over | 0.5 s | 0.5 s | One missed frame can't flash a face |
| Face watchdog | 300 ms | 300 ms | Whole frame pixelates beyond this |
| Object watchdog | 1.5 s | 1.5 s | Same, for signs and plates |
| Pixel block | 16×16 px | 16×16 px | Unreadable, clearly intentional |
| SFace match | Cosine ≥ 0.363 | Cosine ≥ 0.363 | OpenCV's reference threshold |
| ID-swap re-check | Every 1 s | Every 1 s | Catches a stolen track ID |
| Keyword fallback | 2 s after end of turn | 2 s after end of turn | The voice path never hangs |
| Deepgram keyterms | blur, unblur, pixelate, panic, faces, license plate, street sign, house number | same | Better recognition of command words |
| Recordings folder | `~/privacy-cam-recordings` | `~/privacy-cam-recordings` | Outside the repo |

## Mac now, Pi later

Everything is built on the MacBook first with `PROFILE = "mac"`. Moving to the Pi is `PROFILE = "pi"`, the NCNN model copied over, and one check per device, with no code changes.

| Device | Mac (now) | Pi (later) | Swap check |
| --- | --- | --- | --- |
| Compute | M1 Max | Pi 5 with active cooler | Holds 15 FPS preview for 2 minutes without throttling |
| Camera | Built-in camera | USB webcam | Faces blurred in preview and recording |
| Mic | Built-in mic | Webcam mic or USB mic | A held-Talk command works 5 times out of 5 |
| Speaker | Headphones | USB or Bluetooth speaker, or earbud | Reply audible; otherwise text on screen |
| Buttons | Keys R, T, P plus on-screen buttons | Touchscreen buttons | Each works with a thumb |
| Display | Window | Fullscreen touchscreen | Preview and buttons fit; right orientation |
| Object model | PyTorch YOLOE-26n | NCNN export made on the Mac | Same 4 classes blurred on phone photos |

**Pi prep (you, as soon as it's here):** Raspberry Pi OS 64-bit (Bookworm), cooler on, Wi-Fi or hotspot, `sudo apt install libportaudio2`, Python 3.11 venv, the same pip install plus `ncnn`, then a 10-line OpenCV check that the webcam opens and the touchscreen shows it.

## Build plan

One Fable 5.2 orchestrator writes the contracts, briefs each package, reviews and merges. Opus 5.5 implementers build one package each in their own git worktree, at most 3 at once. Templates and full done-when tests are in `docs/claude-code-plan.md`.

| Time | Milestone | Work packages |
| --- | --- | --- |
| 11:20–11:45 | Scaffold | WP0 (Fable): `CLAUDE.md`, contracts, config, stubs, fakes, `BOARD.md`. You: Deepgram key, macOS camera and mic permissions, test clips |
| 11:45–13:00 | **Faces blurred in the preview and the recording (Mac)** | WP1 capture, preview, recorder; WP2 faces, tracking, pixelation, watchdog; WP7 Deepgram voice agent and tools (runs to 14:00); WP6 owner ID and policy takes WP1's slot |
| 13:00–14:00 | **Owner visible; signs and plates blurred** | WP5 YOLOE-26n thread plus the NCNN export; WP3 touch app UI. Eat while it runs |
| 14:00–14:45 | **Voice command works end to end (Mac)** | WP8 (Fable): wire every thread, live voice test |
| 14:45–15:30 | **Runs on the Pi** | WP9 (Fable + you): copy repo and NCNN model, `PROFILE = "pi"`, fullscreen on the touchscreen, tune for the Pi |
| 15:30–16:00 | **Leak review and fixes** | WP10: a fresh read-only session lists every path a raw frame could reach the screen or disk |
| 16:00–17:00 | Freeze, polish, rehearse | Bug fixes and `config.py` only; rehearse the demo twice on the Pi |
| 17:00–17:45 | **Submit** | WP11 README; record the demo video; Devpost |

## Rules for the day

- **Changes.** A stack choice changes only when it fails its smoke test, and only to the fallback in its row. After 16:00, only bug fixes and `config.py`.
- **Privacy.** Never write, show or send a raw frame. Mic audio goes only to Deepgram, only while Talk is held. When unsure, choose the option that fails closed.
- **Portability.** Nothing Mac-only outside `config.py` profiles: pygame for UI, `sounddevice` for audio, OpenCV for camera, `mps` only through config.
- **Light.** No new dependency without a `DECIDE:` from the lead.
- **Secrets.** `DEEPGRAM_API_KEY` lives in `.env` at the repo root, which git ignores.
- **Testing.** Implementers test on files in `tests/data/`, never the live camera, mic or screen. Only the orchestrator runs timing tests, one at a time.
- **Filming.** Demo with teammates and phone photos or printed props, never strangers.

## Risks and open questions

| Risk | Mitigation |
| --- | --- |
| A face reaches the screen or a recording | The nine Never-leaks rules, plus the 15:30 leak review |
| The Pi CPU can't keep up | NCNN at 320 px, YuNet at 320×240, 10 FPS fallback; the watchdog fails closed if it falls behind |
| The Pi throttles when hot | Active cooler; watch `vcgencmd measure_temp` during the 2-minute check |
| YOLOE misses signs or plates | Test prompt wording on the Mac before exporting ("street sign", "road sign", "sign with text"); re-export takes about a minute |
| NCNN export of YOLOE fails | ONNX or OpenVINO at 320 px; last resort PyTorch at 320 px on the Pi, every other frame |
| No speaker on the Pi | Agent replies appear as text on screen |
| Venue Wi-Fi drops Deepgram | Phone hotspot; recording, blurring and panic never need the network |
| Hosted OpenAI model not on our Deepgram plan | Bring our own OpenAI key |
| Claude usage limits | 3 implementers at once; Fable does WP0 and WP8 itself |

**Open questions:**

- Which hosted OpenAI model is fastest with reliable function calling? Pick it in the first WP7 test.
- Touchscreen size, resolution and orientation?
- Pi 5 RAM (4 or 8 GB), and does the webcam have a mic?
- When does the Pi arrive?

## Sources

- [Ultralytics YOLOE docs](https://docs.ultralytics.com/models/yoloe)
- [Ultralytics Raspberry Pi guide: NCNN and Pi 5 benchmarks](https://docs.ultralytics.com/guides/raspberry-pi)
- [YOLOE export freezes prompt classes](https://daily.dev/posts/yolo26-open-vocabulary-object-detection-with-yoloe-26-rpj3kfesx.md)
- [OpenCV YuNet and SFace tutorial, thresholds](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)
- [supervision ByteTrack deprecation, replaced by trackers](https://supervision.roboflow.com/0.28.0/trackers/)
- [Deepgram Voice Agent settings](https://developers.deepgram.com/docs/configure-voice-agent)
- [Deepgram Voice Agent function calling](https://developers.deepgram.com/docs/voice-agents-function-calling)
- [Deepgram function call response](https://developers.deepgram.com/docs/voice-agent-function-call-response)
- [Deepgram Voice Agent API reference](https://developers.deepgram.com/reference/voice-agent/voice-agent)
- [Deepgram Flux](https://developers.deepgram.com/reference/speech-to-text/listen-flux)
