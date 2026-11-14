# privacy-cam
Handheld privacy camera: Raspberry Pi 5 + webcam + touchscreen; built on a MacBook first.
Spec: docs/brief.md (source of truth).
Principles: blur by default, unblur by consent; fail closed; raw frames never leave the device
and are never written to disk.
Never write, show or send a raw frame. Mic audio goes only to Deepgram, only while Talk is held.
Voice: Deepgram Voice Agent API (Flux STT, hosted OpenAI think model, Aura-2 TTS);
the 5 tools run client-side on the device.
Vision: YuNet + SFace + YOLOE-26n with 4 fixed classes (PyTorch on the Mac, NCNN on the Pi).
Portable: PROFILE = "mac" | "pi" in config.py is the only switch. No Mac-only code paths.
Keep it light: no new dependency without a DECIDE.
Key from env: DEEPGRAM_API_KEY. Never commit it.
Shared paths: venv ~/privacy-cam-venv, weights ~/privacy-cam-weights, recordings ~/privacy-cam-recordings.
All tunable numbers live in config.py. Shapes and tool schemas live in contracts.py.
Tests: pytest; test media in tests/data/. Never open the camera, mic or screen in tests.
If a choice isn't in the spec, pick the option that fails closed and say so in your report.

## How to run
- Python: `~/privacy-cam-venv/bin/python` (3.11). Tests: `~/privacy-cam-venv/bin/python -m pytest -q` from the repo or worktree root.
- The key is not in the shell by default: `set -a; source /Users/luiz/Developer/privacy-cam/.env; set +a` in the same command that needs it. Never print it.
- Profile override without editing code: `PRIVACY_CAM_PROFILE=pi`.

## Who owns what
- Orchestrator only: contracts.py, config.py, app.py, pipeline.py, BOARD.md, merges to main.
- Implementers: only the files in their brief. Keep every stub's signature; add helpers inside your own files.
- Raw frames live in `BufferedFrame.image`. The preview and the recorder take `CensoredFrame` only,
  and only vision/render.py and pipeline.fail_closed_frame may build one.
- Time is `time.monotonic()` everywhere and is passed in as `now`, so tests don't sleep.
