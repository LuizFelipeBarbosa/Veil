# Claude Code plan

One Fable 5.2 orchestrator owns the contracts, the task board and every merge. Opus 5.5 implementers each take one work package, touch only their own files and report back in a fixed format. You are the hardware loop: anything that needs the camera, the mic, the Pi, a touch or a judgment about what's on screen. The spec is the Brief (`docs/brief.md`).

## Roles and the loop

| Role | Who | Owns | Never does |
| --- | --- | --- | --- |
| Lead | You | Decisions, demo, judging calls, the Pi, talking to the orchestrator | Merges by hand while sessions are running |
| Orchestrator | Fable 5.2, one long-lived session on the Mac | `contracts.py`, `config.py`, `app.py`, `BOARD.md`, merges to `main`, smoke tests, the Pi port | Writes module code inside `vision/` or `voice/` |
| Implementer | Opus 5.5, one subagent per work package | The files listed in its brief, and their tests | Edits `contracts.py`, another package's files or live devices |
| Reviewer | Opus 5.5, fresh session, read-only | A written leak review of the privacy path | Edits code |

The loop, repeated all day:

1. You tell the orchestrator what you want next, or it picks the next ready package from `BOARD.md`.
2. The orchestrator writes a brief from the template below and spawns an implementer with it, at most 3 at once.
3. The implementer works in its own git worktree and branch, runs its done-when test, and returns the report.
4. The orchestrator reads the diff and report, merges to `main` in dependency order, runs `pytest` and updates `BOARD.md`.
5. Anything that needs a device or eyes goes to you as a `HUMAN:` line ("hold T and say 'blur everything'").
6. A blocked or failed package gets one retry with a sharper brief. A second failure switches to the fallback in the Brief's tech stack.

## Ground rules

- **Contracts first, then frozen.** From 11:20 to 11:45 the orchestrator alone writes `contracts.py` (tracked box, `Policy`, `Scene`, the 5 tool schemas), `config.py` with both profiles, and stub modules with typed signatures. After that, only the orchestrator changes a contract.
- **One owner per file.** Each brief lists the files the implementer may edit. Everything else is read-only, so merges don't conflict.
- **Worktree per package.** Each implementer gets its own git worktree and branch (`wp2-faces`). The orchestrator merges in dependency order and reruns `pytest` after every merge.
- **Portable from day one.** Code runs on the Mac now and the Pi later with only `PROFILE` changed. No Mac-only paths outside `config.py`.
- **Shared, absolute paths.** The venv (`~/privacy-cam-venv`) and weights (`~/privacy-cam-weights`) live outside the repo, so every worktree uses the same ones. The key comes from the shell environment, loaded from `.env` before launch, because `.env` isn't copied into worktrees.
- **Implementers test on recordings.** They use the clips and photos in `tests/data/` (committed, so every worktree has them), never the live camera, mic or screen. Voice is tested by sending text commands to the agent.
- **One timing run at a time.** Only the orchestrator schedules performance runs.
- **Time boxes.** Every brief has one. At the limit the implementer stops and reports what it has.
- **After 16:00:** only bug-fix packages and `config.py` changes.

## Work packages

Eleven packages, timed to the milestones. At most 3 implementers run at once, and the next package starts as a slot frees up. The nine never-leaks rules in the Brief are part of the done-when for every vision package, and WP10 checks all of them.

| WP | Package | Owns | Needs | Slot | Done when |
| --- | --- | --- | --- | --- | --- |
| 0 | Scaffold and contracts (orchestrator) | `CLAUDE.md`, `.claude/agents/implementer.md`, `contracts.py`, `config.py`, stubs, `app.py`, `fakes/`, `BOARD.md` | — | 11:20–11:45 | Every module imports; `pytest` runs; `python app.py --source clip.mp4` opens a window |
| 1 | Capture, basic preview, recorder | `capture.py`, `recorder.py`, basic preview in `ui.py` | 0 | 11:45–12:30 | Clip or camera shows in the preview at 15 FPS, 1 s behind; R starts and stops an MJPG recording of censored frames only; each buffered frame carries its own box list |
| 2 | Faces, face tracker, pixelation, watchdog | `vision/faces.py`, `vision/tracking.py`, `vision/render.py` | 0 | 11:45–13:00 | Leak test passes: YuNet at score 0.3 over a saved recording finds no unblurred face; a stalled detector pixelates the whole frame |
| 7 | Voice: Deepgram Voice Agent and tools | `voice/dg_agent.py`, `voice/tools.py` | 0 | 11:45–14:00 | 20 text commands via `InjectUserMessage` give the expected client-side calls and `FunctionCallResponse`s; holding Talk streams the mic and release sends `ForceEndTurn`; keyword parser acts on the transcript when no tool call comes in 2 s; reply plays on the audio output and its text reaches the UI |
| 6 | Owner ID, ID-swap check, policy, scene builder | `vision/identity.py`, `vision/policy.py` | 0 | 12:30–13:45 (WP1's slot) | Decision-order tests pass; owner stays visible; on the crossing clip a swapped ID never shows a stranger; `get_scene` puts "the person on my left" on the right track |
| 5 | Objects: YOLOE-26n and the NCNN export | `vision/objects.py`, object tracker in `vision/tracking.py`, `scripts/export_ncnn.py` | 2 | 13:00–14:00 (WP2's slot) | Phone photos or props blurred on the clip; YOLOE on its own thread; `scripts/export_ncnn.py` writes an NCNN model at 320 px with the 4 classes, and `objects.py` loads either format by profile |
| 3 | Touch app UI | `ui.py` (after WP1 merges) | 1 | 13:45–14:30 (WP6's slot) | Fullscreen-ready pygame app: preview, Record, hold Talk, hold Panic 1 s, status dot, agent transcript, recording timer; buttons work by mouse and touch; R, T, P keys on the Mac |
| 8 | Integration (orchestrator) | `app.py` | 1–7 | 14:00–14:45 | All threads live on the Mac; a spoken command changes the preview and the recording within 3 s |
| 9 | Pi port (orchestrator + you) | `config.py` Pi profile, setup notes in `README.md` | 8 | 14:45–15:30 | On the Pi: fullscreen app on the touchscreen, NCNN model loaded, 15 FPS for 2 minutes without throttling (or 10 FPS fallback), a recording plays back censored, one voice command works |
| 10 | Leak review (read-only) | report only | 9 | 15:30–16:00 | Written list of every path where a raw frame can reach the screen or disk; each becomes a bug-fix package |
| 11 | README and Devpost text | `README.md` | 9 | 16:30–17:15 | Setup, architecture and principles match the code; you edit and submit |

## Templates

The orchestrator fills in the brief for every package. The implementer always ends with the report, so the orchestrator can read status without reading the whole transcript.

**Task brief** (orchestrator to implementer):

```markdown
# WP<n>: <name>
Goal: <one sentence>
Read first: CLAUDE.md, contracts.py, config.py, docs/brief.md#<section>
You may edit: <files>. Everything else is read-only.
Need a contract change? Stop and say so in your report. Don't edit contracts.py.
Test inputs: <clips, photos or fixtures in tests/data/>
Done when: <exact command(s) that must pass>
Out of scope: <what not to build>
Time box: <minutes>. At the limit, stop and report.
Branch: wp<n>-<slug>
```

**Report** (implementer to orchestrator):

```markdown
WP<n>: DONE | PARTIAL | BLOCKED
Branch + last commit: <sha>
Works: <what, and the command that shows it>
Measured: <ms per frame, FPS, latency, if relevant>
Not done / known issues: <list>
Contract change requested: none | <field and why>
Needs a human: none | <one-line ask, e.g. "hold T and say 'blur everything'">
```

## Repo setup

```
~/privacy-cam/                 the repo; run Claude Code here
  .env                         DEEPGRAM_API_KEY (git ignores it)
  .gitignore
  docs/brief.md                the project brief (source of truth)
  docs/claude-code-plan.md     this file
  tests/data/                  test clips and phone photos (committed)
  (Fable creates CLAUDE.md, .claude/, contracts.py and the code)
~/privacy-cam-venv/            Python 3.11 environment, outside the repo
~/privacy-cam-weights/         model files, outside the repo
~/privacy-cam-recordings/      censored recordings, outside the repo
```

One-time setup on the Mac:

```bash
mkdir -p ~/privacy-cam/{docs,tests/data} ~/privacy-cam-weights ~/privacy-cam-recordings && cd ~/privacy-cam && git init -q
printf '.env\n*.pt\n*.onnx\n*_ncnn_model/\n__pycache__/\n.DS_Store\n' > .gitignore
printf 'DEEPGRAM_API_KEY=paste-here\n' > .env
python3.11 -m venv ~/privacy-cam-venv && source ~/privacy-cam-venv/bin/activate
pip install -q ultralytics opencv-python trackers pygame sounddevice deepgram-sdk pytest
cd ~/privacy-cam-weights \
  && curl -sLO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  && curl -sLO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx \
  && python -c "from ultralytics import YOLOE; m=YOLOE('yoloe-26n-seg.pt'); n=['license plate']; m.set_classes(n, m.get_text_pe(n))" \
  ; cd ~/privacy-cam
```

Launch Fable with the key loaded, so every subagent inherits it:

```bash
cd ~/privacy-cam && source ~/privacy-cam-venv/bin/activate && set -a && source .env && set +a && claude --model claude-fable-5-2
```

On the Pi later (WP9): Raspberry Pi OS 64-bit, `sudo apt install libportaudio2`, the same venv and pip install plus `ncnn`, then copy the repo, `.env`, the two ONNX face models and the NCNN model folder.

## Session files

Fable writes these two files from the blocks below in WP0.

**`CLAUDE.md`** (every session reads it):

```markdown
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
```

**`.claude/agents/implementer.md`** (the Opus subagents):

```markdown
---
name: implementer
description: Implements one privacy-cam work package from a brief, in its own worktree.
model: opus
---
You implement exactly one work package. Edit only the files your brief lists.
Read CLAUDE.md, contracts.py and the Brief section named in your task brief before writing code.
Test on tests/data/ recordings; never open live devices.
Commit to your branch. Finish with the report template, nothing after it.
If blocked or out of time, stop and report rather than widen scope.
```

Check in the first spawn that `model: opus` resolves to Opus 5.5.

## Starter prompt

Paste into the Fable session once it starts. It works from an empty repo.

```markdown
It's Saturday ~11:20 at VexoHacks (Y Combinator, SF). We start privacy-cam from an empty repo now.
Feature freeze 16:00, submission due 17:45. No stretch goals.

You are the orchestrator. You plan, brief, review and merge. Implementers are Opus subagents
(.claude/agents/implementer.md), each in its own git worktree, at most 3 at once.
You don't write code in vision/ or voice/.

Read first: docs/brief.md (the spec, source of truth) and docs/claude-code-plan.md
(work packages, templates, CLAUDE.md and implementer blocks). If they disagree with this prompt,
the docs win; tell me.

The product: a handheld privacy camera app. Target hardware is a Raspberry Pi 5 with a USB webcam
and a touchscreen; we build and test on this MacBook Pro M1 Max first, then port to the Pi at 14:45.
It shows a censored live preview, records censored video only (MJPG), and takes voice commands
through Deepgram's Voice Agent API. No livestreaming.
Principles: blur by default, unblur by consent; fail closed; raw frames never leave the device and
are never written to disk.

Step 1, WP0, now until 11:45, yours alone, committed to main:
- .gitignore already ignores .env, weights, NCNN models and caches; keep it that way
- CLAUDE.md and .claude/agents/implementer.md from the blocks in docs/claude-code-plan.md
- contracts.py: tracked box, Policy, Scene and the 5 tool schemas (get_scene, set_class_rule,
  set_track, blur_all, clear_panic), exactly as the Brief defines them
- config.py: every value in "Config defaults" with PROFILE = "mac" | "pi" (default "mac"),
  plus the shared weights and recordings paths
- stubs for every module in the Brief's repo layout, typed signatures that raise NotImplementedError
- fakes/: a capture source that plays a video file, and fixture scenes for the agent
- requirements.txt: ultralytics, opencv-python, trackers, pygame, sounddevice, deepgram-sdk, pytest
  (Pi adds ncnn)
- pytest passing; python app.py --source <clip> starts and opens a window
- BOARD.md: WP | status | slot | branch | blocker, for all packages
Ask me with a HUMAN: line for anything you need on the machine (keys, permissions, clips, weights).

Step 2, 11:45: send me the briefs for WP1, WP2 and WP7 in one message. On my "go", spawn them.
WP6 takes WP1's slot, WP5 takes WP2's, WP3 takes WP6's. Then run the loop: read each report and
diff, merge in dependency order, run pytest, update BOARD.md, tell me in 2 lines.

Milestones. Run `date` before planning each step; if one will slip, tell me 30 minutes ahead and
propose what to cut:
13:00 faces blurred in preview and recording (Mac) | 14:00 owner visible, signs and plates blurred |
14:45 voice command works end to end (Mac) | 15:30 runs on the Pi | 16:00 leak review done, freeze |
17:45 submitted

How to talk to me:
- Status: 2 lines per event (merged, blocked, failed), no recaps.
- Anything that needs a device or my eyes: start the line with HUMAN: and say exactly what to do.
- Decisions only I can make: start with DECIDE: and give 2 options plus your pick.

Hard rules:
- Mac devices for now: built-in camera and mic, headphones; keys R record, T hold to talk,
  P hold 1 s for panic, plus the same buttons on screen. The Pi swaps in through PROFILE.
- The pygame UI runs on the main thread. The preview and recordings only ever get censored frames.
- YOLOE-26 nano with 4 fixed classes: PyTorch on mps on the Mac, NCNN at 320 px on the Pi.
- Never write, show or send a raw frame. Mic audio goes only to Deepgram, only while Talk is held.
- The only key is DEEPGRAM_API_KEY, already in the environment. Never commit it.
- No new dependency without a DECIDE: line. When unsure, choose the option that fails closed.
```
