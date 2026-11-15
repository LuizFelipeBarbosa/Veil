# Board

Milestones: 13:00 faces blurred (Mac) | 14:00 owner + signs/plates | 14:45 voice end to end | 15:30 on the Pi | 16:00 leak review, freeze | 17:45 submitted

| WP | Package | Status | Slot | Branch | Blocker |
| --- | --- | --- | --- | --- | --- |
| 0 | Scaffold and contracts | done | 11:20–11:45 | main | — |
| 1 | Capture, basic preview, recorder | merged 11:31 | 11:27–11:31 | wp1-capture | — |
| 2 | Faces, face tracker, pixelation, watchdog | merged 11:42 as PARTIAL: leak test fails, needs DECIDE on its bar | 11:27–11:42 | wp2-faces | lead: leak-test bar; wall false blur vs weak-face leaks |
| 7 | Voice: Deepgram Voice Agent and tools | merged 12:02; typed command works end to end; think model gpt-4.1 | 11:27–12:02 | wp7-voice | live mic test (HUMAN) |
| 6 | Owner ID, ID-swap check, policy, scene builder | merged 11:37; real-face tests pass | 11:32–11:37 | wp6-identity | — |
| 6b | Identity re-check: 3 strikes before revoking | merged 12:11 | 11:57–12:10 | wp6b-identity-retry | — |
| 5 | Objects: YOLOE-26n and NCNN export | merged 11:59; wired into pipeline 12:01 | 11:52–11:59 | wp5-objects | — |
| 3 | Touch app UI | merged 11:42 | 11:38–11:42 | wp3-ui | — |
| 12 | Faces at an angle: rotation-robust YuNet | merged 12:44 | 12:21–12:43 | wp12-rotated-faces | Pi timing of 3 YuNet passes per frame |
| 13 | Point at a track to blur/unblur; operator highlights | merged 12:31, wired to pipeline.toggle_at | 12:21–12:28 | wp13-point-ui | lead: try it live |
| 14 | Voice: show_all ("unblur everything"), blur/unblur select things | merged 12:53 as PARTIAL: 40/42 live, 2 commands flaky | 12:21–12:52 | wp14-voice-show-all | 'blur her again', 'unblur the guy' sometimes wrong |
| 15 | Remember unblurred people across track changes (session memory, by face) | merged 12:47 | 12:40–12:47 | wp15-remember-people | — |
| 16 | Street signs: only street-name signs blur; stop, one-way and other traffic signs don't | merged 13:01; NCNN at 320 finds no signs on the clip | 12:54–13:00 | wp16-street-name-signs | lead: check intersection clip; Pi sign confidence |
| 17 | Far-away shown faces flip to 'different person' and re-blur | merged 13:19 | 13:05–13:18 | wp17-far-faces | lead: try it live |
| 18 | Phantom faces and big blurred building areas from the rotated face passes | merged 13:36 | 13:21–13:34 | wp18-rotated-false-faces | lead: re-watch intersection clip |
| 19 | Stop sign still blurred (generic sign prompt); far street-name sign missed | merged 13:33; far blue sign still missed on the Mac | 13:21–13:32 | wp19-sign-precision | lead: one-way distractor vs far street sign |
| 20 | Live ad replacement, vision: ad prompts, corner fit and warp (vision/replace.py), NCNN re-export | merged 14:55; tested on synthetic posters only | 14:41–14:52 | wp20-ad-replace | lead: images in assets/ads/, try a real poster |
| 21 | Live ad replacement, voice and UI: set_replacement tool handler, prompt, key A | merged 14:55; live check 46/48 | 14:41–14:54 | wp21-ad-voice-ui | no on-screen Ad button (Pi: voice only) |
| 22 | Replaced ads too jittery: one drawing per sign, state by position, motion filter | merged; lead to re-watch | 15:07– | wp22-ad-smooth | lead: re-watch ad.mp4 |
| 23 | Ads: stale ad after a scene cut, sliding, glass/occluders | merged 15:36; cut fixed, glass only partly (needs masks) | 15:23–15:31 | wp23-ad-cuts-occlusion | lead: re-watch; panel blurred as a sign after the cut |
| 25 | Ads: never stacked, placed by optical flow instead of the detector box; weak ad tracks not replaced until confirmed | merged 16:02, filter 16:08 | 15:47–16:00 | wp25-ad-flow | ad still drawn over a person in front of it (needs masks) |
| 26 | Ghost faces on the bus-stop advert: rotated passes firing on printed ads | merged 16:03 (lead accepted the recall trade) | 15:47–16:00 | wp26-ghost-faces | a face turned past 90° beside a sure upright face can be missed |
| 28 | Voice tools start_recording / stop_recording | merged 16:33; live 52/54, 'Blur her again' asked instead of acting once (known flake since WP14, passed on re-run) | 16:24–16:31 | wp28-recording-tools | — |
| 29 | Wake phrase "hey veil" on-device (vosk), one-turn mic, spoken yes for reveal commands | merged 16:36; unit-tested only | 16:27–16:34 | wp29-wake-phrase | lead: live test with a real voice; Pi untested |
| 30 | Prompt lines for the confirmation flow | merged 16:43; live 53/54 | 16:37–16:41 | wp30-confirm-prompt | 'Blur her again' asked instead of acting in 2 of the last 3 live runs |
| 8 | Integration (orchestrator) | all threads wired 12:03; live voice test pending | 12:03– | main | lead: hold T and speak |
| 9 | Pi port (orchestrator + lead) | queued | 14:45–15:30 | main | needs WP8, the Pi |
| 10 | Leak review (read-only) | done 15:36: no pixel leak; analytics, backfill, mic tail, debug-log fixes merged | 15:23–15:36 | — | residual: swap exposure ~0.9 s, sign labelled as ad is shown |
| 24 | Mic audio sent for ~120 ms after Talk release | merged 15:45 | 15:37–15:41 | wp24-mic-tail | — |
| 11 | README and Devpost text | merged 16:13 | 16:09–16:13 | wp11-readme | lead: fill TODO(lead), edit, submit |
| 27 | Each sign in view shows a different ad image | on main 16:19 (cherry-picked after the author rewrite) | 16:11–16:17 | wp27-different-ads | — |

Code frozen 16:08 (lead: stop at ad occlusion). After this: bug fixes and config.py only.
Public repo `Veil` created and main pushed 16:16 (author address rewritten to the GitHub no-reply one). Later commits are not pushed until the lead says.
Code reopened by the lead for hands-free work; hard stop for code 17:00. No push until the lead says.
WP32 merged 16:58: yes confirmation after a wake phrase is behind config.WAKE_REVEAL_NEEDS_YES, off by the lead's choice.
