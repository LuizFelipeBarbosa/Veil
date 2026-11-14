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
| 8 | Integration (orchestrator) | all threads wired 12:03; live voice test pending | 12:03– | main | lead: hold T and speak |
| 9 | Pi port (orchestrator + lead) | queued | 14:45–15:30 | main | needs WP8, the Pi |
| 10 | Leak review (read-only) | queued | 15:30–16:00 | — | needs WP9 |
| 11 | README and Devpost text | queued | 16:30–17:15 | wp11-readme | needs WP9 |
