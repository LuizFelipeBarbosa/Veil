# Board

Milestones: 13:00 faces blurred (Mac) | 14:00 owner + signs/plates | 14:45 voice end to end | 15:30 on the Pi | 16:00 leak review, freeze | 17:45 submitted

| WP | Package | Status | Slot | Branch | Blocker |
| --- | --- | --- | --- | --- | --- |
| 0 | Scaffold and contracts | done | 11:20–11:45 | main | — |
| 1 | Capture, basic preview, recorder | merged 11:31 | 11:27–11:31 | wp1-capture | — |
| 2 | Faces, face tracker, pixelation, watchdog | running since 11:27 (70 min box) | 11:27–12:40 | wp2-faces | face clip in tests/data (HUMAN) |
| 7 | Voice: Deepgram Voice Agent and tools | running since 11:27 (120 min box) | 11:27–13:30 | wp7-voice | — |
| 6 | Owner ID, ID-swap check, policy, scene builder | running since 11:32 (60 min box) | 11:32–12:35 | wp6-identity | crossing clip in tests/data (HUMAN) |
| 5 | Objects: YOLOE-26n and NCNN export | queued (WP2's slot) | 13:00–14:00 | wp5-objects | needs WP2; props clip (HUMAN) |
| 3 | Touch app UI | queued (WP6's slot) | 13:45–14:30 | wp3-ui | needs WP1 |
| 8 | Integration (orchestrator) | queued | 14:00–14:45 | main | needs WP1–7 |
| 9 | Pi port (orchestrator + lead) | queued | 14:45–15:30 | main | needs WP8, the Pi |
| 10 | Leak review (read-only) | queued | 15:30–16:00 | — | needs WP9 |
| 11 | README and Devpost text | queued | 16:30–17:15 | wp11-readme | needs WP9 |
