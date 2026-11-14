"""WP7 live check: 20 typed commands through Deepgram's Voice Agent against the fixture scenes.

    set -a; source .env; set +a; python scripts/voice_check.py [--model gpt-4.1-mini ...]

Text only (use_audio=False): no mic or speaker is opened. Each command gets a fresh
Policy and a fresh agent session, so no command sees another's history. The keyword
fallback is off, so the score measures the model alone. Exits nonzero below 20/20.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from contracts import Policy, Scene  # noqa: E402
from fakes import scenes  # noqa: E402
from voice.dg_agent import VoiceAgent  # noqa: E402
from voice.tools import ToolHandlers  # noqa: E402

GET_SCENE = ("get_scene", {})
STATE_TOOLS = {"set_class_rule", "set_track", "blur_all", "clear_panic"}
CAN_HIDE_WORDS = ("face", "plate", "street", "house", "storefront")


@dataclass(frozen=True)
class Case:
    scene: str  # key in fakes.scenes.ALL
    text: str
    expected: tuple[tuple[str, dict[str, Any]], ...]  # in order; get_scene listed only where required
    asks: bool = False  # must reply with a question, no state change
    lists_types: bool = False  # must reply naming what it can hide, no state change


def _rule(label: str, action: str) -> tuple[str, dict[str, Any]]:
    return ("set_class_rule", {"label": label, "action": action})


def _track(track_id: str, action: str) -> tuple[str, dict[str, Any]]:
    return ("set_track", {"track_id": track_id, "action": action})


CASES = [
    Case("owner_and_friend_left", "Unblur the person on my left, she's with me.", (GET_SCENE, _track("face#7", "show"))),
    Case("two_candidates", "Unblur the guy.", (GET_SCENE,), asks=True),
    Case("empty", "Hide the logos.", (), lists_types=True),
    Case("owner_and_friend_left", "Blur everything.", (("blur_all", {}),)),
    Case("owner_and_friend_left", "Blur all faces.", (_rule("face", "blur"),)),
    Case("signs_and_plate", "Show the street signs again.", (_rule("street sign", "show"),)),
    Case("signs_and_plate", "Hide license plates.", (_rule("license plate", "blur"),)),
    Case("panic", "Clear panic.", (("clear_panic", {}),)),
    Case("panic", "Stop the panic, back to normal.", (("clear_panic", {}),)),
    Case("signs_and_plate", "Show the house numbers.", (_rule("house number", "show"),)),
    Case("empty", "Show storefront signs.", (_rule("storefront sign", "show"),)),
    Case("owner_and_friend_left", "Panic!", (("blur_all", {}),)),
    Case("two_candidates", "Unblur the person on my right.", (GET_SCENE, _track("face#9", "show"))),
    Case("two_candidates", "Show the one in the back.", (GET_SCENE, _track("face#9", "show"))),
    Case("signs_and_plate", "Unblur the plate on the right.", (GET_SCENE, _track("obj#3", "show"))),
    Case("signs_and_plate", "Show the sign on the left.", (GET_SCENE, _track("obj#1", "show"))),
    Case("empty", "Hide the tattoos.", (), lists_types=True),
    Case("owner_and_friend_left", "Blur my face too.", (GET_SCENE, _track("face#2", "blur"))),
    Case("empty", "Pixelate the house numbers.", (_rule("house number", "blur"),)),
    Case("signs_and_plate", "Unblur the license plates.", (_rule("license plate", "show"),)),
]


@dataclass
class _Run:
    """Everything one command produced, with times relative to the injection."""

    sent_at: float = 0.0
    calls: list[tuple[str, dict[str, Any], dict[str, Any], float]] = field(default_factory=list)
    replies: list[tuple[str, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    last_event: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ms(self) -> float:
        return (time.monotonic() - self.sent_at) * 1000


class _RecordingHandlers(ToolHandlers):
    def __init__(self, policy: Policy, scene: Scene, run: _Run) -> None:
        super().__init__(policy, lambda: scene)
        self._run = run

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = super().handle(name, arguments)
        with self._run.lock:
            self._run.calls.append((name, arguments, result, self._run.ms()))
            self._run.last_event = time.monotonic()
        return result


def run_case(case: Case, overrides: dict[str, Any], timeout_s: float = 15.0) -> _Run:
    run = _Run()
    policy = Policy()

    def on_transcript(role: str, text: str) -> None:
        if role != "agent":
            return
        with run.lock:
            if text.startswith(("Voice", "Busy", "Mic")):
                run.errors.append(text)
            else:
                run.replies.append((text, run.ms()))
            run.last_event = time.monotonic()

    agent = VoiceAgent(
        _RecordingHandlers(policy, scenes.ALL[case.scene], run),
        on_transcript,
        use_audio=False,
        keyword_fallback_s=None,
        **overrides,
    )
    agent.start()
    try:
        if not agent.wait_ready(10.0):
            run.errors.append("no SettingsApplied within 10 s")
            return run
        run.sent_at = time.monotonic()
        agent.inject_text(case.text)
        deadline = run.sent_at + timeout_s
        while time.monotonic() < deadline:
            time.sleep(0.05)
            with run.lock:
                settled = run.replies and time.monotonic() - run.last_event > 1.0
                if settled or run.errors:
                    break
    finally:
        agent.stop()
        agent.join(timeout=3.0)
    return run


def judge(case: Case, run: _Run) -> tuple[bool, str]:
    """Expected calls in order (extra get_scene calls allowed), and nothing else that changes state."""
    actual = [(name, args) for name, args, _, _ in run.calls]
    remaining = list(case.expected)
    for call in actual:
        if remaining and call == remaining[0]:
            remaining.pop(0)
    if remaining:
        return False, f"missing {remaining[0]}"
    extra = [c for c in actual if c[0] in STATE_TOOLS and c not in case.expected]
    if extra:
        return False, f"unexpected {extra[0]}"
    reply = " ".join(text for text, _ in run.replies).lower()
    if not reply:
        return False, "no reply text"
    if case.asks and "?" not in reply:
        return False, "expected a question"
    if case.lists_types and sum(word in reply for word in CAN_HIDE_WORDS) < 3:
        return False, "reply doesn't list the hideable types"
    return True, "ok"


def check_model(overrides: dict[str, Any]) -> tuple[int, list[float], list[float]]:
    model = overrides.get("think_model") or config.DG_THINK_MODEL
    print(f"\n=== {overrides.get('think_provider') or config.DG_THINK_PROVIDER} / {model} ===")
    passed, to_tool, to_reply = 0, [], []
    for number, case in enumerate(CASES, 1):
        run = run_case(case, overrides)
        ok, why = judge(case, run)
        passed += ok
        first_tool = run.calls[0][3] if run.calls else None
        last_reply = run.replies[-1][1] if run.replies else None
        if first_tool is not None:
            to_tool.append(first_tool)
        if last_reply is not None:
            to_reply.append(last_reply)
        print(f"{number:2}. {'PASS' if ok else 'FAIL'} [{case.scene}] {case.text!r}  ({why})")
        print(f"    expected: {_calls_text(case.expected) or '(no tool call)'}")
        print(f"    actual:   {_calls_text([(n, a) for n, a, _, _ in run.calls]) or '(no tool call)'}")
        print(f"    reply:    {' | '.join(text for text, _ in run.replies) or '-'}")
        print(f"    latency:  first tool {_ms(first_tool)}, reply {_ms(last_reply)}")
        for error in run.errors:
            print(f"    error:    {error}")
    print(f"{passed}/{len(CASES)}  {model}  median first tool {_median(to_tool)}, median reply {_median(to_reply)}")
    return passed, to_tool, to_reply


def _calls_text(calls: Any) -> str:
    return ", ".join(f"{name}({', '.join(f'{k}={v}' for k, v in sorted(args.items()))})" for name, args in calls)


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def _median(values: list[float]) -> str:
    return _ms(statistics.median(values)) if values else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", help="hosted think model; repeat to compare (default: config)")
    parser.add_argument("--provider", help=f"think provider (default: {config.DG_THINK_PROVIDER})")
    parser.add_argument("--no-temperature", action="store_true", help="leave temperature out of Settings")
    args = parser.parse_args()

    results = []
    for model in args.model or [None]:
        overrides: dict[str, Any] = {"think_model": model, "think_provider": args.provider}
        if args.no_temperature:
            overrides["temperature"] = None
        passed, _, _ = check_model(overrides)
        results.append((model or config.DG_THINK_MODEL, passed))
    if len(results) > 1:
        print("\nsummary: " + ", ".join(f"{model} {passed}/{len(CASES)}" for model, passed in results))
    return 0 if all(passed == len(CASES) for _, passed in results) else 1


if __name__ == "__main__":
    sys.exit(main())
