"""WP7/WP14/WP21 live check: typed commands through Deepgram's Voice Agent against fixture scenes.

    set -a; source .env; set +a; python scripts/voice_check.py [--model gpt-4.1-mini ...]

Text only (use_audio=False): no mic or speaker is opened. Each command gets a fresh
Policy and a fresh agent session, so no command sees another's history. The keyword
fallback is off, so the score measures the model alone. Exits nonzero below a full score.
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
from contracts import AD_CLASSES, DEFAULT_CLASS_RULES, KNOWN_CLASSES, Policy, Scene, SceneTrack  # noqa: E402
from fakes import scenes  # noqa: E402
from voice.dg_agent import VoiceAgent  # noqa: E402
from voice.tools import ToolHandlers  # noqa: E402

GET_SCENE = ("get_scene", {})
STATE_TOOLS = {"set_class_rule", "set_track", "set_replacement", "blur_all", "clear_panic", "show_all"}
CAN_HIDE_WORDS = ("face", "plate", "street", "house", "storefront")
# Every case runs with these on the "device", so the old commands are checked with the ad prompt too.
FAKE_AD_ASSETS = ("summer_sale.png", "coffee.jpg", "concert_tickets.png")


@dataclass(frozen=True)
class Case:
    scene: str  # key in SCENES
    text: str
    expected: tuple[tuple[str, dict[str, Any]], ...]  # in order; get_scene listed only where required
    asks: bool = False  # must reply with a question, no state change
    lists_types: bool = False  # must reply naming what it can hide, no state change
    any_order: tuple[tuple[str, dict[str, Any]], ...] = ()  # all required, in any order, after `expected`
    reply_lacks: tuple[str, ...] = ()  # the reply must contain none of these
    follow_up: str = ""  # a second user turn in the same session, sent once the first reply settles
    then_scene: str = ""  # the scene after the first get_scene call (a track got a new id)


def _rule(label: str, action: str) -> tuple[str, dict[str, Any]]:
    return ("set_class_rule", {"label": label, "action": action})


def _track(track_id: str, action: str) -> tuple[str, dict[str, Any]]:
    return ("set_track", {"track_id": track_id, "action": action})


def _replacement(label: str, asset: str) -> tuple[str, dict[str, Any]]:
    return ("set_replacement", {"label": label, "asset": asset})


def _sighting(track_id: str, label: str, side: str, distance: str, state: str) -> SceneTrack:
    return SceneTrack(id=track_id, label=label, side=side, distance=distance, state=state)  # type: ignore[arg-type]


def _scene(owner: str | None, *tracks: SceneTrack, shown: tuple[str, ...] = (), replaced: tuple[str, ...] = ()) -> Scene:
    """Ads keep their default (shown) unless replaced; every other class is blurred unless shown."""
    rules = {label: "show" if label in shown else DEFAULT_CLASS_RULES[label] for label in KNOWN_CLASSES}
    rules.update({label: "replace" for label in replaced})
    return Scene(owner=owner, tracks=tracks, class_rules=rules, panic=False)  # type: ignore[arg-type]


_OWNER = _sighting("face#2", "face", "center", "near", "show")

# Extra WP14 scenes, built here because fakes/ is read-only for this package.
SCENES: dict[str, Scene] = {
    **scenes.ALL,
    # two people on the owner's left, one on the right
    "two_left": _scene(
        "face#2", _OWNER,
        _sighting("face#5", "face", "owner_left", "near", "blur"),
        _sighting("face#7", "face", "owner_left", "far", "blur"),
        _sighting("face#9", "face", "owner_right", "near", "blur"),
    ),
    # both friends were unblurred earlier
    "friends_shown": _scene(
        "face#2", _OWNER,
        _sighting("face#7", "face", "owner_left", "near", "show"),
        _sighting("face#9", "face", "owner_right", "far", "show"),
    ),
    # one friend shown, one still blurred: "her again" can only be face#7
    "one_friend_shown": _scene(
        "face#2", _OWNER,
        _sighting("face#7", "face", "owner_left", "near", "show"),
        _sighting("face#9", "face", "owner_right", "far", "blur"),
    ),
    # plates and signs shown as a class; no owner, so sides are the operator's view
    "objects_shown": _scene(
        None,
        _sighting("obj#1", "street sign", "left", "far", "show"),
        _sighting("obj#3", "license plate", "right", "near", "show"),
        _sighting("obj#6", "license plate", "left", "near", "show"),
        shown=("license plate", "street sign"),
    ),
    # the owner in frame, a street sign far behind
    "owner_and_sign": _scene(
        "face#2", _OWNER,
        _sighting("obj#1", "street sign", "center", "far", "show"),
        _sighting("obj#3", "license plate", "owner_right", "near", "blur"),
        shown=("street sign",),
    ),
    # two plates, both blurred: "the plate" alone is ambiguous
    "two_plates": _scene(
        None,
        _sighting("obj#3", "license plate", "right", "near", "blur"),
        _sighting("obj#6", "license plate", "left", "far", "blur"),
    ),
    # the face class is shown: everyone is unblurred
    "faces_shown_class": _scene(
        "face#2", _OWNER,
        _sighting("face#7", "face", "owner_left", "near", "show"),
        _sighting("face#9", "face", "owner_right", "far", "show"),
        shown=("face",),
    ),
    # the face class is shown, one other person in view
    "one_face_shown_class": _scene(
        "face#2", _OWNER,
        _sighting("face#7", "face", "owner_left", "near", "show"),
        shown=("face",),
    ),
    # owner_and_friend_left after the tracker lost the friend and found her again as face#8
    "friend_renumbered": _scene(
        "face#2", _OWNER,
        _sighting("face#8", "face", "owner_left", "near", "blur"),
        _sighting("obj#3", "license plate", "right", "far", "blur"),
    ),
    # WP21: a billboard and a poster in view, shown as ads are by default
    "ads_in_view": _scene(
        "face#2", _OWNER,
        _sighting("obj#4", "billboard", "owner_left", "far", "show"),
        _sighting("obj#5", "poster", "owner_right", "near", "show"),
    ),
    # the same, with every ad class replaced by an image
    "ads_replaced": _scene(
        "face#2", _OWNER,
        _sighting("obj#4", "billboard", "owner_left", "far", "replace"),
        _sighting("obj#5", "poster", "owner_right", "near", "replace"),
        replaced=AD_CLASSES,
    ),
}

_RESET = (("clear_panic", {}), *(_rule(label, "blur") for label in KNOWN_CLASSES))
_SHOW_FACES = (_rule("face", "show"),)
# Showing the face class keeps new people unblurred: no follow-up question, no "they stay blurred".
_STILL_BLURRED = ("still be blurred", "stay blurred", "will be blurred", "remain blurred", "?")
_REPLACE_ADS = tuple(_replacement(label, FAKE_AD_ASSETS[0]) for label in AD_CLASSES)
_SHOW_ADS = tuple(_rule(label, "show") for label in AD_CLASSES)

CASES = [
    Case("owner_and_friend_left", "Unblur the person on my left, she's with me.", (GET_SCENE, _track("face#7", "show"))),
    Case("two_candidates", "Unblur the guy.", (GET_SCENE,), asks=True),
    Case("empty", "Hide the logos.", (), lists_types=True),
    Case("owner_and_friend_left", "Blur everything.", (("blur_all", {}),)),
    Case("owner_and_friend_left", "Blur all faces.", (_rule("face", "blur"),)),
    Case("signs_and_plate", "Show the street signs again.", (_rule("street sign", "show"),)),
    Case("signs_and_plate", "Hide license plates.", (_rule("license plate", "blur"),)),
    Case("panic", "Clear panic.", (("clear_panic", {}),)),
    # WP14: "back to normal" is now a full reset, not just clear_panic
    Case("panic", "Stop the panic, back to normal.", (), any_order=_RESET),
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
    # --- WP14: everything, selective, both directions ---
    Case("owner_and_friend_left", "Unblur everything.", (("show_all", {}),)),
    Case("signs_and_plate", "Show everything.", (("show_all", {}),)),
    Case("two_candidates", "Unblur all faces.", _SHOW_FACES, reply_lacks=_STILL_BLURRED),
    Case("owner_and_friend_left", "Show all the faces.", _SHOW_FACES, reply_lacks=_STILL_BLURRED),
    Case(
        "two_left", "Unblur the two people on my left.", (GET_SCENE,),
        any_order=(_track("face#5", "show"), _track("face#7", "show")),
    ),
    Case("objects_shown", "Blur the plate on the right.", (GET_SCENE, _track("obj#3", "blur"))),
    Case("owner_and_sign", "Hide the sign behind me.", (GET_SCENE, _track("obj#1", "blur"))),
    Case("one_friend_shown", "Blur her again.", (GET_SCENE, _track("face#7", "blur"))),
    Case("friends_shown", "Blur the person on my right.", (GET_SCENE, _track("face#9", "blur"))),
    Case("two_plates", "Unblur the plate.", (GET_SCENE,), asks=True),
    Case("signs_and_plate", "Hide all license plates.", (_rule("license plate", "blur"),)),
    Case("empty", "Show street signs.", (_rule("street sign", "show"),)),
    Case("owner_and_friend_left", "Reset.", (), any_order=_RESET),
    # --- WP14 revision: the face class may be shown ---
    Case("two_candidates", "Unblur everyone.", _SHOW_FACES, reply_lacks=_STILL_BLURRED),
    Case("empty", "Keep everyone unblurred.", _SHOW_FACES, reply_lacks=_STILL_BLURRED),
    Case("owner_and_friend_left", "People should start unblurred.", _SHOW_FACES, reply_lacks=_STILL_BLURRED),
    Case("faces_shown_class", "Blur everyone.", (_rule("face", "blur"),)),
    Case("faces_shown_class", "Blur people again.", (_rule("face", "blur"),)),
    Case("faces_shown_class", "Blur the person on my left.", (GET_SCENE, _track("face#7", "blur"))),
    Case("one_face_shown_class", "Blur her.", (GET_SCENE, _track("face#7", "blur"))),
    Case(
        "two_candidates", "Unblur the guy.", (GET_SCENE, _track("face#7", "show")),
        asks=True, follow_up="The closer one.",
    ),
    Case(
        "owner_and_friend_left", "Unblur the person on my left.",
        (GET_SCENE, _track("face#7", "show"), GET_SCENE, _track("face#8", "show")),
        then_scene="friend_renumbered",
    ),
    # --- WP21: replacing ads with an image on the device ---
    Case("ads_in_view", "Replace the ads.", (), any_order=_REPLACE_ADS),
    Case("ads_in_view", "Put the coffee ad on the billboards.", (_replacement("billboard", "coffee.jpg"),)),
    Case("ads_in_view", "Swap the posters for the concert tickets ad.", (_replacement("poster", "concert_tickets.png"),)),
    Case("ads_replaced", "Stop replacing the ads.", (), any_order=_SHOW_ADS),
    Case("ads_replaced", "Show the real ads.", (), any_order=_SHOW_ADS),
    Case("ads_in_view", "Blur the posters.", (_rule("poster", "blur"),)),
]


@dataclass
class _Run:
    """Everything one command produced, with times relative to the injection."""

    sent_at: float = 0.0
    calls: list[tuple[str, dict[str, Any], dict[str, Any], float]] = field(default_factory=list)
    replies: list[tuple[str, float]] = field(default_factory=list)
    first_turn_replies: int = 0  # how many replies came before the follow-up was sent
    errors: list[str] = field(default_factory=list)
    last_event: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def ms(self) -> float:
        return (time.monotonic() - self.sent_at) * 1000


class _RecordingHandlers(ToolHandlers):
    """Records every call; switches to `then_scene` once the agent has read the first scene."""

    def __init__(self, policy: Policy, scene: Scene, then_scene: Scene | None, run: _Run) -> None:
        super().__init__(policy, lambda: self._scene, ad_assets=FAKE_AD_ASSETS)
        self._scene = scene
        self._then_scene = then_scene
        self._run = run

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = super().handle(name, arguments)
        if name == "get_scene" and self._then_scene is not None:
            self._scene, self._then_scene = self._then_scene, None
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

    then_scene = SCENES[case.then_scene] if case.then_scene else None
    agent = VoiceAgent(
        _RecordingHandlers(policy, SCENES[case.scene], then_scene, run),
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
        _wait_until_settled(run, run.sent_at + timeout_s)
        if case.follow_up and not run.errors:
            with run.lock:
                run.first_turn_replies = len(run.replies)
            agent.inject_text(case.follow_up)
            _wait_until_settled(run, time.monotonic() + timeout_s, after=run.first_turn_replies)
    finally:
        agent.stop()
        agent.join(timeout=3.0)
    return run


def _wait_until_settled(run: _Run, deadline: float, after: int = 0) -> None:
    """Until a reply (beyond the first `after`) has come and nothing else happened for 1 s."""
    while time.monotonic() < deadline:
        time.sleep(0.05)
        with run.lock:
            settled = len(run.replies) > after and time.monotonic() - run.last_event > 1.0
            if settled or run.errors:
                return


def judge(case: Case, run: _Run) -> tuple[bool, str]:
    """Expected calls in order (extra get_scene calls allowed), then every any-order call,
    and nothing else that changes state."""
    actual = [(name, args) for name, args, _, _ in run.calls]
    remaining = list(case.expected)
    after_expected = 0
    for position, call in enumerate(actual):
        if remaining and call == remaining[0]:
            remaining.pop(0)
            after_expected = position + 1
    if remaining:
        return False, f"missing {remaining[0]}"
    unordered_missing = [c for c in case.any_order if c not in actual[after_expected:]]
    if unordered_missing:
        return False, f"missing {unordered_missing[0]}"
    allowed = (*case.expected, *case.any_order)
    extra = [c for c in actual if c[0] in STATE_TOOLS and c not in allowed]
    if extra:
        return False, f"unexpected {extra[0]}"
    reply = " ".join(text for text, _ in run.replies).lower()
    if not reply:
        return False, "no reply text"
    if case.follow_up and len(run.replies) <= run.first_turn_replies:
        return False, "no reply to the follow-up"
    first_turn = run.replies[: run.first_turn_replies] if case.follow_up else run.replies
    if case.asks and "?" not in " ".join(text for text, _ in first_turn):
        return False, "expected a question"
    if case.lists_types and sum(word in reply for word in CAN_HIDE_WORDS) < 3:
        return False, "reply doesn't list the hideable types"
    said = [word for word in case.reply_lacks if word in reply]
    if said:
        return False, f"reply shouldn't say {said[0]!r}"
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
        # a two-turn case's last reply includes the second user turn: leave it out of the median
        last_reply = run.replies[-1][1] if run.replies and not case.follow_up else None
        if first_tool is not None:
            to_tool.append(first_tool)
        if last_reply is not None:
            to_reply.append(last_reply)
        follow_up = f" then {case.follow_up!r}" if case.follow_up else ""
        print(f"{number:2}. {'PASS' if ok else 'FAIL'} [{case.scene}] {case.text!r}{follow_up}  ({why})")
        any_order = f" + any order: {_calls_text(case.any_order)}" if case.any_order else ""
        print(f"    expected: {_calls_text(case.expected) or '(no tool call)'}{any_order}")
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
    parser.add_argument("--reasoning", help="reasoning_mode for OpenAI reasoning models: low, medium, high")
    args = parser.parse_args()

    results = []
    for model in args.model or [None]:
        overrides: dict[str, Any] = {"think_model": model, "think_provider": args.provider}
        if args.no_temperature:
            overrides["temperature"] = None
        if args.reasoning:
            overrides["reasoning_mode"] = args.reasoning
        passed, _, _ = check_model(overrides)
        results.append((model or config.DG_THINK_MODEL, passed))
    if len(results) > 1:
        print("\nsummary: " + ", ".join(f"{model} {passed}/{len(CASES)}" for model, passed in results))
    return 0 if all(passed == len(CASES) for _, passed in results) else 1


if __name__ == "__main__":
    sys.exit(main())
