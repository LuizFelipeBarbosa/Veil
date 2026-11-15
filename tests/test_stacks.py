"""Stacked detections: one highlight per object, and a tap flips the whole stack."""

from __future__ import annotations

from contracts import Highlight
from pipeline import _one_per_stack


def test_tracks_stacked_on_one_ad_show_one_highlight_with_the_most_censored_state() -> None:
    panel = Highlight("obj#1", "advertisement", (100, 100, 300, 400), "replace")
    poster = Highlight("obj#2", "poster", (110, 110, 290, 390), "blur")
    twin = Highlight("obj#3", "advertisement", (105, 100, 300, 395), "replace")
    shown, stacks = _one_per_stack([poster, panel, twin])
    assert [(h.id, h.xyxy, h.state) for h in shown] == [("obj#1", (100, 100, 300, 400), "blur")]
    assert set(stacks["obj#1"]) == {"obj#2", "obj#3"}


def test_faces_other_families_and_objects_apart_keep_their_own_highlight() -> None:
    panel = Highlight("obj#1", "advertisement", (100, 100, 300, 400), "replace")
    face = Highlight("face#1", "face", (150, 150, 250, 250), "blur")
    other_face = Highlight("face#2", "face", (155, 150, 250, 250), "blur")
    sign = Highlight("obj#2", "street sign", (120, 120, 200, 160), "blur")
    far = Highlight("obj#3", "poster", (400, 100, 600, 400), "show")
    shown, stacks = _one_per_stack([panel, face, other_face, sign, far])
    assert {h.id for h in shown} == {"obj#1", "face#1", "face#2", "obj#2", "obj#3"}
    assert stacks == {}
