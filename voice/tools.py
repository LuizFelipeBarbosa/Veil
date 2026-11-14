"""Tool handlers and the keyword fallback (WP7). Schemas come from contracts.TOOL_SCHEMAS."""

from __future__ import annotations

from typing import Any, Callable

from contracts import Policy, Scene


class ToolHandlers:
    """Runs the 5 client-side tools against the local Policy and the live Scene."""

    def __init__(self, policy: Policy, get_scene: Callable[[], Scene]) -> None:
        raise NotImplementedError("WP7")

    def handle(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool and return its JSON-ready result.

        Never raises: a refusal (unknown tool or label, face/show, track not in
        the live scene) comes back as {"ok": False, "error": "..."}.
        """
        raise NotImplementedError("WP7")


def parse_keywords(transcript: str) -> tuple[str, dict[str, Any]] | None:
    """Map a transcript to (tool name, arguments) without the model, or None if nothing matches.

    Only class-wide and panic commands; anything about a specific person returns None.
    """
    raise NotImplementedError("WP7")
