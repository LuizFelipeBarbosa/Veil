"""On-device phrase spotting with Vosk: the wake phrase, and panic words for an offline wake turn.

Audio fed here stays in memory: nothing is stored, sent or logged. The recognizer is the only
thing that holds audio, and it is reset after every hit and periodically.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Sequence

import config

# Decoys give the recognizer somewhere else to put near-misses ("they fail", "okay").
# Measured on macOS `say` voices: 24/24 hits; false fires 11/36 with no decoys, 2/36 with these.
WAKE_GRAMMAR = (
    "hey veil", "hey", "veil", "okay", "they", "fail", "everyone",
    "there", "the", "say", "sail", "hail", "mail", "they will", "[unk]",
)
PANIC_PHRASES = ("panic", "blur everything", "blur all")
_RESET_EVERY_BLOCKS = 600  # about 30 s of 50 ms blocks: keeps the recognizer from growing


class WakeUnavailable(RuntimeError):
    """vosk or its model is missing: no listening (fail closed), Talk still works."""


@functools.lru_cache(maxsize=2)
def _load_model(path: str) -> Any:
    try:
        import vosk
    except ImportError as error:
        raise WakeUnavailable("Wake phrase off: vosk is not installed") from error
    if not Path(path).is_dir():
        raise WakeUnavailable(f"Wake phrase off: no Vosk model at {path}")
    vosk.SetLogLevel(-1)  # Vosk logs nothing about what it hears
    return vosk.Model(path)


class WakeDetector:
    """Feed 16 kHz mono int16 blocks; feed() is True exactly once per detected phrase.

    A phrase counts when it is the end of the recognizer's partial guess for `stable_blocks`
    blocks in a row, or the end of a final result.
    """

    def __init__(
        self,
        phrases: Sequence[str] = (config.WAKE_PHRASE,),
        grammar: Sequence[str] = WAKE_GRAMMAR,
        *,
        stable_blocks: int = config.WAKE_STABLE_BLOCKS,
        model_path: Path = config.VOSK_MODEL,
    ) -> None:
        model = _load_model(str(model_path))  # raises WakeUnavailable first if vosk is missing
        import vosk

        self._phrases = tuple(phrases)
        self._stable_blocks = stable_blocks
        self._recognizer = vosk.KaldiRecognizer(model, config.MIC_SAMPLE_RATE, json.dumps(list(grammar)))
        self._stable = 0
        self._blocks = 0

    def feed(self, block: bytes) -> bool:
        self._blocks += 1
        if self._recognizer.AcceptWaveform(block):
            heard = json.loads(self._recognizer.Result()).get("text", "")
            if self._ends_with_phrase(heard):
                self.reset()
                return True
            self._stable = 0
        else:
            partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
            self._stable = self._stable + 1 if self._ends_with_phrase(partial) else 0
            if self._stable >= self._stable_blocks:
                self.reset()
                return True
        if self._blocks >= _RESET_EVERY_BLOCKS and self._stable == 0:
            self.reset()
        return False

    def reset(self) -> None:
        self._recognizer.Reset()
        self._stable = 0
        self._blocks = 0

    def _ends_with_phrase(self, text: str) -> bool:
        words = text.split()
        return any(words[-len(p.split()):] == p.split() for p in self._phrases)


def panic_spotter() -> WakeDetector:
    """For an offline wake turn: only panic words, never anything that reveals."""
    return WakeDetector(PANIC_PHRASES, (*PANIC_PHRASES, "[unk]"), stable_blocks=2)
