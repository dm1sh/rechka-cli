"""User dictionary: ordered word replacements, ported from the Android app.

Format (one rule per line):
  ключ=значение   replace whole word 'ключ' with 'значение'
  ключ            (no '=') delete the word entirely
  # comment       ignored
Matching is case-insensitive and word-boundary aware (Cyrillic + Latin +
digits count as word characters). Rules apply top-to-bottom.
"""

from __future__ import annotations

import re
from pathlib import Path

_WORD_CHARS = "а-яёА-ЯЁa-zA-Z0-9"


class Dictionary:
    def __init__(self, rules: list[tuple[re.Pattern[str], str]]):
        self._rules = rules

    @classmethod
    def load(cls, path: Path) -> "Dictionary":
        rules: list[tuple[re.Pattern[str], str]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                trimmed = line.strip()
                if not trimmed or trimmed.startswith("#"):
                    continue
                if "=" in trimmed:
                    source, target = (part.strip() for part in trimmed.split("=", 1))
                else:
                    source, target = trimmed, ""
                if not source:
                    continue
                pattern = re.compile(
                    rf"(?<![{_WORD_CHARS}]){re.escape(source)}(?![{_WORD_CHARS}])",
                    re.IGNORECASE,
                )
                rules.append((pattern, target))
        return cls(rules)

    def apply(self, text: str) -> str:
        if not text or not self._rules:
            return text
        result = text
        for pattern, target in self._rules:
            result = pattern.sub(target, result)
        # Deletion rules can leave double spaces behind.
        return re.sub(r" {2,}", " ", result).strip()
