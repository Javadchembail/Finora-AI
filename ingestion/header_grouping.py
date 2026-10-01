"""
Universal financial statement header grouping.

Uses PDF coordinates to preserve logical columns.

Important:
- Multi-word headers such as "Value Date" are merged.
- Adjacent independent columns are NOT merged.
- Header grouping is based on semantic phrases + position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HeaderWord:
    text: str
    x0: float
    x1: float
    top: float


@dataclass
class HeaderGroup:
    text: str
    x0: float
    x1: float
    top: float
    words: List[HeaderWord]


class HeaderGrouper:

    # Only merge phrases that are genuinely known
    # multi-word financial concepts.
    MULTI_WORD_PATTERNS = {
        ("value", "date"),
        ("post", "date"),
        ("trxn.", "date"),
        ("txn", "date"),
        ("transaction", "date"),
        ("tran", "id"),
        ("transaction", "id"),
        ("account", "number"),
        ("opening", "balance"),
        ("closing", "balance"),
        ("current", "balance"),
        ("available", "balance"),
        ("effective", "available", "balance"),
        ("minimum", "payment"),
        ("payment", "due"),
        ("due", "date"),
    }

    def group(
        self,
        words: List[Dict],
        vertical_tolerance: float = 4.0,
    ) -> List[HeaderGroup]:

        normalized = [
            HeaderWord(
                text=str(word["text"]).strip(),
                x0=float(word["x0"]),
                x1=float(word["x1"]),
                top=float(word["top"]),
            )
            for word in words
            if str(word.get("text", "")).strip()
        ]

        normalized.sort(
            key=lambda word: (word.top, word.x0)
        )

        lines = self._group_lines(
            normalized,
            vertical_tolerance,
        )

        groups: List[HeaderGroup] = []

        for line in lines:
            groups.extend(
                self._group_line(line)
            )

        return groups

    # ---------------------------------------------------------
    # Group words into physical lines
    # ---------------------------------------------------------

    def _group_lines(
        self,
        words: List[HeaderWord],
        tolerance: float,
    ) -> List[List[HeaderWord]]:

        lines: List[List[HeaderWord]] = []

        for word in words:

            matched_line = None

            for line in lines:

                if abs(
                    line[0].top - word.top
                ) <= tolerance:

                    matched_line = line
                    break

            if matched_line is None:
                lines.append([word])
            else:
                matched_line.append(word)

        for line in lines:
            line.sort(key=lambda word: word.x0)

        return lines

    # ---------------------------------------------------------
    # Build logical groups
    # ---------------------------------------------------------

    def _group_line(
        self,
        line: List[HeaderWord],
    ) -> List[HeaderGroup]:

        if not line:
            return []

        groups: List[HeaderGroup] = []

        i = 0

        while i < len(line):

            matched = False

            # Try longest patterns first.
            patterns = sorted(
                self.MULTI_WORD_PATTERNS,
                key=len,
                reverse=True,
            )

            for pattern in patterns:

                length = len(pattern)

                if i + length > len(line):
                    continue

                candidate = line[i:i + length]

                candidate_words = tuple(
                    word.text.lower()
                    for word in candidate
                )

                if candidate_words != pattern:
                    continue

                # Ensure the words are reasonably close.
                if not self._words_are_close(candidate):
                    continue

                groups.append(
                    HeaderGroup(
                        text=" ".join(
                            word.text
                            for word in candidate
                        ),
                        x0=min(
                            word.x0
                            for word in candidate
                        ),
                        x1=max(
                            word.x1
                            for word in candidate
                        ),
                        top=min(
                            word.top
                            for word in candidate
                        ),
                        words=candidate,
                    )
                )

                i += length
                matched = True
                break

            if matched:
                continue

            word = line[i]

            groups.append(
                HeaderGroup(
                    text=word.text,
                    x0=word.x0,
                    x1=word.x1,
                    top=word.top,
                    words=[word],
                )
            )

            i += 1

        return groups

    # ---------------------------------------------------------
    # Horizontal proximity
    # ---------------------------------------------------------

    def _words_are_close(
        self,
        words: List[HeaderWord],
    ) -> bool:

        if len(words) <= 1:
            return True

        for previous, current in zip(
            words,
            words[1:],
        ):

            gap = current.x0 - previous.x1

            # Multi-word headers normally have
            # relatively small spacing.
            if gap > 25:
                return False

        return True