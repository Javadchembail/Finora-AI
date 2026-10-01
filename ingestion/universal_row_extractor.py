from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class UniversalRowExtractor:
    """
    Universal, bank-agnostic transaction row extractor.

    Responsibilities:
    - Group PDF words into rows.
    - Reconstruct split dates.
    - Keep posting/transaction/value dates separate.
    - Extract numeric fields using semantic column positions.
    - Extract descriptions using the space between semantic columns.
    - Reject obvious non-transaction rows.
    """

    MONTHS = {
        "JAN", "FEB", "MAR", "APR",
        "MAY", "JUN", "JUL", "AUG",
        "SEP", "OCT", "NOV", "DEC",
    }

    METADATA_TERMS = {
        "opening balance",
        "closing balance",
        "available balance",
        "statement period",
        "credit limit",
        "cash limit",
        "payment due",
        "minimum payment",
        "total amount",
        "total payments",
        "total purchases",
        "total fees",
        "total interest",
        "transaction details",
        "transaction history",
        "account summary",
        "card summary",
        "statement summary",
        "previous balance",
        "new balance",
    }

    # ================================================================
    # PUBLIC API
    # ================================================================

    def extract(
        self,
        words: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not words or not columns:
            return []

        columns = self._normalize_columns(columns)

        if not columns:
            return []

        header_top = max(
            float(column.get("top", 0))
            for column in columns
        )

        data_words = [
            word
            for word in words
            if float(word.get("top", 0))
            > header_top + 6
        ]

        grouped_rows = self._group_words_by_row(
            data_words
        )

        extracted_rows = []

        for row_words in grouped_rows:

            parsed = self._parse_row(
                row_words,
                columns,
            )

            if not parsed:
                continue

            if not self._is_transaction(parsed):
                continue

            extracted_rows.append(parsed)

        return extracted_rows

    # ================================================================
    # COLUMN NORMALIZATION
    # ================================================================

    def _normalize_columns(
        self,
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        result = []

        for column in columns:

            try:

                result.append(
                    {
                        "header": str(
                            column.get(
                                "header",
                                "",
                            )
                        ),
                        "semantic_type": str(
                            column.get(
                                "semantic_type",
                                "",
                            )
                        ),
                        "x0": float(
                            column.get(
                                "x0",
                                0,
                            )
                        ),
                        "x1": float(
                            column.get(
                                "x1",
                                0,
                            )
                        ),
                        "top": float(
                            column.get(
                                "top",
                                0,
                            )
                        ),
                        "confidence": float(
                            column.get(
                                "confidence",
                                0,
                            )
                        ),
                    }
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return sorted(
            result,
            key=lambda x: x["x0"],
        )

    # ================================================================
    # ROW GROUPING
    # ================================================================

    def _group_words_by_row(
        self,
        words: List[Dict[str, Any]],
        tolerance: float = 3.0,
    ) -> List[List[Dict[str, Any]]]:

        if not words:
            return []

        sorted_words = sorted(
            words,
            key=lambda w: (
                float(
                    w.get(
                        "top",
                        0,
                    )
                ),
                float(
                    w.get(
                        "x0",
                        0,
                    )
                ),
            ),
        )

        rows = []

        for word in sorted_words:

            top = float(
                word.get(
                    "top",
                    0,
                )
            )

            if not rows:
                rows.append([word])
                continue

            previous_top = float(
                rows[-1][0].get(
                    "top",
                    0,
                )
            )

            if abs(
                top - previous_top
            ) <= tolerance:

                rows[-1].append(word)

            else:

                rows.append([word])

        for row in rows:

            row.sort(
                key=lambda w: float(
                    w.get(
                        "x0",
                        0,
                    )
                )
            )

        return rows

    # ================================================================
    # ROW PARSING
    # ================================================================

    def _parse_row(
        self,
        row_words: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        if not row_words:
            return None

        semantic_columns = {}

        for column in columns:

            semantic = column.get(
                "semantic_type"
            )

            if semantic:
                semantic_columns[
                    semantic
                ] = column

        result: Dict[str, Any] = {}

        used_indexes = set()

        # ------------------------------------------------------------
        # DATES
        # ------------------------------------------------------------

        date_semantics = [
            semantic
            for semantic in (
                "posting_date",
                "transaction_date",
                "value_date",
                "date",
            )
            if semantic in semantic_columns
        ]

        for semantic in date_semantics:

            column = semantic_columns[
                semantic
            ]

            date_value, indexes = (
                self._extract_date_from_column(
                    row_words,
                    column,
                    columns,
                )
            )

            if date_value:

                result[
                    semantic
                ] = date_value

                used_indexes.update(
                    indexes
                )

        # ------------------------------------------------------------
        # NUMERIC FIELDS
        # ------------------------------------------------------------

        numeric_semantics = [
            semantic
            for semantic in (
                "debit",
                "credit",
                "amount",
                "withdrawals",
                "deposits",
                "balance",
            )
            if semantic in semantic_columns
        ]

        for semantic in numeric_semantics:

            column = semantic_columns[
                semantic
            ]

            value, index = (
                self._extract_numeric_from_column(
                    row_words,
                    column,
                    used_indexes,
                )
            )

            if value is not None:

                result[
                    semantic
                ] = value

                if index is not None:
                    used_indexes.add(
                        index
                    )

        # ------------------------------------------------------------
        # TRANSACTION ID
        # ------------------------------------------------------------

        if (
            "transaction_id"
            in semantic_columns
        ):

            value, indexes = (
                self._extract_text_from_region(
                    row_words,
                    "transaction_id",
                    semantic_columns,
                    columns,
                    used_indexes,
                )
            )

            if value:

                result[
                    "transaction_id"
                ] = value

                used_indexes.update(
                    indexes
                )

        # ------------------------------------------------------------
        # DESCRIPTION
        # ------------------------------------------------------------

        if (
            "description"
            in semantic_columns
        ):

            value, indexes = (
                self._extract_text_from_region(
                    row_words,
                    "description",
                    semantic_columns,
                    columns,
                    used_indexes,
                )
            )

            if value:

                result[
                    "description"
                ] = value

                used_indexes.update(
                    indexes
                )

        # ------------------------------------------------------------
        # FALLBACK DESCRIPTION
        # ------------------------------------------------------------

        if not result.get(
            "description"
        ):

            fallback_words = []

            for index, word in enumerate(
                row_words
            ):

                if index in used_indexes:
                    continue

                text = str(
                    word.get(
                        "text",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                if self._is_number(text):
                    continue

                if self._is_date_token(text):
                    continue

                fallback_words.append(text)

            if fallback_words:

                result[
                    "description"
                ] = " ".join(
                    fallback_words
                )

        return {
            key: value
            for key, value in result.items()
            if value not in (
                None,
                "",
                [],
            )
        }

    # ================================================================
    # DATE EXTRACTION
    # ================================================================

    def _extract_date_from_column(
        self,
        row_words: List[Dict[str, Any]],
        column: Dict[str, Any],
        columns: List[Dict[str, Any]],
    ) -> Tuple[
        Optional[str],
        List[int],
    ]:

        region = self._get_date_region(
            column,
            columns,
        )

        candidates = []

        for index, word in enumerate(
            row_words
        ):

            x0 = float(
                word.get(
                    "x0",
                    0,
                )
            )

            x1 = float(
                word.get(
                    "x1",
                    x0,
                )
            )

            center = (
                x0 + x1
            ) / 2

            if (
                region[0]
                <= center
                <= region[1]
            ):

                text = str(
                    word.get(
                        "text",
                        "",
                    )
                ).strip()

                if text:

                    candidates.append(
                        (
                            index,
                            word,
                        )
                    )

        if not candidates:
            return None, []

        candidates.sort(
            key=lambda item: float(
                item[1].get(
                    "x0",
                    0,
                )
            )
        )

        texts = [
            str(
                word.get(
                    "text",
                    "",
                )
            ).strip()
            for _, word in candidates
        ]

        # Try combinations of adjacent tokens.
        for start in range(
            len(texts)
        ):

            for end in range(
                start + 1,
                min(
                    len(texts),
                    start + 4,
                ) + 1,
            ):

                candidate = " ".join(
                    texts[start:end]
                )

                normalized = (
                    self._normalize_date_text(
                        candidate
                    )
                )

                if normalized:

                    indexes = [
                        candidates[i][0]
                        for i in range(
                            start,
                            end,
                        )
                    ]

                    return (
                        normalized,
                        indexes,
                    )

        # Individual token fallback.
        for index, text in zip(
            [
                item[0]
                for item in candidates
            ],
            texts,
        ):

            normalized = (
                self._normalize_date_text(
                    text
                )
            )

            if normalized:

                return (
                    normalized,
                    [index],
                )

        return None, []

    # ================================================================
    # NUMERIC EXTRACTION
    # ================================================================

    def _extract_numeric_from_column(
        self,
        row_words: List[Dict[str, Any]],
        column: Dict[str, Any],
        used_indexes: set,
    ) -> Tuple[
        Optional[str],
        Optional[int],
    ]:

        column_x0 = float(
            column.get(
                "x0",
                0,
            )
        )

        column_x1 = float(
            column.get(
                "x1",
                column_x0,
            )
        )

        # PDF table headers are often narrower than the actual
        # values underneath them. Allow the numeric value to extend
        # beyond the header while still keeping the region narrow
        # enough to avoid neighboring columns.
        tolerance = 20.0

        region_start = (
            column_x0 - tolerance
        )

        region_end = (
            column_x1 + tolerance
        )

        candidates = []

        for index, word in enumerate(
            row_words
        ):

            if index in used_indexes:
                continue

            x0 = float(
                word.get(
                    "x0",
                    0,
                )
            )

            x1 = float(
                word.get(
                    "x1",
                    x0,
                )
            )

            center = (
                x0 + x1
            ) / 2

            if not (
                region_start
                <= center
                <= region_end
            ):
                continue

            text = str(
                word.get(
                    "text",
                    "",
                )
            ).strip()

            if self._is_number(text):

                candidates.append(
                    (
                        index,
                        word,
                    )
                )

        if not candidates:
            return None, None

        candidates.sort(
            key=lambda item: float(
                item[1].get(
                    "x0",
                    0,
                )
            )
        )

        index, word = candidates[-1]

        return (
            str(
                word.get(
                    "text",
                    "",
                )
            ).strip(),
            index,
        )

    # ================================================================
    # TEXT EXTRACTION
    # ================================================================

    def _extract_text_from_region(
        self,
        row_words: List[Dict[str, Any]],
        semantic: str,
        semantic_columns: Dict[
            str,
            Dict[str, Any],
        ],
        columns: List[Dict[str, Any]],
        used_indexes: set,
    ) -> Tuple[
        Optional[str],
        List[int],
    ]:

        target = semantic_columns[
            semantic
        ]

        region = self._get_text_region(
            target,
            columns,
        )

        candidates = []

        for index, word in enumerate(
            row_words
        ):

            if index in used_indexes:
                continue

            x0 = float(
                word.get(
                    "x0",
                    0,
                )
            )

            x1 = float(
                word.get(
                    "x1",
                    x0,
                )
            )

            center = (
                x0 + x1
            ) / 2

            if not (
                region[0]
                <= center
                <= region[1]
            ):
                continue

            text = str(
                word.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            if self._is_number(text):
                continue

            if self._is_date_token(text):
                continue

            candidates.append(
                (
                    index,
                    word,
                )
            )

        if not candidates:
            return None, []

        candidates.sort(
            key=lambda item: float(
                item[1].get(
                    "x0",
                    0,
                )
            )
        )

        values = [
            str(
                word.get(
                    "text",
                    "",
                )
            ).strip()
            for _, word in candidates
        ]

        indexes = [
            index
            for index, _ in candidates
        ]

        return (
            " ".join(values),
            indexes,
        )

    # ================================================================
    # DATE REGION
    # ================================================================

    def _get_date_region(
        self,
        column: Dict[str, Any],
        columns: List[Dict[str, Any]],
    ) -> Tuple[
        float,
        float,
    ]:

        x0 = float(
            column.get(
                "x0",
                0,
            )
        )

        x1 = float(
            column.get(
                "x1",
                x0,
            )
        )

        # Dates are commonly split into multiple PDF words such as
        # "13" + "AUG", so give them a little room.
        return (
            x0 - 10.0,
            x1 + 10.0,
        )

    # ================================================================
    # TEXT REGION
    # ================================================================

    def _get_text_region(
        self,
        target: Dict[str, Any],
        columns: List[Dict[str, Any]],
    ) -> Tuple[
        float,
        float,
    ]:

        target_x0 = float(
            target.get(
                "x0",
                0,
            )
        )

        target_x1 = float(
            target.get(
                "x1",
                target_x0,
            )
        )

        ordered = sorted(
            columns,
            key=lambda c: float(
                c.get(
                    "x0",
                    0,
                )
            ),
        )

        previous_x1 = None
        next_x0 = None

        for column in ordered:

            if column is target:
                continue

            column_x0 = float(
                column.get(
                    "x0",
                    0,
                )
            )

            column_x1 = float(
                column.get(
                    "x1",
                    column_x0,
                )
            )

            if column_x1 <= target_x0:

                if (
                    previous_x1 is None
                    or column_x1 > previous_x1
                ):

                    previous_x1 = column_x1

            elif column_x0 >= target_x1:

                if (
                    next_x0 is None
                    or column_x0 < next_x0
                ):

                    next_x0 = column_x0

        if previous_x1 is None:
            start = target_x0 - 20
        else:
            start = previous_x1 + 8

        if next_x0 is None:
            end = target_x1 + 150
        else:
            end = next_x0 - 8

        return (
            start,
            end,
        )

    # ================================================================
    # TRANSACTION VALIDATION
    # ================================================================

    def _is_transaction(
        self,
        row: Dict[str, Any],
    ) -> bool:

        if not row:
            return False

        date_value = (
            row.get(
                "transaction_date"
            )
            or row.get(
                "posting_date"
            )
            or row.get(
                "value_date"
            )
            or row.get(
                "date"
            )
        )

        if not date_value:
            return False

        numeric_value = any(
            row.get(key)
            not in (
                None,
                "",
            )
            for key in (
                "amount",
                "debit",
                "credit",
                "withdrawals",
                "deposits",
            )
        )

        if not numeric_value:
            return False

        description = str(
            row.get(
                "description",
                "",
            )
            or ""
        ).strip()

        transaction_id = str(
            row.get(
                "transaction_id",
                "",
            )
            or ""
        ).strip()

        if (
            not description
            and not transaction_id
        ):
            return False

        if self._is_metadata_row(
            description
        ):
            return False

        return True

    def _is_metadata_row(
        self,
        description: str,
    ) -> bool:

        text = description.lower().strip()

        if not text:
            return False

        for term in self.METADATA_TERMS:

            if term in text:
                return True

        header_terms = {
            "post date",
            "trxn date",
            "transaction date",
            "description",
            "amount",
            "withdrawals",
            "deposits",
            "balance",
            "particulars",
            "value date",
            "transaction id",
        }

        if text in header_terms:
            return True

        return False

    # ================================================================
    # DATE HELPERS
    # ================================================================

    def _normalize_date_text(
        self,
        value: str,
    ) -> Optional[str]:

        value = str(
            value or ""
        ).strip().upper()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        if re.fullmatch(
            r"\d{1,2}\s+[A-Z]{3}",
            value,
        ):

            day, month = value.split()

            if month in self.MONTHS:

                return (
                    f"{day.zfill(2)} {month}"
                )

        if re.fullmatch(
            r"\d{1,2}\s+[A-Z]{3}\s+\d{4}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{1,2}-[A-Z]{3}-\d{4}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{1,2}/\d{1,2}/\d{4}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{1,2}-\d{1,2}-\d{4}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{1,2}\.\d{1,2}\.\d{4}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{4}-\d{1,2}-\d{1,2}",
            value,
        ):
            return value

        if re.fullmatch(
            r"\d{4}/\d{1,2}/\d{1,2}",
            value,
        ):
            return value

        return None

    # ================================================================
    # TOKEN HELPERS
    # ================================================================

    def _is_date_token(
        self,
        text: str,
    ) -> bool:

        value = str(
            text or ""
        ).strip().upper()

        if not value:
            return False

        if value in self.MONTHS:
            return True

        patterns = (
            r"^\d{1,2}$",
            r"^\d{1,2}-[A-Z]{3}$",
            r"^\d{1,2}/[A-Z]{3}$",
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            r"^\d{1,2}-\d{1,2}-\d{4}$",
            r"^\d{4}-\d{1,2}-\d{1,2}$",
            r"^\d{1,2}-[A-Z]{3}-\d{4}$",
        )

        return any(
            re.fullmatch(
                pattern,
                value,
            )
            for pattern in patterns
        )

    def _is_number(
        self,
        text: str,
    ) -> bool:

        if text is None:
            return False

        value = str(
            text
        ).strip().upper()

        if not value:
            return False

        value = value.replace(
            ",",
            "",
        )

        value = re.sub(
            r"^[₹$€£¥]+",
            "",
            value,
        )

        value = re.sub(
            r"[₹$€£¥]+$",
            "",
            value,
        )

        value = re.sub(
            r"(CR|DR)$",
            "",
            value,
        ).strip()

        if (
            value.startswith("(")
            and value.endswith(")")
        ):

            value = value[
                1:-1
            ].strip()

        return bool(
            re.fullmatch(
                r"[+-]?\d+(?:\.\d+)?",
                value,
            )
        )


# Backward-compatible alias.
RowRegionExtractor = UniversalRowExtractor