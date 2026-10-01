# ingestion/row_region_extractor.py

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


class UniversalRowExtractor:
    """
    Universal transaction-row extractor.

    This extractor is intentionally bank-agnostic.

    It does NOT contain rules such as:
        if bank == "Federal Bank"
        if bank == "Emirates Islamic"

    Instead, it uses the semantic structure detected by
    UniversalStructureDetector.

    Supported structures include:

        Date | Description | Amount | Balance

        Post Date | Trxn Date | Description | Amount

        Date | Value Date | Particulars | Tran ID |
        Withdrawals | Deposits | Balance

    The extractor works from PDF word coordinates rather than
    relying on hard-coded column positions.
    """

    # --------------------------------------------------------------
    # DATE PATTERNS
    # --------------------------------------------------------------

    MONTHS = {
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    }

    FULL_DATE_PATTERNS = [
        # 31-AUG-2026
        re.compile(
            r"^\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}$",
            re.IGNORECASE,
        ),

        # 31-08-2026
        re.compile(
            r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$"
        ),

        # 2026-08-31
        re.compile(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"
        ),

        # 31.08.2026
        re.compile(
            r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"
        ),

        # 25/09/26
        re.compile(
            r"^\d{1,2}/\d{1,2}/\d{2,4}$"
        ),
    ]

    # --------------------------------------------------------------
    # NUMERIC / AMOUNT PATTERN
    # --------------------------------------------------------------

    NUMBER_PATTERN = re.compile(
        r"""
        ^
        [\(\-+]?
        (?:
            \d{1,3}(?:,\d{3})+
            |
            \d+
        )
        (?:\.\d+)?
        (?:CR|DR)?
        \)?
        $
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # --------------------------------------------------------------
    # TRANSACTION ID PATTERNS
    # --------------------------------------------------------------

    TRANSACTION_ID_PATTERNS = [
        re.compile(
            r"^[A-Z]\d{5,}$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^[A-Z]{1,4}\d{4,}$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\d{6,}$"
        ),

        re.compile(
            r"^[A-Z0-9]{6,}/[A-Z0-9/_-]+$",
            re.IGNORECASE,
        ),
    ]

    # Generic transaction/reference markers.
    TRANSACTION_MARKERS = {
        "TFR",
        "TRF",
        "TXN",
        "TRAN",
        "TRANSFER",
        "REF",
        "REFERENCE",
    }

    # Words that should not become descriptions.
    IGNORED_TOKENS = {
        "DR",
        "CR",
    }

    # --------------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------------

    def __init__(
        self,
        row_tolerance: float = 4.0,
        minimum_description_length: int = 1,
    ):
        self.row_tolerance = row_tolerance
        self.minimum_description_length = minimum_description_length

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def extract(
        self,
        words: Sequence[Dict[str, Any]],
        columns: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract transaction rows from PDF words.

        Parameters
        ----------
        words:
            Output of pdfplumber.extract_words()

        columns:
            Semantic columns produced by UniversalStructureDetector.

        Returns
        -------
        List[Dict[str, Any]]
            Generic transaction rows.
        """

        if not words:
            return []

        if not columns:
            return []

        normalized_words = self._normalize_words(words)

        rows = self._group_rows(normalized_words)

        # Sort semantic columns from left to right.
        sorted_columns = sorted(
            columns,
            key=lambda column: (
                float(column.get("x0", 0)),
                float(column.get("x1", 0)),
            ),
        )

        results: List[Dict[str, Any]] = []

        for physical_row in rows:

            parsed = self._parse_row(
                physical_row,
                sorted_columns,
            )

            if parsed is None:
                continue

            if self._is_valid_transaction(
                parsed,
                sorted_columns,
            ):
                results.append(parsed)

        return results

    # ==============================================================
    # WORD NORMALIZATION
    # ==============================================================

    def _normalize_words(
        self,
        words: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        normalized = []

        for word in words:

            text = str(
                word.get("text", "")
            ).strip()

            if not text:
                continue

            try:
                x0 = float(word.get("x0", 0))
                x1 = float(word.get("x1", x0))
                top = float(word.get("top", 0))
                bottom = float(
                    word.get("bottom", top)
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            normalized.append(
                {
                    "text": text,
                    "x0": x0,
                    "x1": x1,
                    "top": top,
                    "bottom": bottom,
                    "center_x": (x0 + x1) / 2.0,
                    "center_y": (top + bottom) / 2.0,
                }
            )

        normalized.sort(
            key=lambda item: (
                item["top"],
                item["x0"],
            )
        )

        return normalized

    # ==============================================================
    # PHYSICAL ROW GROUPING
    # ==============================================================

    def _group_rows(
        self,
        words: Sequence[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:

        rows: List[List[Dict[str, Any]]] = []

        for word in words:

            if not rows:
                rows.append([word])
                continue

            current_row = rows[-1]

            average_top = sum(
                item["top"]
                for item in current_row
            ) / len(current_row)

            if abs(
                word["top"] - average_top
            ) <= self.row_tolerance:

                current_row.append(word)

            else:
                rows.append([word])

        for row in rows:
            row.sort(
                key=lambda item: item["x0"]
            )

        return rows

    # ==============================================================
    # ROW PARSING
    # ==============================================================

    def _parse_row(
        self,
        words: Sequence[Dict[str, Any]],
        columns: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        if not words:
            return None

        tokens = list(words)

        # ----------------------------------------------------------
        # DATE DETECTION
        # ----------------------------------------------------------

        date_candidates = self._find_date_candidates(
            tokens
        )

        if not date_candidates:
            return None

        used_indices = set()

        date_values = []

        for candidate in date_candidates:

            for index in candidate["indices"]:
                used_indices.add(index)

            date_values.append(candidate)

        # ----------------------------------------------------------
        # NUMERIC COLUMNS
        # ----------------------------------------------------------

        numeric_columns = [
            column
            for column in columns
            if column.get("semantic_type")
            in {
                "amount",
                "debit",
                "credit",
                "balance",
            }
        ]

        # ----------------------------------------------------------
        # NUMERIC TOKENS
        # ----------------------------------------------------------

        numeric_candidates = []

        for index, token in enumerate(tokens):

            if index in used_indices:
                continue

            text = token["text"].strip()

            if not self._is_number(text):
                continue

            # Avoid treating isolated day numbers as amounts.
            if self._looks_like_day_number(
                text,
                tokens,
                index,
            ):
                continue

            numeric_candidates.append(
                {
                    "index": index,
                    "text": text,
                    "x0": token["x0"],
                    "x1": token["x1"],
                    "center_x": token["center_x"],
                }
            )

        # ----------------------------------------------------------
        # ASSIGN NUMERIC VALUES TO SEMANTIC COLUMNS
        # ----------------------------------------------------------

        numeric_mapping = self._assign_numeric_fields(
            numeric_candidates,
            numeric_columns,
        )

        for index in numeric_mapping.values():

            if index is None:
                continue

            used_indices.add(index)

        # ----------------------------------------------------------
        # TRANSACTION ID
        # ----------------------------------------------------------

        transaction_id = self._extract_transaction_id(
            tokens=tokens,
            columns=columns,
            used_indices=used_indices,
        )

        if transaction_id:

            for index in transaction_id["indices"]:
                used_indices.add(index)

        # ----------------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------------

        description_tokens = []

        for index, token in enumerate(tokens):

            if index in used_indices:
                continue

            text = token["text"].strip()

            if not text:
                continue

            if text.upper() in self.IGNORED_TOKENS:
                continue

            description_tokens.append(text)

        description = " ".join(
            description_tokens
        ).strip()

        # ----------------------------------------------------------
        # BUILD RESULT
        # ----------------------------------------------------------

        result: Dict[str, Any] = {}

        # ----------------------------------------------------------
        # DATE VALUES
        # ----------------------------------------------------------

        self._assign_date_values(
            result,
            date_values,
            columns,
        )

        # ----------------------------------------------------------
        # NUMERIC VALUES
        # ----------------------------------------------------------

        for semantic, token_index in numeric_mapping.items():

            if token_index is None:
                continue

            token = tokens[token_index]

            value = self._clean_numeric_value(
                token["text"]
            )

            if value is None:
                continue

            result[semantic] = value

        # ----------------------------------------------------------
        # TRANSACTION ID
        # ----------------------------------------------------------

        if transaction_id:

            result["transaction_id"] = (
                transaction_id["text"]
            )

        # ----------------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------------

        if description:

            result["description"] = description

        return result

    # ==============================================================
    # DATE DETECTION
    # ==============================================================

    def _find_date_candidates(
        self,
        tokens: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        candidates = []

        index = 0

        while index < len(tokens):

            text = tokens[index]["text"].strip()

            # ------------------------------------------------------
            # Full date in a single PDF word.
            # ------------------------------------------------------

            if self._is_full_date(text):

                candidates.append(
                    {
                        "text": text,
                        "indices": [index],
                        "center_x": tokens[index]["center_x"],
                    }
                )

                index += 1
                continue

            # ------------------------------------------------------
            # Split date:
            #
            # 09 AUG
            #
            # PDF extraction often returns these as two separate
            # words.
            # ------------------------------------------------------

            if self._is_day(text):

                if index + 1 < len(tokens):

                    next_text = (
                        tokens[index + 1]["text"]
                        .strip()
                        .upper()
                        .rstrip(".")
                    )

                    if next_text in self.MONTHS:

                        combined = (
                            f"{text} {next_text}"
                        )

                        candidates.append(
                            {
                                "text": combined,
                                "indices": [
                                    index,
                                    index + 1,
                                ],
                                "center_x": (
                                    tokens[index]["center_x"]
                                    + tokens[index + 1]["center_x"]
                                ) / 2.0,
                            }
                        )

                        index += 2
                        continue

            index += 1

        # Keep dates in physical left-to-right order.
        candidates.sort(
            key=lambda item: item["center_x"]
        )

        return candidates

    # ==============================================================
    # DATE ASSIGNMENT
    # ==============================================================

    def _assign_date_values(
        self,
        result: Dict[str, Any],
        date_values: Sequence[Dict[str, Any]],
        columns: Sequence[Dict[str, Any]],
    ) -> None:

        date_columns = [
            column
            for column in columns
            if column.get("semantic_type")
            in {
                "transaction_date",
                "posting_date",
                "value_date",
            }
        ]

        date_columns.sort(
            key=lambda column: (
                float(column.get("x0", 0))
                + float(column.get("x1", 0))
            ) / 2.0
        )

        sorted_dates = sorted(
            date_values,
            key=lambda item: item["center_x"],
        )

        # ----------------------------------------------------------
        # Normal case:
        #
        # number of detected dates == number of date columns
        # ----------------------------------------------------------

        if len(sorted_dates) == len(date_columns):

            for column, date in zip(
                date_columns,
                sorted_dates,
            ):
                result[
                    column["semantic_type"]
                ] = date["text"]

            return

        # ----------------------------------------------------------
        # More dates than semantic date columns.
        # Assign the closest dates while maintaining order.
        # ----------------------------------------------------------

        if len(sorted_dates) > len(date_columns):

            selected = self._best_ordered_assignment(
                sorted_dates,
                date_columns,
                lambda date, column: abs(
                    date["center_x"]
                    - self._column_center(column)
                ),
            )

            for column, date in selected:

                result[
                    column["semantic_type"]
                ] = date["text"]

            return

        # ----------------------------------------------------------
        # Fewer dates than columns.
        #
        # Assign from left to right but leave unmatched
        # semantic columns empty.
        # ----------------------------------------------------------

        for date, column in zip(
            sorted_dates,
            date_columns,
        ):

            result[
                column["semantic_type"]
            ] = date["text"]

    # ==============================================================
    # NUMERIC FIELD ASSIGNMENT
    # ==============================================================

    def _assign_numeric_fields(
        self,
        numeric_candidates: Sequence[Dict[str, Any]],
        numeric_columns: Sequence[Dict[str, Any]],
    ) -> Dict[str, Optional[int]]:
        """
        Assign numeric tokens to numeric semantic columns.

        Important:

        This does NOT greedily assign each value to the nearest
        column.

        Greedy assignment caused the following problem:

            Federal:
                260.00
                2845.15

        Columns:

            Withdrawals
            Deposits
            Balance

        A greedy algorithm could assign:

            260.00  -> Withdrawals
            2845.15 -> Deposits

        even though the second value belongs to Balance.

        Instead, we search ordered column combinations and choose
        the lowest total horizontal distance.
        """

        mapping: Dict[str, Optional[int]] = {}

        if not numeric_candidates:
            return mapping

        if not numeric_columns:
            return mapping

        sorted_values = sorted(
            numeric_candidates,
            key=lambda item: item["center_x"],
        )

        sorted_columns = sorted(
            numeric_columns,
            key=self._column_center,
        )

        value_count = len(sorted_values)
        column_count = len(sorted_columns)

        # ----------------------------------------------------------
        # Case 1:
        # More/equal columns than numeric values.
        #
        # Example:
        #
        # values:
        #   260
        #   2845
        #
        # columns:
        #   debit
        #   credit
        #   balance
        #
        # Try every subset of columns and choose the best
        # left-to-right assignment.
        # ----------------------------------------------------------

        if value_count <= column_count:

            best_assignment = None
            best_cost = float("inf")

            for selected_columns in itertools.combinations(
                sorted_columns,
                value_count,
            ):

                cost = 0.0

                pairs = []

                valid = True

                for value, column in zip(
                    sorted_values,
                    selected_columns,
                ):

                    distance = abs(
                        value["center_x"]
                        - self._column_center(column)
                    )

                    # A very large horizontal jump is unlikely to
                    # represent the correct column.
                    if distance > 150:
                        valid = False
                        break

                    cost += distance

                    pairs.append(
                        (
                            column,
                            value,
                        )
                    )

                if valid and cost < best_cost:

                    best_cost = cost
                    best_assignment = pairs

            if best_assignment:

                for column, value in best_assignment:

                    mapping[
                        column["semantic_type"]
                    ] = value["index"]

            return mapping

        # ----------------------------------------------------------
        # Case 2:
        # More numeric values than numeric columns.
        #
        # This can happen when PDF extraction contains additional
        # numeric content inside the row.
        #
        # Choose the best subset of numeric values.
        # ----------------------------------------------------------

        best_assignment = None
        best_cost = float("inf")

        for selected_values in itertools.combinations(
            sorted_values,
            column_count,
        ):

            cost = 0.0

            pairs = []

            valid = True

            for value, column in zip(
                selected_values,
                sorted_columns,
            ):

                distance = abs(
                    value["center_x"]
                    - self._column_center(column)
                )

                if distance > 150:
                    valid = False
                    break

                cost += distance

                pairs.append(
                    (
                        column,
                        value,
                    )
                )

            if valid and cost < best_cost:

                best_cost = cost
                best_assignment = pairs

        if best_assignment:

            for column, value in best_assignment:

                mapping[
                    column["semantic_type"]
                ] = value["index"]

        return mapping

    # ==============================================================
    # TRANSACTION ID
    # ==============================================================

    def _extract_transaction_id(
        self,
        tokens: Sequence[Dict[str, Any]],
        columns: Sequence[Dict[str, Any]],
        used_indices: set,
    ) -> Optional[Dict[str, Any]]:

        id_columns = [
            column
            for column in columns
            if column.get("semantic_type")
            == "transaction_id"
        ]

        if not id_columns:
            return None

        id_column = min(
            id_columns,
            key=self._column_center,
        )

        id_center = self._column_center(
            id_column
        )

        candidates = []

        for index, token in enumerate(tokens):

            if index in used_indices:
                continue

            text = token["text"].strip()

            if not text:
                continue

            upper = text.upper()

            if self._looks_like_transaction_id(
                text
            ):

                candidates.append(
                    {
                        "index": index,
                        "text": text,
                        "distance": abs(
                            token["center_x"]
                            - id_center
                        ),
                    }
                )

        # ----------------------------------------------------------
        # No obvious ID.
        # ----------------------------------------------------------

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["distance"]
        )

        selected = candidates[0]

        selected_index = selected["index"]
        selected_text = selected["text"]

        selected_indices = [
            selected_index
        ]

        # ----------------------------------------------------------
        # Generic transaction marker + ID:
        #
        # TFR S63313909
        # TXN ABC12345
        # REF 123456789
        #
        # If a marker is directly adjacent to the selected ID,
        # combine them.
        # ----------------------------------------------------------

        for neighbor_index in (
            selected_index - 1,
            selected_index + 1,
        ):

            if (
                neighbor_index < 0
                or neighbor_index >= len(tokens)
            ):
                continue

            if neighbor_index in used_indices:
                continue

            neighbor_text = (
                tokens[neighbor_index]["text"]
                .strip()
            )

            if (
                neighbor_text.upper()
                in self.TRANSACTION_MARKERS
            ):

                selected_indices.append(
                    neighbor_index
                )

                selected_indices.sort()

                selected_text = " ".join(
                    tokens[index]["text"]
                    for index in selected_indices
                )

                break

        return {
            "text": selected_text,
            "indices": selected_indices,
        }

    # ==============================================================
    # VALIDATION
    # ==============================================================

    def _is_valid_transaction(
        self,
        row: Dict[str, Any],
        columns: Sequence[Dict[str, Any]],
    ) -> bool:
        """
        Validate a parsed row using the detected document structure.

        This is intentionally structural rather than bank-specific.
        """

        # ----------------------------------------------------------
        # DATE VALIDATION
        # ----------------------------------------------------------

        date_columns = [
            column
            for column in columns
            if column.get("semantic_type")
            in {
                "transaction_date",
                "posting_date",
                "value_date",
            }
        ]

        date_values = []

        for column in date_columns:

            semantic = column["semantic_type"]

            value = row.get(semantic)

            if value:
                date_values.append(value)

        if not date_values:
            return False

        # ----------------------------------------------------------
        # MONEY VALIDATION
        # ----------------------------------------------------------

        has_amount = (
            row.get("amount") is not None
            or row.get("debit") is not None
            or row.get("credit") is not None
        )

        if not has_amount:
            return False

        # ----------------------------------------------------------
        # DESCRIPTION VALIDATION
        # ----------------------------------------------------------

        description = (
            row.get("description") or ""
        ).strip()

        if len(description) < self.minimum_description_length:
            return False

        # ----------------------------------------------------------
        # TWO-DATE STRUCTURE VALIDATION
        #
        # If the detected document has two date columns, don't
        # accept a row containing only one date.
        #
        # This prevents statement-summary rows from becoming
        # transactions.
        # ----------------------------------------------------------

        if len(date_columns) >= 2:

            if len(date_values) < 2:
                return False

        # ----------------------------------------------------------
        # SUMMARY / STATEMENT-LEVEL ROW PROTECTION
        # ----------------------------------------------------------
        #
        # Rows containing multiple large numeric values and only
        # one date are suspicious in a two-date transaction table.
        #
        # The two-date requirement above already rejects these in
        # the normal case.
        #
        # This additional check protects against unusual extraction
        # cases where one date column wasn't populated.
        # ----------------------------------------------------------

        if len(date_columns) >= 2:

            if len(date_values) != len(date_columns):
                return False

        return True

    # ==============================================================
    # DATE HELPERS
    # ==============================================================

    def _is_full_date(
        self,
        text: str,
    ) -> bool:

        value = text.strip()

        for pattern in self.FULL_DATE_PATTERNS:

            if pattern.match(value):
                return True

        return False

    def _is_day(
        self,
        text: str,
    ) -> bool:

        value = text.strip()

        if not re.fullmatch(
            r"\d{1,2}",
            value,
        ):
            return False

        try:
            day = int(value)
        except ValueError:
            return False

        return 1 <= day <= 31

    def _looks_like_day_number(
        self,
        text: str,
        tokens: Sequence[Dict[str, Any]],
        index: int,
    ) -> bool:

        if not self._is_day(text):
            return False

        # If followed by a month, it is part of a split date.
        if index + 1 < len(tokens):

            next_text = (
                tokens[index + 1]["text"]
                .strip()
                .upper()
                .rstrip(".")
            )

            if next_text in self.MONTHS:
                return True

        # If preceded by a month, it is also part of a date.
        if index > 0:

            previous_text = (
                tokens[index - 1]["text"]
                .strip()
                .upper()
                .rstrip(".")
            )

            if previous_text in self.MONTHS:
                return True

        return False

    # ==============================================================
    # NUMBER HELPERS
    # ==============================================================

    def _is_number(
        self,
        text: str,
    ) -> bool:

        value = (
            text
            .strip()
            .replace("\u00a0", "")
        )

        if not value:
            return False

        return bool(
            self.NUMBER_PATTERN.match(value)
        )

    def _clean_numeric_value(
        self,
        text: str,
    ) -> Optional[float]:

        value = (
            text
            .strip()
            .replace(",", "")
            .replace("\u00a0", "")
        )

        if not value:
            return None

        # Remove accounting suffix.
        value = re.sub(
            r"(CR|DR)$",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        # Parentheses mean negative.
        negative = (
            value.startswith("(")
            and value.endswith(")")
        )

        value = value.strip("()")

        try:

            number = float(value)

            if negative:
                number = -number

            return number

        except ValueError:
            return None

    # ==============================================================
    # TRANSACTION ID HELPERS
    # ==============================================================

    def _looks_like_transaction_id(
        self,
        text: str,
    ) -> bool:

        value = text.strip()

        if not value:
            return False

        for pattern in self.TRANSACTION_ID_PATTERNS:

            if pattern.match(value):
                return True

        return False

    # ==============================================================
    # COLUMN HELPERS
    # ==============================================================

    def _column_center(
        self,
        column: Dict[str, Any],
    ) -> float:

        return (
            float(column.get("x0", 0))
            + float(column.get("x1", 0))
        ) / 2.0

    # ==============================================================
    # ORDERED ASSIGNMENT
    # ==============================================================

    def _best_ordered_assignment(
        self,
        values: Sequence[Any],
        columns: Sequence[Dict[str, Any]],
        distance_function,
    ) -> List[Tuple[Dict[str, Any], Any]]:
        """
        Assign values to columns while preserving horizontal order.
        """

        value_count = len(values)
        column_count = len(columns)

        if value_count == 0 or column_count == 0:
            return []

        if value_count <= column_count:

            best = None
            best_cost = float("inf")

            for selected_columns in itertools.combinations(
                columns,
                value_count,
            ):

                cost = 0.0
                pairs = []

                for value, column in zip(
                    values,
                    selected_columns,
                ):

                    distance = distance_function(
                        value,
                        column,
                    )

                    cost += distance

                    pairs.append(
                        (
                            column,
                            value,
                        )
                    )

                if cost < best_cost:

                    best_cost = cost
                    best = pairs

            return best or []

        # More values than columns.

        best = None
        best_cost = float("inf")

        for selected_values in itertools.combinations(
            values,
            column_count,
        ):

            cost = 0.0
            pairs = []

            for value, column in zip(
                selected_values,
                columns,
            ):

                distance = distance_function(
                    value,
                    column,
                )

                cost += distance

                pairs.append(
                    (
                        column,
                        value,
                    )
                )

            if cost < best_cost:

                best_cost = cost
                best = pairs

        return best or []


# ------------------------------------------------------------------
# Backward-compatible alias
# ------------------------------------------------------------------

RowRegionExtractor = UniversalRowExtractor