from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class UniversalRowExtractor:
    """
    Universal financial transaction row extractor.

    Bank-agnostic.

    Uses semantic columns from the structure detector together with
    PDF geometry.

    Handles:
        - normal single-token dates
        - dates split across PDF words
        - multi-token descriptions
        - generic amount columns
        - debit / credit columns
        - balance columns
        - transaction IDs
        - metadata rows
    """

    ROW_TOLERANCE = 4.0

    DATE_FIELDS = {
        "transaction_date",
        "posting_date",
        "value_date",
    }

    NUMERIC_FIELDS = {
        "amount",
        "debit",
        "credit",
        "balance",
    }

    TRANSACTION_AMOUNT_FIELDS = {
        "amount",
        "debit",
        "credit",
    }

    IGNORE_WORDS = {
        "DR",
        "CR",
    }

    METADATA_TERMS = {
        "opening balance",
        "closing balance",
        "balance forward",
        "brought forward",
        "carried forward",
        "card limit",
        "available limit",
        "credit limit",
        "minimum payment",
        "current balance",
        "statement balance",
        "total balance",
        "account balance",
        "available balance",
        "reward points",
        "reward plus",
        "scheme opening",
        "points accrued",
        "points redeemed",
        "adjustment bonus",
        "warning statements",
        "bank deposits are covered",
        "insurance scheme",
        "nomination details",
        "report irregularities",
        "contacting our branch",
        "abbreviations used",
        "disclaimer",
        "end of statement",
        "computer generated statement",
        "licensed by",
        "corporate office",
        "page of",
        "address last updated",
        "regd. mobile number",
        "customer id",
        "email id",
        "account open date",
        "type of account",
        "account status",
        "effective available balance",
        "date of issue",
        "ifsc",
        "micr code",
        "swift code",
        "nomination",
    }

    MONTHS = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    @staticmethod
    def _column_center(column):
        """Return a column's horizontal center without assuming a
        particular structure-detector dictionary shape."""
        center = column.get("center")
        if center is not None:
            return float(center)

        x0 = column.get("x0")
        x1 = column.get("x1")
        if x0 is not None and x1 is not None:
            return (float(x0) + float(x1)) / 2.0

        left = column.get("left")
        right = column.get("right")
        if left is not None and right is not None:
            return (float(left) + float(right)) / 2.0

        x = column.get("x")
        width = column.get("width")
        if x is not None and width is not None:
            return float(x) + float(width) / 2.0

        raise KeyError(f"Cannot determine column center: {column}")

    # =============================================================
    # MAIN
    # =============================================================

    def extract(
        self,
        words: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not words or not columns:
            return []

        normalized_columns = self._normalize_columns(
            columns=columns,
            words=words,
        )

        if not normalized_columns:
            return []

        has_bank_shape = (
            any(
                column["semantic_type"]
                in {"debit", "credit"}
                for column in normalized_columns
            )
            and any(
                column["semantic_type"] == "balance"
                for column in normalized_columns
            )
        )

        statement_year = self._infer_statement_year(
            words
        )

        if has_bank_shape:
            transactions = (
                self._extract_logical_transactions(
                    words=words,
                    columns=normalized_columns,
                )
            )
        else:
            header_top = max(
                float(column.get("top", 0))
                for column in normalized_columns
            )

            data_words = [
                word
                for word in words
                if float(word.get("top", 0))
                > header_top + 8
            ]

            rows = self._group_rows(
                data_words
            )

            transactions = []

            for row in rows:

                parsed = self._parse_row(
                    row,
                    normalized_columns,
                )

                if self._is_transaction(
                    parsed
                ):
                    transactions.append(
                        parsed
                    )

        if statement_year is not None:

            for transaction in transactions:
                self._resolve_statement_year(
                    transaction,
                    statement_year,
                )

        return transactions

    def _infer_statement_year(
        self,
        words,
    ):
        years = []

        for word in words:

            text = str(
                word.get("text", "")
            ).strip()

            match = re.fullmatch(
                r"(19\d{2}|20\d{2}|21\d{2})",
                text,
            )

            if match:
                years.append(
                    int(match.group(1))
                )

        if not years:
            return None

        counts = {}

        for year in years:
            counts[year] = (
                counts.get(year, 0)
                + 1
            )

        return max(
            counts,
            key=counts.get,
        )

    def _resolve_statement_year(
        self,
        row,
        statement_year,
    ):
        for key, value in list(
            row.items()
        ):

            if (
                isinstance(value, str)
                and value.startswith("0001-")
            ):
                row[key] = (
                    f"{statement_year}"
                    f"{value[4:]}"
                )

    def _is_metadata_row(
        self,
        row,
    ):
        if not row:
            return True

        description = str(
            row.get(
                "description",
                "",
            )
        ).strip().lower()

        if description in {
            "total",
            "grand total",
        }:
            return True

        text = " ".join(
            str(value).lower()
            for value in row.values()
            if value is not None
        )

        for term in self.METADATA_TERMS:
            if term in text:
                return True

        return False

    def _extract_logical_transactions(self, words, columns):
        """
        Extract transactions from bank-style tables.

        The key invariant is:

            amount + balance on a physical row = transaction boundary

        The date is context, not the boundary.  Banks frequently print the
        date only on the first transaction of a block and leave it blank for
        following transactions.  Descriptions can also wrap over several PDF
        lines.

        Therefore we:
          1. find physical rows containing a transaction amount and balance;
          2. use those rows as transaction anchors;
          3. inherit the latest transaction date when the date cell is blank;
          4. collect description text from the anchor until the next anchor;
          5. assign debit/credit from the actual column geometry.
        """
        if not words or not columns:
            return []

        date_columns = [
            c for c in columns
            if c.get("semantic_type") in self.DATE_FIELDS
        ]

        balance_columns = [
            c for c in columns
            if c.get("semantic_type") == "balance"
        ]

        amount_columns = [
            c for c in columns
            if c.get("semantic_type") in {
                "amount", "debit", "credit"
            }
        ]

        if not balance_columns or not amount_columns:
            return []

        balance_column = min(
            balance_columns,
            key=lambda c: self._column_center(c),
        )

        # A bank-shaped page is normally already page-scoped by the PDF
        # reader.  Still support page metadata when it exists, so the
        # extractor remains safe if a caller passes several pages at once.
        def explicit_page_key(word):
            for key in (
                "page",
                "page_number",
                "page_num",
                "source_page",
            ):
                if key in word:
                    return (key, word[key])
            return None

        if any(explicit_page_key(w) is not None for w in words):
            grouped = {}
            order = []
            for word in words:
                key = explicit_page_key(word)
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(word)
            pages = [grouped[key] for key in order]
        else:
            pages = [words]

        def token_geometry(word):
            x0 = float(word.get("x0", 0))
            x1 = float(word.get("x1", x0))
            top = float(word.get("top", 0))
            bottom = float(word.get("bottom", top))
            return x0, x1, top, bottom, (x0 + x1) / 2.0

        def is_numeric_word(word):
            return self._is_number(
                str(word.get("text", "")).strip()
            )

        def numeric_value(text):
            value = (
                str(text)
                .strip()
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .replace("₹", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
                .replace("AED", "")
                .replace("INR", "")
                .replace("USD", "")
                .replace("EUR", "")
                .strip()
            )
            return value

        def word_in_column(word, column, tolerance=10.0):
            _, _, _, _, center = token_geometry(word)
            left = float(column.get("x0", 0))
            right = float(column.get("x1", left))
            return (
                left - tolerance
                <= center
                <= right + tolerance
            )

        def rowize(page_words):
            return self._group_rows(page_words)

        def date_anchors(page_words, column):
            """
            Return (top, parsed_date) anchors for one date column.

            _find_date_groups handles split dates such as 16 | Jun | 19.
            Complete date words are added as a fallback.
            """
            anchors = []

            groups = self._find_date_groups(
                page_words,
                [column],
            )

            center = self._column_center(column)

            for group in groups:
                if abs(group["center"] - center) > 18:
                    continue

                indexes = group.get("indexes", [])
                if not indexes:
                    continue

                first = page_words[indexes[0]]
                anchors.append(
                    (
                        float(first.get("top", 0)),
                        group["value"],
                    )
                )

            for word in page_words:
                text = str(word.get("text", "")).strip()
                parsed = self._parse_date_text(text)
                if parsed is None:
                    continue

                _, _, top, _, word_center = token_geometry(word)

                if abs(word_center - center) > 18:
                    continue

                anchors.append((top, parsed))

            anchors.sort(key=lambda item: item[0])

            # De-duplicate date entries on the same physical line.
            result = []
            for top, value in anchors:
                if result and abs(top - result[-1][0]) <= self.ROW_TOLERANCE:
                    # Prefer a real four-digit/two-digit-year value over a
                    # duplicate sentinel value if both were detected.
                    result[-1] = (
                        result[-1][0],
                        value,
                    )
                else:
                    result.append((top, value))

            return result

        def latest_date(anchors, top):
            selected = None
            for anchor_top, value in anchors:
                if anchor_top <= top + self.ROW_TOLERANCE:
                    selected = value
                else:
                    break
            return selected

        def nearest_date_on_row(page_words, column, row_top):
            if column is None:
                return None

            candidates = []
            center = self._column_center(column)

            for word in page_words:
                top = float(word.get("top", 0))
                if abs(top - row_top) > self.ROW_TOLERANCE:
                    continue

                text = str(word.get("text", "")).strip()
                parsed = self._parse_date_text(text)
                if parsed is None:
                    continue

                _, _, _, _, word_center = token_geometry(word)
                distance = abs(word_center - center)
                if distance <= 90:
                    candidates.append((distance, parsed))

            if not candidates:
                # Split date fallback: inspect date groups whose first token
                # is on this row.
                groups = self._find_date_groups(
                    page_words,
                    [column],
                )
                for group in groups:
                    indexes = group.get("indexes", [])
                    if not indexes:
                        continue
                    first = page_words[indexes[0]]
                    top = float(first.get("top", 0))
                    if abs(top - row_top) <= self.ROW_TOLERANCE:
                        distance = abs(
                            group["center"]
                            - center
                        )
                        if distance <= 90:
                            candidates.append(
                                (distance, group["value"])
                            )

            if not candidates:
                return None

            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        transactions = []

        for page_words in pages:
            if not page_words:
                continue

            page_words = sorted(
                page_words,
                key=lambda word: (
                    float(word.get("top", 0)),
                    float(word.get("x0", 0)),
                ),
            )

            rows = rowize(page_words)

            tx_date_column = None
            if date_columns:
                tx_date_column = min(
                    date_columns,
                    key=self._column_center,
                )

            value_date_column = None
            if len(date_columns) > 1:
                sorted_dates = sorted(
                    date_columns,
                    key=self._column_center,
                )
                value_date_column = sorted_dates[1]

            tx_date_anchors = (
                date_anchors(
                    page_words,
                    tx_date_column,
                )
                if tx_date_column is not None
                else []
            )

            # ---------------------------------------------------------
            # PHYSICAL TRANSACTION BOUNDARIES
            # ---------------------------------------------------------
            #
            # A balance-only row such as "BALANCE FORWARD" is NOT a
            # transaction.  A transaction boundary needs:
            #
            #     transaction amount + balance
            #
            # on the same physical row.
            #
            boundaries = []

            for row in rows:
                if not row:
                    continue

                row_top = min(
                    float(word.get("top", 0))
                    for word in row
                )

                balance_tokens = [
                    word for word in row
                    if (
                        is_numeric_word(word)
                        and word_in_column(
                            word,
                            balance_column,
                            tolerance=14.0,
                        )
                    )
                ]

                if not balance_tokens:
                    continue

                # Find candidate transaction amount tokens.  Balance itself
                # is excluded because its x-position belongs to the balance
                # column.
                amount_tokens = []
                for word in row:
                    if not is_numeric_word(word):
                        continue

                    _, _, _, _, center = token_geometry(word)

                    if abs(
                        center
                        - self._column_center(balance_column)
                    ) <= 25:
                        continue

                    for column in amount_columns:
                        if column.get("semantic_type") == "balance":
                            continue

                        if word_in_column(
                            word,
                            column,
                            tolerance=14.0,
                        ):
                            distance = abs(
                                center
                                - self._column_center(column)
                            )
                            amount_tokens.append(
                                (
                                    distance,
                                    word,
                                    column,
                                )
                            )
                            break

                if not amount_tokens:
                    continue

                # There should normally be exactly one transaction amount.
                # Pick the geometrically strongest candidate.
                amount_tokens.sort(
                    key=lambda item: item[0]
                )

                _, amount_word, amount_column = amount_tokens[0]

                # If multiple balance numbers happen to be present, use the
                # closest one to the balance column.
                balance_tokens.sort(
                    key=lambda word: abs(
                        token_geometry(word)[4]
                        - self._column_center(balance_column)
                    )
                )

                balance_word = balance_tokens[0]

                boundaries.append(
                    {
                        "top": row_top,
                        "row": row,
                        "amount_word": amount_word,
                        "amount_column": amount_column,
                        "balance_word": balance_word,
                    }
                )

            # De-duplicate boundaries that came from the same physical row.
            unique_boundaries = []
            for boundary in boundaries:
                if (
                    unique_boundaries
                    and abs(
                        boundary["top"]
                        - unique_boundaries[-1]["top"]
                    ) <= self.ROW_TOLERANCE
                ):
                    continue
                unique_boundaries.append(boundary)

            boundaries = unique_boundaries

            if not boundaries:
                continue

            # ---------------------------------------------------------
            # BUILD TRANSACTIONS BETWEEN AMOUNT+BALANCE ANCHORS
            # ---------------------------------------------------------
            for index, boundary in enumerate(boundaries):
                start = boundary["top"]
                if index + 1 < len(boundaries):
                    end = boundaries[index + 1]["top"]
                else:
                    end = float("inf")

                region = [
                    word
                    for word in page_words
                    if (
                        start - self.ROW_TOLERANCE
                        <= float(word.get("top", 0))
                        < end - self.ROW_TOLERANCE
                    )
                ]

                if not region:
                    continue

                tx_date = latest_date(
                    tx_date_anchors,
                    boundary["top"],
                )

                if tx_date is None:
                    # No date context means this is not safe to promote to a
                    # canonical transaction.  The caller can still handle
                    # other table shapes through the generic path.
                    continue

                row = {
                    "transaction_date": tx_date,
                }

                # Value/posting date is usually printed on the transaction's
                # first physical line.  If it is blank, inherit the last
                # available value date anchor rather than guessing.
                if value_date_column is not None:
                    value_date = nearest_date_on_row(
                        page_words,
                        value_date_column,
                        boundary["top"],
                    )

                    if value_date is not None:
                        row["value_date"] = value_date

                # -----------------------------------------------------
                # AMOUNT + DIRECTION
                # -----------------------------------------------------
                amount_text = str(
                    boundary["amount_word"].get(
                        "text",
                        "",
                    )
                ).strip()

                semantic = boundary["amount_column"].get(
                    "semantic_type"
                )

                # Explicit DR/CR markers, when present, override ambiguous
                # geometry.  Search only on the same physical row and near
                # the selected amount.
                amount_x1 = float(
                    boundary["amount_word"].get(
                        "x1",
                        boundary["amount_word"].get("x0", 0),
                    )
                )
                amount_top = float(
                    boundary["amount_word"].get(
                        "top",
                        boundary["top"],
                    )
                )

                marker = None
                marker_score = None

                for word in boundary["row"]:
                    marker_text = str(
                        word.get("text", "")
                    ).strip().upper()

                    if marker_text not in self.IGNORE_WORDS:
                        continue

                    marker_x0 = float(word.get("x0", 0))
                    marker_top = float(
                        word.get(
                            "top",
                            boundary["top"],
                        )
                    )

                    vertical_distance = abs(
                        marker_top - amount_top
                    )
                    horizontal_distance = abs(
                        marker_x0 - amount_x1
                    )

                    if (
                        vertical_distance <= 8
                        and horizontal_distance <= 35
                    ):
                        score = (
                            vertical_distance
                            + horizontal_distance
                        )
                        if (
                            marker_score is None
                            or score < marker_score
                        ):
                            marker = marker_text
                            marker_score = score

                if marker == "DR":
                    semantic = "debit"
                elif marker == "CR":
                    semantic = "credit"

                if semantic in {
                    "debit",
                    "credit",
                    "amount",
                }:
                    row[semantic] = amount_text

                balance_text = str(
                    boundary["balance_word"].get(
                        "text",
                        "",
                    )
                ).strip()

                row["balance"] = balance_text

                # -----------------------------------------------------
                # DESCRIPTION
                # -----------------------------------------------------
                #
                # Do not use a fixed bank-specific description width.
                # Use the detected column geometry and the first numeric
                # column as the right edge.  This keeps the method usable
                # across different bank layouts.
                description_column = next(
                    (
                        c for c in columns
                        if c.get("semantic_type")
                        == "description"
                    ),
                    None,
                )

                if description_column is not None:
                    desc_left = float(
                        description_column.get(
                            "x0",
                            0,
                        )
                    ) - 8.0

                    numeric_left_candidates = [
                        float(c.get("x0", 0))
                        for c in amount_columns
                        if c.get("semantic_type")
                        != "balance"
                    ]

                    desc_right = (
                        min(numeric_left_candidates) - 5.0
                        if numeric_left_candidates
                        else 620.0
                    )
                else:
                    desc_left = max(
                        float(c.get("x1", 0))
                        for c in date_columns
                    ) + 5.0 if date_columns else 0.0

                    desc_right = (
                        min(
                            float(c.get("x0", 0))
                            for c in amount_columns
                            if c.get("semantic_type")
                            != "balance"
                        )
                        - 5.0
                        if any(
                            c.get("semantic_type")
                            != "balance"
                            for c in amount_columns
                        )
                        else 620.0
                    )

                description_parts = []

                # Statement footers often appear after the final transaction
                # on the same PDF page.  They can be physically inside the
                # description column, so filter them by physical line rather
                # than letting them poison the transaction.
                description_rows = self._group_rows(region)
                metadata_started = False

                for physical_row in description_rows:
                    row_text = " ".join(
                        str(w.get("text", "")).strip()
                        for w in physical_row
                        if str(w.get("text", "")).strip()
                    ).lower()

                    if any(
                        term in row_text
                        for term in self.METADATA_TERMS
                    ):
                        # Footer/boilerplate starts here.  Do not let later
                        # footer lines become part of the transaction.
                        if (
                            float(physical_row[0].get("top", 0))
                            > boundary["top"] + 20.0
                        ):
                            metadata_started = True
                            break

                    if metadata_started:
                        break

                    for word in physical_row:
                        text = str(
                            word.get("text", "")
                        ).strip()

                        if not text:
                            continue

                        _, _, _, _, center = token_geometry(word)

                        if not (
                            desc_left
                            <= center
                            <= desc_right
                        ):
                            continue

                        if self._parse_date_text(text) is not None:
                            continue

                        if self._is_number(text):
                            continue

                        if text.upper() in self.IGNORE_WORDS:
                            continue

                        # A transaction reference can contain numbers,
                        # slashes and hyphens.  Preserve it unless it is a
                        # pure numeric token.
                        description_parts.append(
                            (
                                float(word.get("top", 0)),
                                float(word.get("x0", 0)),
                                text,
                            )
                        )

                description_parts.sort(
                    key=lambda item: (
                        item[0],
                        item[1],
                    )
                )

                if description_parts:
                    row["description"] = " ".join(
                        item[2]
                        for item in description_parts
                    ).strip()

                # -----------------------------------------------------
                # FINAL TRANSACTION GUARD
                # -----------------------------------------------------
                if self._is_transaction(row):
                    transactions.append(row)

        return transactions

    # =============================================================
    # COLUMN NORMALIZATION
    # =============================================================

    def _normalize_columns(
        self,
        columns,
        words,
    ):

        normalized = []

        for column in columns:

            if not isinstance(column, dict):
                continue

            semantic_type = (
                column.get("semantic_type")
                or column.get("type")
            )

            if not semantic_type:
                continue

            header = str(
                column.get(
                    "header",
                    column.get("name", ""),
                )
            ).strip()

            try:
                x0 = float(column.get("x0", 0))
            except Exception:
                x0 = 0.0

            try:
                x1 = float(
                    column.get(
                        "x1",
                        x0 + 30,
                    )
                )
            except Exception:
                x1 = x0 + 30

            try:
                top = float(column.get("top", 0))
            except Exception:
                top = 0.0

            if x1 < x0:
                x0, x1 = x1, x0

            normalized.append(
                {
                    **column,
                    "header": header,
                    "semantic_type": semantic_type,
                    "x0": x0,
                    "x1": x1,
                    "top": top,
                    "center": (x0 + x1) / 2,
                }
            )

        normalized = self._resolve_split_value_date(
            normalized,
            words,
        )

        normalized.sort(
            key=lambda column: column["x0"]
        )

        return normalized

    def _resolve_split_value_date(
        self,
        columns,
        words,
    ):

        if not columns or not words:
            return columns

        date_word_candidates = []

        for word in words:

            text = str(
                word.get("text", "")
            ).strip()

            if text.upper() != "DATE":
                continue

            try:
                x0 = float(word.get("x0", 0))
                x1 = float(word.get("x1", x0))
                top = float(word.get("top", 0))
            except Exception:
                continue

            date_word_candidates.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "center": (x0 + x1) / 2,
                    "top": top,
                }
            )

        if not date_word_candidates:
            return columns

        for column in columns:

            header = str(
                column.get("header", "")
            ).strip().lower()

            if header != "value":
                continue

            if column.get("semantic_type") != "amount":
                continue

            column_center = self._column_center(column)
            column_top = column["top"]

            matches = []

            for date_word in date_word_candidates:

                horizontal_distance = abs(
                    date_word["center"]
                    - column_center
                )

                vertical_distance = (
                    date_word["top"]
                    - column_top
                )

                if (
                    horizontal_distance <= 12
                    and 0 < vertical_distance <= 20
                ):
                    matches.append(
                        (
                            horizontal_distance
                            + vertical_distance,
                            date_word,
                        )
                    )

            if matches:

                matches.sort(
                    key=lambda item: item[0]
                )

                column["semantic_type"] = "value_date"
                column["header"] = "Value Date"
                column["confidence"] = 1.0

        return columns

    # =============================================================
    # ROW GROUPING
    # =============================================================

    def _group_rows(
        self,
        words,
    ):

        words = sorted(
            words,
            key=lambda word: (
                float(word.get("top", 0)),
                float(word.get("x0", 0)),
            ),
        )

        rows = []

        for word in words:

            top = float(
                word.get("top", 0)
            )

            matched_row = None

            for row in rows:

                row_top = float(
                    row[0].get("top", 0)
                )

                if abs(top - row_top) <= self.ROW_TOLERANCE:
                    matched_row = row
                    break

            if matched_row is None:
                rows.append([word])
            else:
                matched_row.append(word)

        for row in rows:
            row.sort(
                key=lambda word: float(
                    word.get("x0", 0)
                )
            )

        return rows

    # =============================================================
    # ROW PARSING
    # =============================================================

    def _parse_row(
        self,
        words,
        columns,
    ):

        tokens = []

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
            except Exception:
                continue

            tokens.append(
                {
                    "text": text,
                    "x0": x0,
                    "x1": x1,
                    "center": (x0 + x1) / 2,
                    "top": top,
                    "bottom": bottom,
                }
            )

        if not tokens:
            return {}

        result = {}
        used_indexes = set()

        # ---------------------------------------------------------
        # DATE COLUMNS
        # ---------------------------------------------------------

        date_columns = [
            column
            for column in columns
            if column["semantic_type"]
            in self.DATE_FIELDS
        ]

        # First try complete date tokens.
        complete_dates = [
            (
                index,
                token,
            )
            for index, token in enumerate(tokens)
            if self._is_date(token["text"])
        ]

        # Then try reconstructing split dates from adjacent PDF words.
        reconstructed_dates = self._find_date_groups(
            tokens,
            date_columns,
        )

        assigned_date_indexes = set()

        for index, column in enumerate(date_columns):

            semantic = column["semantic_type"]

            # Prefer a date group that is geometrically close to
            # this date column.
            group = self._select_date_group(
                reconstructed_dates,
                column,
                assigned_date_indexes,
            )

            if group is not None:

                result[semantic] = group["value"]

                for token_index in group["indexes"]:
                    used_indexes.add(token_index)
                    assigned_date_indexes.add(token_index)

                continue

            # Fallback to a complete date token.
            available = [
                item
                for item in complete_dates
                if item[0] not in used_indexes
            ]

            if not available:
                continue

            selected = min(
                available,
                key=lambda item: self._distance_to_column(
                    item[1],
                    column,
                ),
            )

            token_index, token = selected

            result[semantic] = token["text"]

            used_indexes.add(token_index)

        # ---------------------------------------------------------
        # TRANSACTION ID
        # ---------------------------------------------------------

        transaction_column = next(
            (
                column
                for column in columns
                if column["semantic_type"]
                == "transaction_id"
            ),
            None,
        )

        if transaction_column:

            candidates = []

            for index, token in enumerate(tokens):

                if index in used_indexes:
                    continue

                if self._looks_like_transaction_id(
                    token["text"]
                ):
                    candidates.append(
                        (
                            index,
                            token,
                        )
                    )

            if candidates:

                selected = min(
                    candidates,
                    key=lambda item: self._distance_to_column(
                        item[1],
                        transaction_column,
                    ),
                )

                index, token = selected

                value = token["text"]

                if value.upper() == "TFR":

                    if index + 1 < len(tokens):

                        next_index = index + 1
                        next_token = tokens[next_index]

                        if self._looks_like_id(
                            next_token["text"]
                        ):

                            value = (
                                "TFR "
                                + next_token["text"]
                            )

                            used_indexes.add(next_index)

                result["transaction_id"] = value
                used_indexes.add(index)

        # ---------------------------------------------------------
        # NUMERIC COLUMNS
        # ---------------------------------------------------------

        numeric_columns = [
            column
            for column in columns
            if column["semantic_type"]
            in self.NUMERIC_FIELDS
        ]

        numeric_tokens = [
            (
                index,
                token,
            )
            for index, token in enumerate(tokens)
            if (
                index not in used_indexes
                and self._is_number(token["text"])
            )
        ]

        assignments = self._assign_numeric_tokens(
            numeric_tokens,
            numeric_columns,
        )

        for semantic, assignment in assignments.items():

            if assignment is None:
                continue

            token_index, token = assignment

            result[semantic] = token["text"]

            used_indexes.add(token_index)

        # ---------------------------------------------------------
        # DESCRIPTION
        # ---------------------------------------------------------

        description_column = next(
            (
                column
                for column in columns
                if column["semantic_type"]
                == "description"
            ),
            None,
        )

        if description_column:

            description_tokens = []

            for index, token in enumerate(tokens):

                if index in used_indexes:
                    continue

                text = token["text"]

                if not text:
                    continue

                if text.upper() in self.IGNORE_WORDS:
                    continue

                if self._is_date(text):
                    continue

                if self._is_number(text):
                    continue

                if self._looks_like_date_fragment(text):
                    continue

                if (
                    text.upper() == "TFR"
                    and "transaction_id" in result
                ):
                    continue

                if (
                    "transaction_id" in result
                    and self._looks_like_id(text)
                ):
                    continue

                description_tokens.append(token)

            description_tokens.sort(
                key=lambda token: token["x0"]
            )

            if description_tokens:

                result["description"] = " ".join(
                    token["text"]
                    for token in description_tokens
                ).strip()

        return result

    # =============================================================
    # DATE GROUPING
    # =============================================================

    def _find_date_groups(
        self,
        tokens,
        date_columns,
    ):
        """
        Detect dates that are split across multiple PDF words.

        Examples:

            16 | Jun | 19
            16 Jun | 19
            16 | June | 2019
            2019 | Jun | 16
            16 / 06 / 2019

        The method is based on token text + geometry and does not
        depend on a particular bank or statement format.
        """

        groups = []

        if not tokens:
            return groups

        # PDF reader words normally contain x0/x1 but do not necessarily
        # contain a precomputed center.  The date-grouping logic needs a
        # center, so derive it locally instead of assuming the reader
        # supplied one.
        normalized_tokens = []

        for token in tokens:
            if "center" in token:
                normalized_tokens.append(token)
                continue

            try:
                x0 = float(token.get("x0", 0))
                x1 = float(token.get("x1", x0))
            except Exception:
                x0 = 0.0
                x1 = 0.0

            normalized = dict(token)
            normalized["center"] = (x0 + x1) / 2.0
            normalized_tokens.append(normalized)

        tokens = normalized_tokens

        max_gap = 18.0

        for start in range(len(tokens)):

            first = tokens[start]

            if not self._looks_like_date_start(
                first["text"]
            ):
                continue

            candidate_indexes = [start]

            current = first

            for index in range(
                start + 1,
                min(start + 4, len(tokens)),
            ):

                candidate = tokens[index]

                horizontal_gap = (
                    candidate["x0"]
                    - current["x1"]
                )

                vertical_difference = abs(
                    candidate["top"]
                    - first["top"]
                )

                # Date parts on a normal transaction row should
                # be horizontally close.
                if vertical_difference > self.ROW_TOLERANCE:
                    break

                if horizontal_gap > max_gap:
                    break

                candidate_indexes.append(index)
                current = candidate

                text = " ".join(
                    tokens[i]["text"]
                    for i in candidate_indexes
                )

                parsed = self._parse_date_text(text)

                if parsed is not None:

                    if len(candidate_indexes) == 2:
                        next_index = index + 1
                        if next_index < len(tokens):
                            next_text = str(
                                tokens[next_index]["text"]
                            ).strip()
                            if re.fullmatch(
                                r"\d{2,4}",
                                next_text,
                            ):
                                year_number = int(next_text)
                                if (
                                    year_number <= 99
                                    or 1900 <= year_number <= 2100
                                ):
                                    continue

                    groups.append(
                        {
                            "indexes": list(candidate_indexes),
                            "value": parsed,
                            "center": (
                                min(
                                    tokens[i]["x0"]
                                    for i in candidate_indexes
                                )
                                + max(
                                    tokens[i]["x1"]
                                    for i in candidate_indexes
                                )
                            )
                            / 2,
                        }
                    )

                    break

        # Also handle dates where PDF extraction stacked the
        # components vertically inside the same date cell.
        for start in range(len(tokens)):

            first = tokens[start]

            if not self._looks_like_date_start(
                first["text"]
            ):
                continue

            candidate_indexes = [start]

            for index in range(
                start + 1,
                min(start + 4, len(tokens)),
            ):

                candidate = tokens[index]

                center_distance = abs(
                    candidate["center"]
                    - first["center"]
                )

                vertical_gap = (
                    candidate["top"]
                    - tokens[candidate_indexes[-1]]["bottom"]
                )

                if center_distance > 18:
                    break

                if vertical_gap < -2:
                    continue

                if vertical_gap > 14:
                    break

                candidate_indexes.append(index)

                text = " ".join(
                    tokens[i]["text"]
                    for i in candidate_indexes
                )

                parsed = self._parse_date_text(text)

                if parsed is not None:

                    if len(candidate_indexes) == 2:
                        next_index = index + 1
                        if next_index < len(tokens):
                            next_text = str(
                                tokens[next_index]["text"]
                            ).strip()
                            if re.fullmatch(
                                r"\d{2,4}",
                                next_text,
                            ):
                                year_number = int(next_text)
                                if (
                                    year_number <= 99
                                    or 1900 <= year_number <= 2100
                                ):
                                    continue

                    groups.append(
                        {
                            "indexes": list(candidate_indexes),
                            "value": parsed,
                            "center": (
                                min(
                                    tokens[i]["x0"]
                                    for i in candidate_indexes
                                )
                                + max(
                                    tokens[i]["x1"]
                                    for i in candidate_indexes
                                )
                            )
                            / 2,
                        }
                    )

                    break

        # Remove duplicate groups.
        unique = {}

        for group in groups:

            key = (
                tuple(group["indexes"]),
                group["value"],
            )

            unique[key] = group

        return list(unique.values())

    def _select_date_group(
        self,
        groups,
        column,
        assigned_indexes,
    ):
        if not groups:
            return None

        candidates = []

        for group in groups:

            if any(
                index in assigned_indexes
                for index in group["indexes"]
            ):
                continue

            distance = abs(
                group["center"]
                - self._column_center(column)
            )

            candidates.append(
                (
                    distance,
                    group,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0]
        )

        # Prefer groups reasonably close to the date column.
        # The row extractor still remains geometry-driven.
        best_distance, best_group = candidates[0]

        if best_distance <= 80:
            return best_group

        return None

    # =============================================================
    # DATE PARSING
    # =============================================================

    def _parse_date_text(
        self,
        value: str,
    ) -> Optional[str]:

        text = str(value).strip()

        if not text:
            return None

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Normalize common separators.
        normalized = re.sub(
            r"\s*([/.\-])\s*",
            r"\1",
            text,
        )

        formats = (
            "%d %b %y",
            "%d-%b-%Y",
            "%d-%b-%y",
            "%d-%B-%Y",
            "%d-%B-%y",
            "%d %B %Y",
            "%d %b %Y",
            "%d %B %y",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%d-%m-%Y",
            "%d-%m-%y",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%d.%m.%y",
        )

        for fmt in formats:

            try:

                parsed = datetime.strptime(
                    normalized,
                    fmt,
                )

                return parsed.strftime(
                    "%Y-%m-%d"
                )

            except ValueError:
                continue

        # Handle textual month without relying on
        # locale-sensitive datetime parsing.
        parts = normalized.split()

        # Full textual date: 09 AUG 2026 / 09 AUG 26
        if len(parts) == 3:

            first, second, third = parts

            month = self.MONTHS.get(
                second.lower().rstrip(".")
            )

            if month is not None:

                if (
                    first.isdigit()
                    and third.isdigit()
                ):

                    day = int(first)
                    year = self._normalize_year(
                        int(third)
                    )

                    if self._valid_date(
                        year,
                        month,
                        day,
                    ):
                        return (
                            f"{year:04d}-"
                            f"{month:02d}-"
                            f"{day:02d}"
                        )

        # Two-token textual date: 09 AUG
        #
        # Many credit-card statements omit the year from every
        # transaction row because the statement period already gives
        # the year. The row extractor intentionally returns a marker
        # here; the normalizer/pipeline can resolve the statement year
        # when available. For extraction purposes, use a safe sentinel
        # year so the date group is recognized.
        if len(parts) == 2:

            first, second = parts

            month = self.MONTHS.get(
                second.lower().rstrip(".")
            )

            if (
                month is not None
                and first.isdigit()
            ):

                day = int(first)

                if 1 <= day <= 31:
                    return (
                        f"0001-{month:02d}-"
                        f"{day:02d}"
                    )

        return None

    def _normalize_year(
        self,
        year: int,
    ) -> int:

        if year < 100:

            if year <= 49:
                return 2000 + year

            return 1900 + year

        return year

    def _valid_date(
        self,
        year: int,
        month: int,
        day: int,
    ) -> bool:

        try:
            datetime(
                year,
                month,
                day,
            )
            return True
        except ValueError:
            return False

    def _looks_like_date_start(
        self,
        value: str,
    ) -> bool:

        text = str(value).strip()

        if not text:
            return False

        if re.fullmatch(
            r"\d{1,2}",
            text,
        ):
            number = int(text)

            return 1 <= number <= 31

        if re.fullmatch(
            r"\d{4}",
            text,
        ):
            number = int(text)

            return 1900 <= number <= 2100

        if text.lower().rstrip(".") in self.MONTHS:
            return True

        return False

    def _looks_like_date_fragment(
        self,
        value: str,
    ) -> bool:

        text = str(value).strip()

        if not text:
            return False

        if re.fullmatch(
            r"\d{1,2}",
            text,
        ):
            number = int(text)
            return 1 <= number <= 31

        if text.lower().rstrip(".") in self.MONTHS:
            return True

        if re.fullmatch(
            r"\d{2,4}",
            text,
        ):
            number = int(text)

            return (
                1900 <= number <= 2100
                or 0 <= number <= 99
            )

        return False

    # =============================================================
    # NUMERIC ASSIGNMENT
    # =============================================================

    def _assign_numeric_tokens(
        self,
        numeric_tokens,
        numeric_columns,
    ):
        assignments = {}

        if not numeric_tokens or not numeric_columns:
            return assignments

        # Generic single-amount tables do not need debit/credit
        # competition. Preserve the closest amount-column behavior.
        if not any(
            column["semantic_type"]
            in {"debit", "credit"}
            for column in numeric_columns
        ):
            remaining = list(numeric_tokens)

            for column in numeric_columns:
                candidates = []

                for item in remaining:
                    token_index, token = item

                    distance = abs(
                        token["center"]
                        - self._column_center(column)
                    )

                    inside = (
                        column["x0"] - 5
                        <= token["center"]
                        <= column["x1"] + 5
                    )

                    if inside:
                        distance -= 1000

                    candidates.append(
                        (
                            distance,
                            item,
                        )
                    )

                if not candidates:
                    continue

                candidates.sort(
                    key=lambda item: item[0]
                )

                selected = candidates[0][1]

                assignments[
                    column["semantic_type"]
                ] = selected

                remaining.remove(selected)

            return assignments

        pairs = []

        for item in numeric_tokens:
            token_index, token = item

            for column_index, column in enumerate(
                numeric_columns
            ):
                inside = (
                    column["x0"] - 5
                    <= token["center"]
                    <= column["x1"] + 5
                )

                distance = abs(
                    token["center"]
                    - self._column_center(column)
                )

                if inside or distance <= 70:
                    score = distance

                    if inside:
                        score -= 1000

                    pairs.append(
                        (
                            score,
                            token_index,
                            column_index,
                            item,
                            column,
                        )
                    )

        pairs.sort(
            key=lambda item: item[0]
        )

        used_tokens = set()
        used_columns = set()

        for (
            _score,
            token_index,
            column_index,
            item,
            column,
        ) in pairs:

            if token_index in used_tokens:
                continue

            if column_index in used_columns:
                continue

            semantic = column["semantic_type"]

            assignments[semantic] = item

            used_tokens.add(token_index)
            used_columns.add(column_index)

        return assignments

    # =============================================================
    # DATE
    # =============================================================

    def _is_date(
        self,
        value,
    ):

        value = str(value).strip()

        if not value:
            return False

        return self._parse_date_text(value) is not None

    # =============================================================
    # NUMBER
    # =============================================================

    def _is_number(
        self,
        value,
    ):

        value = str(value).strip()

        if not value:
            return False

        # Handle trailing credit/debit markers used by some statements.
        # Examples: 14,000.00CR, 500DR, 1,250.50CR
        upper = value.upper()

        if upper.endswith("CR") or upper.endswith("DR"):
            value = value[:-2].strip()

        value = (
            value
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("AED", "")
            .replace("INR", "")
            .replace("USD", "")
            .replace("EUR", "")
            .strip()
        )

        if not value:
            return False

        return bool(
            re.fullmatch(
                r"-?\d+(?:\.\d+)?",
                value,
            )
        )

    # =============================================================
    # TRANSACTION ID
    # =============================================================

    def _looks_like_transaction_id(
        self,
        value,
    ):

        value = str(value).strip()

        if value.upper() == "TFR":
            return True

        return self._looks_like_id(value)

    def _looks_like_id(
        self,
        value,
    ):

        value = str(value).strip()

        return bool(
            re.match(
                r"^[A-Za-z]?\d{6,}$",
                value,
            )
        )

    # =============================================================
    # TRANSACTION VALIDATION
    # =============================================================

    def _is_transaction(
        self,
        row,
    ):

        date_value = (
            row.get("transaction_date")
            or row.get("posting_date")
            or row.get("value_date")
        )

        if not date_value:
            return False

        transaction_amount = (
            row.get("amount")
            or row.get("debit")
            or row.get("credit")
        )

        if not transaction_amount:
            return False

        description = row.get(
            "description"
        )

        transaction_id = row.get(
            "transaction_id"
        )

        if (
            not description
            and not transaction_id
        ):
            return False

        text = " ".join(
            str(value).lower()
            for value in row.values()
            if value is not None
        )

        for term in self.METADATA_TERMS:

            if term in text:
                return False

        return True

    # =============================================================
    # HELPERS
    # =============================================================

    def _distance_to_column(
        self,
        token,
        column,
    ):

        return abs(
            token["center"]
            - self._column_center(column)
        )