from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnDefinition(BaseModel):
    name: str
    semantic_type: str
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class DocumentStructure(BaseModel):
    transaction_header_page: Optional[int] = None
    header_text: Optional[str] = None

    columns: List[ColumnDefinition] = []

    date_columns: List[str] = []
    description_columns: List[str] = []
    amount_columns: List[str] = []
    debit_columns: List[str] = []
    credit_columns: List[str] = []
    balance_columns: List[str] = []
    transaction_id_columns: List[str] = []

    posting_date_columns: List[str] = []
    transaction_date_columns: List[str] = []

    has_debit_credit_columns: bool = False
    has_separate_debit_credit_columns: bool = False
    has_single_amount_column: bool = False
    has_balance_column: bool = False
    has_transaction_id: bool = False

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


class StructureDetector:
    """
    Universal financial statement structure detector.

    This class does not identify or depend on a particular bank.

    It attempts to reconstruct semantic transaction columns from
    imperfect PDF text extraction.
    """

    DATE_PATTERNS = [
        ("Post Date", "posting_date"),
        ("Posting Date", "posting_date"),
        ("Transaction Date", "transaction_date"),
        ("Trxn. Date", "transaction_date"),
        ("Trxn Date", "transaction_date"),
        ("Txn Date", "transaction_date"),
        ("Trans Date", "transaction_date"),
        ("Value Date", "value_date"),
        ("Effective Date", "value_date"),
        ("Booking Date", "transaction_date"),
        ("Entry Date", "transaction_date"),
        ("Process Date", "transaction_date"),
        ("Date", "transaction_date"),
    ]

    DESCRIPTION_PATTERNS = [
        ("Transaction Description", "description"),
        ("Transaction Details", "description"),
        ("Particulars", "description"),
        ("Description", "description"),
        ("Narration", "description"),
        ("Details", "description"),
        ("Remarks", "description"),
        ("Merchant", "description"),
        ("Payee", "description"),
        ("Beneficiary", "description"),
        ("Memo", "description"),
    ]

    AMOUNT_PATTERNS = [
        ("Transaction Amount", "amount"),
        ("Amount", "amount"),
        ("Value", "amount"),
    ]

    DEBIT_PATTERNS = [
        ("Withdrawals", "debit"),
        ("Withdrawal", "debit"),
        ("Debit", "debit"),
        ("Debits", "debit"),
        ("Amount Out", "debit"),
        ("Money Out", "debit"),
        ("Paid Out", "debit"),
        ("Payment Out", "debit"),
        ("DR", "debit"),
        ("DR.", "debit"),
    ]

    CREDIT_PATTERNS = [
        ("Deposits", "credit"),
        ("Deposit", "credit"),
        ("Credit", "credit"),
        ("Credits", "credit"),
        ("Amount In", "credit"),
        ("Money In", "credit"),
        ("Paid In", "credit"),
        ("Payment In", "credit"),
        ("CR", "credit"),
        ("CR.", "credit"),
    ]

    BALANCE_PATTERNS = [
        ("Effective Available Balance", "balance"),
        ("Available Balance", "balance"),
        ("Closing Balance", "balance"),
        ("Running Balance", "balance"),
        ("Current Balance", "balance"),
        ("Account Balance", "balance"),
        ("Ledger Balance", "balance"),
        ("Book Balance", "balance"),
        ("Balance", "balance"),
    ]

    TRANSACTION_ID_PATTERNS = [
        ("Transaction ID", "transaction_id"),
        ("Transaction Number", "transaction_id"),
        ("Transaction No", "transaction_id"),
        ("Transaction #", "transaction_id"),
        ("Txn ID", "transaction_id"),
        ("Txn No", "transaction_id"),
        ("Txn Number", "transaction_id"),
        ("Tran ID", "transaction_id"),
        ("Tran No", "transaction_id"),
        ("Reference Number", "transaction_id"),
        ("Reference No", "transaction_id"),
        ("Ref No", "transaction_id"),
        ("Confirmation Number", "transaction_id"),
        ("Trace Number", "transaction_id"),
        ("Trace ID", "transaction_id"),
    ]

    def detect(
        self,
        pages: List[dict],
    ) -> DocumentStructure:

        candidates = self._find_header_candidates(
            pages
        )

        if not candidates:
            return DocumentStructure(
                confidence=0.0
            )

        best = max(
            candidates,
            key=lambda item: item["score"],
        )

        header_lines = best["lines"]

        header_text = " | ".join(
            header_lines
        )

        columns = self._detect_columns(
            header_lines
        )

        return self._build_structure(
            header_page=best["page_number"],
            header_text=header_text,
            columns=columns,
        )

    # =============================================================
    # Header discovery
    # =============================================================

    def _find_header_candidates(
        self,
        pages: List[dict],
    ) -> List[Dict]:

        candidates = []

        for page in pages:

            page_number = page.get(
                "page_number"
            )

            text = page.get(
                "text",
                "",
            )

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for index in range(
                len(lines)
            ):

                # Look at up to 4 adjacent lines.
                window = lines[
                    index:index + 4
                ]

                combined = " ".join(
                    window
                )

                score = self._header_score(
                    combined
                )

                if score < 4:
                    continue

                candidates.append(
                    {
                        "page_number": page_number,
                        "lines": window,
                        "score": score,
                    }
                )

        return candidates

    # =============================================================
    # Header scoring
    # =============================================================

    def _header_score(
        self,
        text: str,
    ) -> int:

        normalized = self._normalize(
            text
        )

        matches = []

        for pattern, semantic in (
            self.DATE_PATTERNS
            + self.DESCRIPTION_PATTERNS
            + self.AMOUNT_PATTERNS
            + self.DEBIT_PATTERNS
            + self.CREDIT_PATTERNS
            + self.BALANCE_PATTERNS
            + self.TRANSACTION_ID_PATTERNS
        ):

            if self._contains_phrase(
                normalized,
                self._normalize(pattern),
            ):
                matches.append(
                    semantic
                )

        semantic_types = set(
            matches
        )

        score = len(
            semantic_types
        )

        # A transaction header should normally
        # have at least date + description/amount.
        if "transaction_date" in semantic_types:
            if (
                "description"
                in semantic_types
                or "amount"
                in semantic_types
                or "debit"
                in semantic_types
                or "credit"
                in semantic_types
            ):
                score += 3

        if "posting_date" in semantic_types:
            score += 1

        if "value_date" in semantic_types:
            score += 1

        return score

    # =============================================================
    # Column detection
    # =============================================================

    def _detect_columns(
        self,
        header_lines: List[str],
    ) -> List[ColumnDefinition]:

        columns = []

        combined = " ".join(
            header_lines
        )

        # ---------------------------------------------------------
        # First pass:
        # recognize multi-word / known semantic phrases
        # from the complete header.
        # ---------------------------------------------------------

        all_patterns = (
            self.DATE_PATTERNS
            + self.DESCRIPTION_PATTERNS
            + self.AMOUNT_PATTERNS
            + self.DEBIT_PATTERNS
            + self.CREDIT_PATTERNS
            + self.BALANCE_PATTERNS
            + self.TRANSACTION_ID_PATTERNS
        )

        # Longest patterns first prevents:
        #
        # "Transaction Date"
        # from becoming:
        # "Transaction" + "Date"
        #
        all_patterns = sorted(
            all_patterns,
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        )

        working_text = self._normalize(
            combined
        )

        for display_name, semantic_type in all_patterns:

            normalized_pattern = (
                self._normalize(
                    display_name
                )
            )

            if not self._contains_phrase(
                working_text,
                normalized_pattern,
            ):
                continue

            # Special handling:
            #
            # A standalone "Date" must not be added
            # when it is already part of "Value Date",
            # "Post Date", etc.
            if display_name == "Date":

                if any(
                    self._contains_phrase(
                        working_text,
                        self._normalize(
                            phrase
                        ),
                    )
                    for phrase, _ in self.DATE_PATTERNS
                    if phrase != "Date"
                    and phrase.lower().endswith(
                        "date"
                    )
                ):
                    # There can still be a separate
                    # "Date" column, so don't globally
                    # skip it here. We handle it later.
                    pass

            columns.append(
                ColumnDefinition(
                    name=display_name,
                    semantic_type=semantic_type,
                    confidence=1.0,
                )
            )

        # ---------------------------------------------------------
        # Remove duplicate semantic matches.
        # ---------------------------------------------------------

        columns = self._resolve_date_columns(
            columns,
            working_text,
        )

        columns = self._deduplicate_columns(
            columns
        )

        return columns

    # =============================================================
    # Resolve date columns
    # =============================================================

    def _resolve_date_columns(
        self,
        columns: List[ColumnDefinition],
        text: str,
    ) -> List[ColumnDefinition]:

        # Keep all explicitly detected date columns.
        #
        # If the statement has:
        #
        # Date Value Date
        #
        # we need BOTH:
        #
        # Date
        # Value Date
        #
        # If it only has:
        #
        # Value Date
        #
        # we should not invent Date.

        has_value_date = any(
            column.name.lower()
            == "value date"
            for column in columns
        )

        has_post_date = any(
            column.name.lower()
            == "post date"
            for column in columns
        )

        has_transaction_date = any(
            column.name.lower()
            in {
                "transaction date",
                "trxn. date",
                "trxn date",
                "txn date",
                "trans date",
            }
            for column in columns
        )

        # If the original header explicitly contains
        # "Date Value Date", add standalone Date.
        if (
            "date value date"
            in text
            and not any(
                column.name == "Date"
                for column in columns
            )
        ):

            columns.append(
                ColumnDefinition(
                    name="Date",
                    semantic_type="transaction_date",
                    confidence=1.0,
                )
            )

        # Emirates:
        # "Post Date Trxn. Date"
        #
        # The phrase contains two date columns.
        if (
            "post date trxn date"
            in text
        ):

            if not has_post_date:

                columns.append(
                    ColumnDefinition(
                        name="Post Date",
                        semantic_type="posting_date",
                        confidence=1.0,
                    )
                )

            if not has_transaction_date:

                columns.append(
                    ColumnDefinition(
                        name="Trxn. Date",
                        semantic_type="transaction_date",
                        confidence=1.0,
                    )
                )

        return columns

    # =============================================================
    # Build structure
    # =============================================================

    def _build_structure(
        self,
        header_page: int,
        header_text: str,
        columns: List[ColumnDefinition],
    ) -> DocumentStructure:

        date_columns = [
            column.name
            for column in columns
            if column.semantic_type
            in {
                "transaction_date",
                "posting_date",
                "value_date",
            }
        ]

        description_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "description"
        ]

        amount_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "amount"
        ]

        debit_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "debit"
        ]

        credit_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "credit"
        ]

        balance_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "balance"
        ]

        transaction_id_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "transaction_id"
        ]

        posting_date_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "posting_date"
        ]

        transaction_date_columns = [
            column.name
            for column in columns
            if column.semantic_type
            == "transaction_date"
        ]

        separate_debit_credit = (
            bool(debit_columns)
            and bool(credit_columns)
        )

        single_amount = (
            bool(amount_columns)
            and not separate_debit_credit
        )

        checks = [
            bool(date_columns),
            bool(description_columns),
            bool(
                amount_columns
                or debit_columns
                or credit_columns
            ),
        ]

        confidence = (
            sum(checks) / len(checks)
            if checks
            else 0.0
        )

        return DocumentStructure(
            transaction_header_page=header_page,
            header_text=header_text,
            columns=columns,
            date_columns=date_columns,
            description_columns=description_columns,
            amount_columns=amount_columns,
            debit_columns=debit_columns,
            credit_columns=credit_columns,
            balance_columns=balance_columns,
            transaction_id_columns=transaction_id_columns,
            posting_date_columns=posting_date_columns,
            transaction_date_columns=transaction_date_columns,
            has_debit_credit_columns=(
                bool(
                    debit_columns
                    or credit_columns
                )
            ),
            has_separate_debit_credit_columns=(
                separate_debit_credit
            ),
            has_single_amount_column=(
                single_amount
            ),
            has_balance_column=(
                bool(balance_columns)
            ),
            has_transaction_id=(
                bool(transaction_id_columns)
            ),
            confidence=round(
                confidence,
                2,
            ),
        )

    # =============================================================
    # Helpers
    # =============================================================

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        text = text.lower()

        text = text.replace(
            "trxn.",
            "trxn",
        )

        text = re.sub(
            r"[^a-z0-9#./\s-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _contains_phrase(
        text: str,
        phrase: str,
    ) -> bool:

        if not phrase:
            return False

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(phrase)
            + r"(?![a-z0-9])"
        )

        return bool(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _deduplicate_columns(
        columns: List[ColumnDefinition],
    ) -> List[ColumnDefinition]:

        seen = set()

        result = []

        for column in columns:

            key = (
                column.name.lower(),
                column.semantic_type,
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                column
            )

        return result