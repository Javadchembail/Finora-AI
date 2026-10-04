"""
Universal semantic header mapper for Finora AI.

This module maps financial-statement headers to Finora's
canonical transaction fields.

Design principles:
- Bank agnostic
- Country agnostic
- Supports common wording variations
- Uses exact matching first
- Uses normalized/fuzzy semantic matching second
- Never relies on bank names
- Does not force an uncertain header into a semantic field
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple


class SemanticHeaderMapper:
    """
    Convert financial statement column headers into canonical
    Finora semantic fields.

    Canonical fields:

        transaction_date
        posting_date
        value_date
        description
        amount
        debit
        credit
        balance
        transaction_id
    """

    # ------------------------------------------------------------------
    # Canonical semantic vocabulary
    # ------------------------------------------------------------------

    SEMANTIC_PATTERNS = {
        "transaction_date": [
            "date",
            "transaction date",
            "transaction_date",
            "txn date",
            "txn. date",
            "trxn date",
            "trxn. date",
            "trans date",
            "trans. date",
            "transaction dt",
            "txn dt",
            "trxn dt",
            "purchase date",
            "activity date",
            "activity dt",
            "trade date",
            "booking date",
            "book date",
        ],

        "posting_date": [
            "post date",
            "posting date",
            "posted date",
            "posting_date",
            "posted dt",
            "posting dt",
            "post dt",
            "booked date",
            "booking date",
            "accounting date",
            "accounting dt",
        ],

        "value_date": [
            "value date",
            "value_date",
            "value dt",
            "effective date",
            "effective dt",
            "effective value date",
            "effective value dt",
            "settlement date",
            "settlement dt",
        ],

        "description": [
            "description",
            "particulars",
            "particular",
            "details",
            "detail",
            "transaction details",
            "transaction description",
            "txn details",
            "txn description",
            "trxn details",
            "trxn description",
            "merchant",
            "merchant name",
            "merchant details",
            "narration",
            "narrative",
            "remarks",
            "remark",
            "memo",
            "reference description",
            "transaction particulars",
            "payment details",
            "payment description",
        ],

        "amount": [
            "amount",
            "transaction amount",
            "txn amount",
            "trxn amount",
            "transaction value",
            "txn value",
            "trxn value",
            "total amount",
            "payment amount",
            "purchase amount",
            "value",
        ],

        "debit": [
            "debit",
            "debits",
            "debit amount",
            "debit amt",
            "debit value",
            "dr",
            "dr amount",
            "dr amt",
            "withdrawal",
            "withdrawals",
            "withdrawal amount",
            "withdrawal amt",
            "money out",
            "money paid",
            "outgoing",
            "outgoing amount",
            "spent",
            "spend",
            "payments",
            "payment",
        ],

        "credit": [
            "credit",
            "credits",
            "credit amount",
            "credit amt",
            "credit value",
            "cr",
            "cr amount",
            "cr amt",
            "deposit",
            "deposits",
            "deposit amount",
            "deposit amt",
            "money in",
            "money received",
            "incoming",
            "incoming amount",
            "received",
            "receipts",
        ],

        "balance": [
            "balance",
            "closing balance",
            "closing bal",
            "available balance",
            "available bal",
            "current balance",
            "current bal",
            "running balance",
            "running bal",
            "account balance",
            "account bal",
            "ledger balance",
            "ledger bal",
            "ending balance",
            "ending bal",
            "balance after transaction",
            "balance after txn",
        ],

        "transaction_id": [
            "transaction id",
            "transaction_id",
            "txn id",
            "txn. id",
            "trxn id",
            "trxn. id",
            "tran id",
            "tran. id",
            "transaction number",
            "transaction no",
            "transaction no.",
            "txn number",
            "txn no",
            "txn no.",
            "trxn number",
            "trxn no",
            "reference",
            "reference number",
            "reference no",
            "reference no.",
            "reference id",
            "reference identifier",
        ],
    }

    # ------------------------------------------------------------------
    # Semantic keyword signals
    #
    # These are not bank-specific. They provide additional meaning when
    # an exact header is not found.
    # ------------------------------------------------------------------

    SEMANTIC_SIGNALS = {
        "transaction_date": {
            "date",
            "transaction",
            "txn",
            "trxn",
            "trans",
            "purchase",
            "activity",
            "trade",
            "booking",
            "book",
        },

        "posting_date": {
            "post",
            "posting",
            "posted",
            "accounting",
            "booked",
            "booking",
        },

        "value_date": {
            "value",
            "effective",
            "settlement",
        },

        "description": {
            "description",
            "details",
            "detail",
            "particular",
            "particulars",
            "narration",
            "narrative",
            "merchant",
            "remarks",
            "remark",
            "memo",
            "payment",
            "transaction",
        },

        "amount": {
            "amount",
            "value",
            "total",
            "payment",
            "purchase",
            "transaction",
            "txn",
            "trxn",
        },

        "debit": {
            "debit",
            "debits",
            "dr",
            "withdrawal",
            "withdrawals",
            "outgoing",
            "spent",
            "spend",
            "paid",
            "payment",
            "money",
            "out",
        },

        "credit": {
            "credit",
            "credits",
            "cr",
            "deposit",
            "deposits",
            "incoming",
            "received",
            "receipts",
            "money",
            "in",
        },

        "balance": {
            "balance",
            "bal",
            "closing",
            "closing",
            "running",
            "current",
            "available",
            "ending",
            "ledger",
        },

        "transaction_id": {
            "transaction",
            "txn",
            "trxn",
            "tran",
            "reference",
            "ref",
            "number",
            "no",
            "id",
        },
    }

    # ------------------------------------------------------------------
    # Ambiguous terms
    #
    # These must not be blindly classified.
    # ------------------------------------------------------------------

    AMBIGUOUS_HEADERS = {
        "value",
        "payment",
        "transaction",
        "reference",
        "number",
        "id",
        "date",
        "amount",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_header(
        self,
        header: str,
    ) -> Optional[str]:
        """
        Map one header to a canonical semantic field.

        Returns only the semantic type.

        For uncertain headers, returns None.
        """

        semantic_type, confidence = self.map_header_with_confidence(
            header
        )

        if semantic_type is None:
            return None

        # Conservative threshold for automatic semantic mapping.
        if confidence < 0.70:
            return None

        return semantic_type

    def map_header_with_confidence(
        self,
        header: str,
    ) -> Tuple[Optional[str], float]:
        """
        Return:

            (semantic_type, confidence)

        Confidence is based on:
        1. exact normalized match
        2. phrase similarity
        3. semantic keyword overlap
        4. ambiguity penalties
        """

        normalized = self._normalize(header)

        if not normalized:
            return None, 0.0

        # --------------------------------------------------------------
        # Exact match
        # --------------------------------------------------------------

        exact = self._exact_match(normalized)

        if exact is not None:
            return exact, 0.99

        # --------------------------------------------------------------
        # Fuzzy semantic match
        # --------------------------------------------------------------

        fuzzy = self._fuzzy_match(normalized)

        if fuzzy is None:
            return None, 0.0

        semantic_type, confidence = fuzzy

        # Ambiguous standalone terms should not automatically become
        # semantic fields.
        if normalized in self.AMBIGUOUS_HEADERS:
            confidence *= 0.65

        if confidence < 0.70:
            return None, confidence

        return semantic_type, confidence

    def map_headers(
        self,
        headers,
    ) -> Dict[str, str]:
        """
        Map a list of headers.

        Example:

            {
                "Date": "transaction_date",
                "Narrative": "description",
                "Money Out": "debit",
                "Money In": "credit",
                "Closing Bal": "balance",
            }
        """

        result = {}

        for header in headers:

            semantic_type = self.map_header(header)

            if semantic_type:
                result[header] = semantic_type

        return result

    def map_headers_with_confidence(
        self,
        headers,
    ) -> Dict[str, Dict[str, object]]:
        """
        Same as map_headers(), but preserves confidence information.
        """

        result = {}

        for header in headers:

            semantic_type, confidence = (
                self.map_header_with_confidence(
                    header
                )
            )

            result[header] = {
                "semantic_type": semantic_type,
                "confidence": round(
                    confidence,
                    4,
                ),
            }

        return result

    # ------------------------------------------------------------------
    # Exact matching
    # ------------------------------------------------------------------

    def _exact_match(
        self,
        normalized: str,
    ) -> Optional[str]:

        for semantic_type, phrases in (
            self.SEMANTIC_PATTERNS.items()
        ):

            for phrase in phrases:

                if normalized == self._normalize(
                    phrase
                ):
                    return semantic_type

        return None

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------

    def _fuzzy_match(
        self,
        normalized: str,
    ) -> Optional[Tuple[str, float]]:

        candidates = []

        for semantic_type, phrases in (
            self.SEMANTIC_PATTERNS.items()
        ):

            best_similarity = 0.0

            for phrase in phrases:

                phrase_normalized = self._normalize(
                    phrase
                )

                similarity = SequenceMatcher(
                    None,
                    normalized,
                    phrase_normalized,
                ).ratio()

                best_similarity = max(
                    best_similarity,
                    similarity,
                )

            keyword_score = self._keyword_score(
                normalized,
                semantic_type,
            )

            # Combine phrase similarity and semantic keywords.
            confidence = (
                best_similarity * 0.65
                + keyword_score * 0.35
            )

            candidates.append(
                (
                    confidence,
                    semantic_type,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            reverse=True
        )

        best_confidence, best_semantic = (
            candidates[0]
        )

        # Require meaningful separation from the second-best
        # semantic candidate. This prevents arbitrary classification
        # of ambiguous headers.
        if len(candidates) > 1:

            second_confidence = (
                candidates[1][0]
            )

            if (
                best_confidence < 0.82
                and (
                    best_confidence
                    - second_confidence
                    < 0.08
                )
            ):
                return None

        return (
            best_semantic,
            min(
                best_confidence,
                0.95,
            ),
        )

    # ------------------------------------------------------------------
    # Keyword scoring
    # ------------------------------------------------------------------

    def _keyword_score(
        self,
        normalized: str,
        semantic_type: str,
    ) -> float:

        tokens = set(
            normalized.split()
        )

        signals = self.SEMANTIC_SIGNALS.get(
            semantic_type,
            set(),
        )

        if not signals:
            return 0.0

        matches = tokens.intersection(
            signals
        )

        if not matches:
            return 0.0

        return min(
            len(matches) / max(
                len(tokens),
                1,
            ),
            1.0,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:

        value = str(
            value
        ).strip().lower()

        value = value.replace(
            "\u00a0",
            " ",
        )

        # Normalize common separators.
        value = value.replace(
            "_",
            " ",
        )

        value = value.replace(
            ":",
            " ",
        )

        value = value.replace(
            "/",
            " ",
        )

        value = re.sub(
            r"[()\[\]{}]",
            " ",
            value,
        )

        # Keep periods because:
        #
        # txn. date
        # trxn. id
        #
        # can be meaningful.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()