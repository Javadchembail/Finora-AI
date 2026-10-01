"""
Universal semantic header mapper.

Maps financial-statement header phrases to canonical
transaction fields.

This module does NOT assume any particular bank.
"""

from __future__ import annotations

import re
from typing import Optional


class SemanticHeaderMapper:

    SEMANTIC_PATTERNS = {

        "transaction_date": [
            r"^date$",
            r"^transaction date$",
            r"^transaction_date$",
            r"^txn date$",
            r"^txn\. date$",
            r"^trxn date$",
            r"^trxn\. date$",
            r"^trans date$",
            r"^trans\. date$",
            r"^purchase date$",
            r"^activity date$",
        ],

        "posting_date": [
            r"^post date$",
            r"^posting date$",
            r"^posted date$",
            r"^posting_date$",
            r"^value posted date$",
        ],

        "value_date": [
            r"^value date$",
            r"^value_date$",
            r"^effective date$",
            r"^effective value date$",
        ],

        "description": [
            r"^description$",
            r"^particulars$",
            r"^details$",
            r"^transaction details$",
            r"^transaction description$",
            r"^merchant$",
            r"^merchant name$",
            r"^narration$",
            r"^remarks$",
            r"^memo$",
            r"^reference description$",
        ],

        "amount": [
            r"^amount$",
            r"^transaction amount$",
            r"^txn amount$",
            r"^trxn amount$",
            r"^transaction value$",
            r"^value$",
        ],

        "debit": [
            r"^debit$",
            r"^debits$",
            r"^withdrawal$",
            r"^withdrawals$",
            r"^withdrawal amount$",
            r"^debit amount$",
            r"^dr$",
        ],

        "credit": [
            r"^credit$",
            r"^credits$",
            r"^deposit$",
            r"^deposits$",
            r"^deposit amount$",
            r"^credit amount$",
            r"^cr$",
        ],

        "balance": [
            r"^balance$",
            r"^closing balance$",
            r"^available balance$",
            r"^current balance$",
            r"^running balance$",
            r"^account balance$",
            r"^ledger balance$",
        ],

        "transaction_id": [
            r"^transaction id$",
            r"^transaction_id$",
            r"^txn id$",
            r"^txn\. id$",
            r"^trxn id$",
            r"^trxn\. id$",
            r"^tran id$",
            r"^tran\. id$",
            r"^reference$",
            r"^reference number$",
            r"^reference no$",
            r"^reference no\.$",
            r"^transaction number$",
            r"^transaction no$",
            r"^transaction no\.$",
        ],
    }

    def map_header(
        self,
        header: str,
    ) -> Optional[str]:

        normalized = self._normalize(header)

        for semantic_type, patterns in self.SEMANTIC_PATTERNS.items():

            for pattern in patterns:

                if re.match(
                    pattern,
                    normalized,
                    re.IGNORECASE,
                ):
                    return semantic_type

        return None

    def map_headers(self, headers):
        """
        Map a list of headers.

        Returns:

        {
            "Date": "transaction_date",
            "Description": "description",
            ...
        }
        """

        result = {}

        for header in headers:

            semantic_type = self.map_header(header)

            if semantic_type:
                result[header] = semantic_type

        return result

    @staticmethod
    def _normalize(value: str) -> str:

        value = str(value).strip().lower()

        value = value.replace(
            "\u00a0",
            " ",
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value