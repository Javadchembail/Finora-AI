from __future__ import annotations

import re
from typing import List, Dict


class TransactionExtractor:
    """
    Extract transaction-like rows from extracted statement text.

    This layer only extracts raw transaction candidates.
    It does not classify merchants, categories, income, or expenses.
    """

    # Date patterns supported initially:
    #
    # 09 AUG
    # 09/08/2026
    # 09-08-2026
    # 2026-08-09
    #
    DATE_TOKEN = (
        r"(?:"
        r"\d{1,2}\s+[A-Za-z]{3}"
        r"|"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|"
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r")"
    )

    # Flexible amount:
    #
    # 10.00
    # 1,299.50
    # 1299
    # 10.00CR
    # (10.00)
    # -10.00
    #
    AMOUNT_TOKEN = (
        r"(?:"
        r"\(?-?"
        r"(?:\d{1,3}(?:,\d{3})*|\d+)"
        r"(?:\.\d{1,2})?"
        r"\)?"
        r"(?:\s*(?:CR|DR))?"
        r")"
    )

    TRANSACTION_PATTERN = re.compile(
        rf"^\s*"
        rf"(?P<post_date>{DATE_TOKEN})"
        rf"\s+"
        rf"(?P<trxn_date>{DATE_TOKEN})"
        rf"\s+"
        rf"(?P<description>.*?)"
        rf"\s+"
        rf"(?P<amount>{AMOUNT_TOKEN})"
        rf"\s*$",
        re.IGNORECASE,
    )

    def _parse_amount(self, raw_amount: str) -> Dict:
        """
        Normalize an extracted amount while preserving
        whether the statement explicitly marked it CR/DR.
        """

        value = raw_amount.strip()

        credit = bool(
            re.search(r"\bCR\b", value, re.IGNORECASE)
        )

        debit = bool(
            re.search(r"\bDR\b", value, re.IGNORECASE)
        )

        cleaned = re.sub(
            r"\b(?:CR|DR)\b",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        negative_parentheses = (
            cleaned.startswith("(")
            and cleaned.endswith(")")
        )

        cleaned = cleaned.replace("(", "")
        cleaned = cleaned.replace(")", "")
        cleaned = cleaned.replace(",", "")

        negative = cleaned.startswith("-")

        cleaned = cleaned.lstrip("-")

        try:
            amount = float(cleaned)
        except ValueError:
            return {
                "value": None,
                "is_credit": credit,
                "is_debit": debit,
            }

        if negative or negative_parentheses:
            amount = -amount

        return {
            "value": amount,
            "is_credit": credit,
            "is_debit": debit,
        }

    def extract_from_page(
        self,
        page_number: int,
        text: str,
    ) -> List[Dict]:

        transactions = []

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            match = self.TRANSACTION_PATTERN.match(line)

            if not match:
                continue

            data = match.groupdict()

            amount_info = self._parse_amount(
                data["amount"]
            )

            transactions.append(
                {
                    "page_number": page_number,
                    "line_number": line_number,

                    "post_date_raw": data["post_date"],

                    "transaction_date_raw": data["trxn_date"],

                    "description_raw": (
                        data["description"].strip()
                    ),

                    "amount_raw": data["amount"].strip(),

                    "amount": amount_info["value"],

                    "explicit_credit": (
                        amount_info["is_credit"]
                    ),

                    "explicit_debit": (
                        amount_info["is_debit"]
                    ),
                }
            )

        return transactions

    def extract_from_pages(
        self,
        pages: List[Dict],
    ) -> List[Dict]:

        all_transactions = []

        for page in pages:

            page_transactions = self.extract_from_page(
                page_number=page["page_number"],
                text=page.get("text", ""),
            )

            all_transactions.extend(page_transactions)

        return all_transactions