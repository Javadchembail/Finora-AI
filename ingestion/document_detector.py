from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):

    document_type: str = "unknown"

    statement_type: str = "unknown"

    country: Optional[str] = None

    currency: Optional[str] = None

    currency_symbol: Optional[str] = None

    date_format: Optional[str] = None

    statement_start_date: Optional[str] = None

    statement_end_date: Optional[str] = None

    statement_year: Optional[int] = None

    language: Optional[str] = None

    has_transaction_table: bool = False

    has_posting_date: bool = False

    has_transaction_date: bool = False

    has_debit_credit_indicator: bool = False

    has_withdrawal_column: bool = False

    has_deposit_column: bool = False

    has_balance_column: bool = False

    has_transaction_id: bool = False

    is_password_protected: bool = False

    page_count: int = 0

    confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    detected_keywords: List[str] = []


class DocumentDetector:

    STATEMENT_KEYWORDS = [
        "statement",
        "account statement",
        "statement of account",
        "statement of card account",
        "transaction",
        "transactions",
    ]

    CREDIT_CARD_KEYWORDS = [
        "credit card",
        "card account",
        "card limit",
        "available limit",
        "minimum payment due",
        "total payment due",
        "payment due date",
    ]

    BANK_ACCOUNT_KEYWORDS = [
        "savings account",
        "current account",
        "type of account",
        "account status",
        "opening balance",
        "closing balance",
        "withdrawals",
        "deposits",
        "effective available balance",
        "value date",
        "tran id",
        "tran type",
    ]

    CURRENCY_PATTERNS = {
        "AED": [
            r"\bAED\b",
            r"\bARE\b",
            r"د\.إ",
            r"درهم",
        ],
        "INR": [
            r"\bINR\b",
            r"₹",
            r"\bRs\.?\b",
            r"\bRupees?\b",
        ],
        "USD": [
            r"\bUSD\b",
            r"\$",
        ],
        "EUR": [
            r"\bEUR\b",
            r"€",
        ],
        "GBP": [
            r"\bGBP\b",
            r"£",
        ],
        "SAR": [
            r"\bSAR\b",
        ],
        "QAR": [
            r"\bQAR\b",
        ],
        "KWD": [
            r"\bKWD\b",
        ],
        "BHD": [
            r"\bBHD\b",
        ],
        "OMR": [
            r"\bOMR\b",
        ],
    }

    COUNTRY_CURRENCY_MAP = {
        "AED": "United Arab Emirates",
        "INR": "India",
        "USD": "United States",
        "EUR": "European Union",
        "GBP": "United Kingdom",
        "SAR": "Saudi Arabia",
        "QAR": "Qatar",
        "KWD": "Kuwait",
        "BHD": "Bahrain",
        "OMR": "Oman",
    }

    def detect(
        self,
        pages: List[dict],
    ) -> DocumentMetadata:

        full_text = "\n".join(
            page.get("text", "")
            for page in pages
        )

        text = full_text.lower()

        detected_keywords = []

        # =================================================
        # Document type
        # =================================================

        statement_score = 0

        for keyword in self.STATEMENT_KEYWORDS:

            if keyword in text:

                statement_score += 1

                detected_keywords.append(
                    keyword
                )

        document_type = (
            "financial_statement"
            if statement_score > 0
            else "unknown"
        )

        # =================================================
        # Statement type
        # =================================================

        credit_score = 0
        bank_score = 0

        for keyword in self.CREDIT_CARD_KEYWORDS:

            if keyword in text:

                credit_score += 1

                detected_keywords.append(
                    keyword
                )

        for keyword in self.BANK_ACCOUNT_KEYWORDS:

            if keyword in text:

                bank_score += 1

                detected_keywords.append(
                    keyword
                )

        if (
            credit_score > bank_score
            and credit_score > 0
        ):

            statement_type = "credit_card"

        elif bank_score > 0:

            statement_type = "bank_account"

        else:

            statement_type = "unknown"

        # =================================================
        # Currency
        # =================================================

        currency_scores = {}

        for currency, patterns in (
            self.CURRENCY_PATTERNS.items()
        ):

            score = 0

            for pattern in patterns:

                score += len(
                    re.findall(
                        pattern,
                        full_text,
                        flags=re.IGNORECASE,
                    )
                )

            if score:

                currency_scores[
                    currency
                ] = score

        currency = None

        if currency_scores:

            currency = max(
                currency_scores,
                key=currency_scores.get,
            )

        country = (
            self.COUNTRY_CURRENCY_MAP.get(
                currency
            )
            if currency
            else None
        )

        # =================================================
        # Date columns
        # =================================================

        has_posting_date = (
            "post date" in text
            or "posting date" in text
            or "value date" in text
        )

        has_transaction_date = (
            "trxn. date" in text
            or "transaction date" in text
            or "txn date" in text
            or (
                "date" in text
                and "particulars" in text
            )
        )

        date_format = None

        if re.search(
            r"\b\d{1,2}\s+[A-Za-z]{3}\b",
            full_text,
        ):

            date_format = "DD MMM"

        elif re.search(
            r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
            full_text,
        ):

            date_format = "DD-MMM-YYYY"

        elif re.search(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            full_text,
        ):

            date_format = "numeric"

        elif re.search(
            r"\b\d{4}-\d{2}-\d{2}\b",
            full_text,
        ):

            date_format = "YYYY-MM-DD"

        # =================================================
        # Statement period
        # =================================================

        start_date = None
        end_date = None
        statement_year = None

        # Format:
        # From: 1st Aug 2026
        # To: 31st Aug 2026

        period_match = re.search(
            r"From:\s*"
            r"(\d{1,2})"
            r"(?:st|nd|rd|th)?"
            r"\s+"
            r"([A-Za-z]{3,9})"
            r"\s+"
            r"(\d{4})"
            r".*?"
            r"To:\s*"
            r"(\d{1,2})"
            r"(?:st|nd|rd|th)?"
            r"\s+"
            r"([A-Za-z]{3,9})"
            r"\s+"
            r"(\d{4})",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if period_match:

            start_date = (
                f"{period_match.group(1)} "
                f"{period_match.group(2)} "
                f"{period_match.group(3)}"
            )

            end_date = (
                f"{period_match.group(4)} "
                f"{period_match.group(5)} "
                f"{period_match.group(6)}"
            )

            statement_year = int(
                period_match.group(3)
            )

        # Format:
        # period 2026-08-31 to 2026-09-30

        if not period_match:

            iso_period = re.search(
                r"period\s+"
                r"(\d{4}-\d{2}-\d{2})"
                r"\s+to\s+"
                r"(\d{4}-\d{2}-\d{2})",
                full_text,
                flags=re.IGNORECASE,
            )

            if iso_period:

                start_date = (
                    iso_period.group(1)
                )

                end_date = (
                    iso_period.group(2)
                )

                statement_year = int(
                    iso_period.group(1)[:4]
                )

        # =================================================
        # Columns
        # =================================================

        has_withdrawal_column = (
            "withdrawals" in text
        )

        has_deposit_column = (
            "deposits" in text
        )

        has_balance_column = (
            "balance" in text
        )

        has_transaction_id = (
            "tran id" in text
            or "transaction id" in text
        )

        has_debit_credit_indicator = (
            "dr /cr" in text
            or "dr/cr" in text
            or "debit" in text
            or "credit" in text
        )

        has_transaction_table = (
            (
                has_transaction_date
                and "particulars" in text
            )
            or
            (
                has_posting_date
                and has_transaction_date
                and "description" in text
            )
        )

        # =================================================
        # Confidence
        # =================================================

        checks = [
            document_type != "unknown",
            statement_type != "unknown",
            currency is not None,
            has_transaction_table,
        ]

        confidence = (
            sum(checks) / len(checks)
            if checks
            else 0.0
        )

        return DocumentMetadata(

            document_type=document_type,

            statement_type=statement_type,

            country=country,

            currency=currency,

            date_format=date_format,

            statement_start_date=start_date,

            statement_end_date=end_date,

            statement_year=statement_year,

            has_transaction_table=(
                has_transaction_table
            ),

            has_posting_date=(
                has_posting_date
            ),

            has_transaction_date=(
                has_transaction_date
            ),

            has_debit_credit_indicator=(
                has_debit_credit_indicator
            ),

            has_withdrawal_column=(
                has_withdrawal_column
            ),

            has_deposit_column=(
                has_deposit_column
            ),

            has_balance_column=(
                has_balance_column
            ),

            has_transaction_id=(
                has_transaction_id
            ),

            page_count=len(pages),

            confidence=round(
                confidence,
                2,
            ),

            detected_keywords=(
                detected_keywords
            ),
        )