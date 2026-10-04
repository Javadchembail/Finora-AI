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

    # Financial snapshot values for statements that expose a labeled
    # summary (especially credit-card statements). These are document
    # metadata, not transaction fields.
    opening_balance: Optional[float] = None
    card_limit: Optional[float] = None
    available_limit: Optional[float] = None
    minimum_payment_due: Optional[float] = None
    payment_due_date: Optional[str] = None
    total_payment_due: Optional[float] = None
    profit_other_charges: Optional[float] = None
    current_balance: Optional[float] = None


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

    # Strong currency indicators are kept separate from weak/fallback
    # indicators. Short text fragments such as "ARE" can appear in
    # merchant addresses ("ABU DHABI ARE") or ordinary footer text and
    # must not override an explicit currency name/code.
    CURRENCY_PATTERNS = {
        "AED": [
            r"\bAED\b",
            r"Ø¯\.Ø¥",
            r"Ø¯Ø±Ù‡Ù…",
        ],
        "INR": [
            r"\bINR\b",
            r"â‚¹",
            r"\bRs\.?\b",
            r"\bRupees?\b",
            r"\bIndian Rupees?\b",
            r"\bIndian Currency\b",
        ],
        "USD": [
            r"\bUSD\b",
            r"\$",
            r"\bUS Dollars?\b",
        ],
        "EUR": [
            r"\bEUR\b",
            r"â‚¬",
            r"\bEuros?\b",
        ],
        "GBP": [
            r"\bGBP\b",
            r"Â£",
            r"\bPounds?\b",
        ],
        "SAR": [
            r"\bSAR\b",
            r"\bSaudi Riyals?\b",
        ],
        "QAR": [
            r"\bQAR\b",
            r"\bQatari Riyals?\b",
        ],
        "KWD": [
            r"\bKWD\b",
            r"\bKuwaiti Dinars?\b",
        ],
        "BHD": [
            r"\bBHD\b",
            r"\bBahraini Dinars?\b",
        ],
        "OMR": [
            r"\bOMR\b",
            r"\bOmani Rials?\b",
        ],
    }

    # "ARE" is only a weak UAE location/currency hint. It is deliberately
    # excluded from the main AED score because statements can contain
    # unrelated text such as "ARE YOU A MERCHANT..." or "ABU DHABI ARE".
    WEAK_CURRENCY_PATTERNS = {
        "AED": [
            r"\bARE\b",
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

        # Strong indicators: explicit ISO codes, symbols, and currency
        # names. These are the primary source of truth.
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

        # Weak indicators are used only when no strong currency indicator
        # exists anywhere in the document. This prevents "ARE" from
        # overriding "INDIAN RUPEES" in an Indian bank statement.
        if not currency_scores:

            for currency, patterns in (
                self.WEAK_CURRENCY_PATTERNS.items()
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
        # Statement financial snapshot
        # =================================================

        def _money_value(value: str) -> Optional[float]:
            try:
                return float(
                    str(value).replace(",", "").strip()
                )
            except Exception:
                return None

        opening_balance = None
        opening_match = re.search(
            r"OPENING\s+BALANCE\s+([0-9][0-9,]*\.\d{2})",
            full_text,
            flags=re.IGNORECASE,
        )
        if opening_match:
            opening_balance = _money_value(
                opening_match.group(1)
            )

        card_limit = None
        available_limit = None
        minimum_payment_due = None
        payment_due_date = None
        total_payment_due = None
        profit_other_charges = None
        current_balance = None

        # Credit-card summary values are frequently laid out as a visual
        # table. PDF text extraction can reorder the cells, especially when
        # Arabic and English header text are both present, so do not rely on
        # one long header-to-values regex. Instead: locate the card-summary
        # section, extract the first 3 amounts + due date, then take the next
        # 3 monetary values for Total Due, Charges and Current Balance.
        summary_match = None
        card_header_index = text.find("card limit")

        if card_header_index >= 0:
            summary_window = full_text[
                card_header_index : card_header_index + 8000
            ]

            first_part = re.search(
                r"([0-9][0-9,]*\.\d{2})\s+"
                r"([0-9][0-9,]*\.\d{2})\s+"
                r"([0-9][0-9,]*\.\d{2})\s+"
                r"(\d{1,2}/\d{1,2}/\d{2,4})",
                summary_window,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if first_part:
                after_due = summary_window[first_part.end() :]
                trailing_values = re.findall(
                    r"[0-9][0-9,]*\.\d{2}",
                    after_due[:1200],
                )

                if len(trailing_values) >= 3:
                    summary_match = (
                        first_part.group(1),
                        first_part.group(2),
                        first_part.group(3),
                        first_part.group(4),
                        trailing_values[0],
                        trailing_values[1],
                        trailing_values[2],
                    )

        if summary_match:
            card_limit = _money_value(summary_match[0])
            available_limit = _money_value(summary_match[1])
            minimum_payment_due = _money_value(summary_match[2])
            payment_due_date = summary_match[3]
            total_payment_due = _money_value(summary_match[4])
            profit_other_charges = _money_value(summary_match[5])
            current_balance = _money_value(summary_match[6])

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

            opening_balance=opening_balance,
            card_limit=card_limit,
            available_limit=available_limit,
            minimum_payment_due=minimum_payment_due,
            payment_due_date=payment_due_date,
            total_payment_due=total_payment_due,
            profit_other_charges=profit_other_charges,
            current_balance=current_balance,
        )
