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

    confidence: float = Field(default=0.0, ge=0, le=1)
    detected_keywords: List[str] = Field(default_factory=list)

    # Statement-level financial snapshot
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

    # Weighted indicators: explicit product wording has more weight than
    # general financial terms that may appear on several document types.
    CREDIT_CARD_SIGNALS = {
        "credit card statement": 10,
        "credit card": 5,
        "card account": 3,
        "card limit": 5,
        "credit limit": 4,
        "available credit": 5,
        "available limit": 5,
        "minimum payment due": 5,
        "minimum amount due": 4,
        "total payment due": 5,
        "total amount due": 2,
        "payment due date": 4,
        "cashback card": 3,
        "cash back card": 3,
        "cardholder": 2,
        "statement balance": 2,
        "retail purchases": 2,
        "finance charges": 2,
        "profit charges": 2,
        "previous balance": 1,
        "current balance": 1,
    }

    BANK_ACCOUNT_SIGNALS = {
        "bank account statement": 10,
        "savings account": 5,
        "current account": 5,
        "type of account": 4,
        "account type": 3,
        "account status": 3,
        "withdrawals": 2,
        "deposits": 2,
        "opening balance": 2,
        "closing balance": 2,
        "effective available balance": 4,
        "available account balance": 4,
        "value date": 2,
        "tran id": 2,
        "transaction id": 2,
        "account activity": 2,
        "debit account": 2,
        "credit account": 2,
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

    CURRENCY_PATTERNS = {
        "AED": [
            r"\bAED\b",
            r"\bDHS?\.?\b",
            r"\bDIRHAMS?\b",
            r"د\.إ",
        ],
        "INR": [
            r"\bINR\b",
            r"\bRUPEES?\b",
            r"₹",
        ],
        "USD": [
            r"\bUSD\b",
            r"\bUS DOLLARS?\b",
            r"\$",
        ],
        "EUR": [
            r"\bEUR\b",
            r"\bEUROS?\b",
            r"€",
        ],
        "GBP": [
            r"\bGBP\b",
            r"\bPOUNDS?\b",
            r"£",
        ],
        "SAR": [
            r"\bSAR\b",
            r"\bSAUDI RIYALS?\b",
        ],
        "QAR": [
            r"\bQAR\b",
            r"\bQATARI RIYALS?\b",
        ],
        "KWD": [
            r"\bKWD\b",
            r"\bKUWAITI DINARS?\b",
        ],
        "BHD": [
            r"\bBHD\b",
            r"\bBAHRAINI DINARS?\b",
        ],
        "OMR": [
            r"\bOMR\b",
            r"\bOMANI RIALS?\b",
        ],
    }

    CURRENCY_SYMBOLS = {
        "AED": "د.إ",
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "SAR": "﷼",
        "QAR": "﷼",
        "KWD": "د.ك",
        "BHD": ".د.ب",
        "OMR": "ر.ع.",
    }

    DATE_PATTERNS = [
        re.compile(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        ),
        re.compile(
            r"\b(\d{4}-\d{2}-\d{2})\b"
        ),
        re.compile(
            r"\b(\d{1,2}\s+"
            r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
            r"[A-Z]*\s+\d{4})\b",
            re.IGNORECASE,
        ),
    ]

    def detect(self, pages: List[dict]) -> DocumentMetadata:
        """
        Detect document metadata.

        First inspect the first six pages. If the essential metadata is
        incomplete, scan the full document.
        """
        if not pages:
            return DocumentMetadata(page_count=0)

        result = self._detect_text(pages[:6])
        result.page_count = len(pages)

        if (
            result.confidence >= 0.75
            and result.document_type != "unknown"
            and result.statement_type != "unknown"
            and result.currency is not None
            and result.has_transaction_table
        ):
            return result

        result = self._detect_text(pages)
        result.page_count = len(pages)
        return result

    @staticmethod
    def _money_value(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None

        try:
            normalized = (
                str(value)
                .replace(",", "")
                .replace(" ", "")
                .strip()
            )
            result = float(normalized)

            # Avoid accepting NaN or infinity as valid monetary values.
            if result != result or result in (float("inf"), float("-inf")):
                return None

            return result
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = text.replace("\u200b", "")
        return re.sub(r"[ \t]+", " ", text).lower()

    def _classify_statement_type(self, text: str) -> tuple[str, list[str]]:
        credit_score = 0
        bank_score = 0

        credit_matches = []
        bank_matches = []

        for keyword, weight in self.CREDIT_CARD_SIGNALS.items():
            if keyword in text:
                credit_score += weight
                credit_matches.append(keyword)

        for keyword, weight in self.BANK_ACCOUNT_SIGNALS.items():
            if keyword in text:
                bank_score += weight
                bank_matches.append(keyword)

        # Strong explicit signals override weak generic signals.
        explicit_credit = any(
            keyword in text
            for keyword in (
                "credit card statement",
                "card limit",
                "credit limit",
                "available credit",
                "minimum payment due",
                "minimum amount due",
                "cashback card",
                "cash back card",
            )
        )

        explicit_bank = any(
            keyword in text
            for keyword in (
                "bank account statement",
                "savings account",
                "current account",
                "type of account",
            )
        )

        if explicit_credit and not explicit_bank:
            result = "credit_card"
        elif explicit_bank and not explicit_credit:
            result = "bank_account"
        elif credit_score > bank_score and credit_score > 0:
            result = "credit_card"
        elif bank_score > credit_score and bank_score > 0:
            result = "bank_account"
        else:
            result = "unknown"

        return result, credit_matches + bank_matches

    def _detect_text(self, pages: List[dict]) -> DocumentMetadata:
        full_text = "\n".join(
            str(page.get("text", "") or "")
            for page in pages
        )

        text = self._normalize_text(full_text)
        detected_keywords: list[str] = []

        # -------------------------------------------------
        # Document type
        # -------------------------------------------------
        statement_score = sum(
            1 for keyword in self.STATEMENT_KEYWORDS
            if keyword in text
        )

        document_type = (
            "financial_statement"
            if statement_score > 0
            else "unknown"
        )

        # -------------------------------------------------
        # Statement type
        # -------------------------------------------------
        statement_type, statement_keywords = (
            self._classify_statement_type(text)
        )
        detected_keywords.extend(statement_keywords)

        for keyword in self.STATEMENT_KEYWORDS:
            if keyword in text:
                detected_keywords.append(keyword)

        # -------------------------------------------------
        # Currency
        # -------------------------------------------------
        currency_scores = {}

        for currency, patterns in self.CURRENCY_PATTERNS.items():
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
                currency_scores[currency] = score

        currency = None
        currency_symbol = None
        country = None

        if currency_scores:
            # ISO codes are more reliable than symbols that can represent
            # more than one currency.
            iso_scores = {}
            for code in self.CURRENCY_PATTERNS:
                iso_scores[code] = len(
                    re.findall(
                        rf"\b{re.escape(code)}\b",
                        full_text,
                        flags=re.IGNORECASE,
                    )
                )

            strongest_iso = max(iso_scores, key=iso_scores.get)
            if iso_scores[strongest_iso] > 0:
                currency = strongest_iso
            else:
                currency = max(
                    currency_scores,
                    key=currency_scores.get,
                )

            currency_symbol = self.CURRENCY_SYMBOLS.get(currency)
            country = self.COUNTRY_CURRENCY_MAP.get(currency)

        # -------------------------------------------------
        # Dates
        # -------------------------------------------------
        all_dates = []

        for pattern in self.DATE_PATTERNS:
            all_dates.extend(pattern.findall(full_text))

        start_date = None
        end_date = None
        statement_year = None
        date_format = None

        # Use explicitly labeled statement periods where available.
        period_patterns = [
            re.compile(
                r"(?:statement period|period)\s*[:\-]?\s*"
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*"
                r"(?:to|through|\-)\s*"
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                re.IGNORECASE,
            ),
            re.compile(
                r"(20\d{2}-\d{2}-\d{2})\s*(?:to|\-)\s*"
                r"(20\d{2}-\d{2}-\d{2})",
                re.IGNORECASE,
            ),
        ]

        for pattern in period_patterns:
            match = pattern.search(full_text)
            if match:
                start_date = match.group(1)
                end_date = match.group(2)
                break

        if not start_date and not end_date:
            iso_period = re.search(
                r"\b(20\d{2}-\d{2}-\d{2})\s*(?:to|\-)\s*"
                r"(20\d{2}-\d{2}-\d{2})\b",
                full_text,
                flags=re.IGNORECASE,
            )
            if iso_period:
                start_date = iso_period.group(1)
                end_date = iso_period.group(2)

        if end_date:
            year_match = re.search(r"(?:19|20)\d{2}", end_date)
            if year_match:
                statement_year = int(year_match.group(0))
        elif start_date:
            year_match = re.search(r"(?:19|20)\d{2}", start_date)
            if year_match:
                statement_year = int(year_match.group(0))

        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", full_text):
            date_format = "YYYY-MM-DD"
        elif re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", full_text):
            date_format = "DD/MM/YYYY or MM/DD/YYYY"
        elif re.search(r"\b\d{1,2}-\d{1,2}-\d{4}\b", full_text):
            date_format = "DD-MM-YYYY or MM-DD-YYYY"

        # -------------------------------------------------
        # Statement financial snapshot
        # -------------------------------------------------
        opening_balance = None
        opening_match = re.search(
            r"OPENING\s+BALANCE\s*[:\-]?\s*"
            r"([0-9][0-9,]*\.\d{2})",
            full_text,
            flags=re.IGNORECASE,
        )

        if opening_match:
            opening_balance = self._money_value(opening_match.group(1))

        card_limit = None
        available_limit = None
        minimum_payment_due = None
        payment_due_date = None
        total_payment_due = None
        profit_other_charges = None
        current_balance = None

        card_header_index = text.find("card limit")

        if card_header_index >= 0:
            summary_window = full_text[
                card_header_index:card_header_index + 8000
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
                after_due = summary_window[first_part.end():]
                trailing_values = re.findall(
                    r"[0-9][0-9,]*\.\d{2}",
                    after_due[:1200],
                )

                if len(trailing_values) >= 3:
                    card_limit = self._money_value(first_part.group(1))
                    available_limit = self._money_value(first_part.group(2))
                    minimum_payment_due = self._money_value(first_part.group(3))
                    payment_due_date = first_part.group(4)
                    total_payment_due = self._money_value(trailing_values[0])
                    profit_other_charges = self._money_value(trailing_values[1])
                    current_balance = self._money_value(trailing_values[2])

        # -------------------------------------------------
        # Transaction table / columns
        # -------------------------------------------------
        has_posting_date = any(
            term in text
            for term in ("posting date", "posted date", "value date")
        )

        has_transaction_date = any(
            term in text
            for term in (
                "transaction date",
                "trans date",
                "txn date",
                "date",
            )
        )

        has_withdrawal_column = "withdrawals" in text
        has_deposit_column = "deposits" in text
        has_balance_column = "balance" in text

        has_transaction_id = (
            "tran id" in text
            or "transaction id" in text
            or "reference number" in text
            or "reference no" in text
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
                and any(
                    term in text
                    for term in (
                        "particulars",
                        "description",
                        "transaction details",
                        "transaction description",
                        "merchant",
                    )
                )
            )
            or (
                has_posting_date
                and has_transaction_date
                and "description" in text
            )
        )

        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------
        checks = [
            document_type != "unknown",
            statement_type != "unknown",
            currency is not None,
            has_transaction_table,
        ]

        confidence = sum(checks) / len(checks) if checks else 0.0

        return DocumentMetadata(
            document_type=document_type,
            statement_type=statement_type,
            country=country,
            currency=currency,
            currency_symbol=currency_symbol,
            date_format=date_format,
            statement_start_date=start_date,
            statement_end_date=end_date,
            statement_year=statement_year,
            has_transaction_table=has_transaction_table,
            has_posting_date=has_posting_date,
            has_transaction_date=has_transaction_date,
            has_debit_credit_indicator=has_debit_credit_indicator,
            has_withdrawal_column=has_withdrawal_column,
            has_deposit_column=has_deposit_column,
            has_balance_column=has_balance_column,
            has_transaction_id=has_transaction_id,
            page_count=len(pages),
            confidence=round(confidence, 2),
            detected_keywords=list(dict.fromkeys(detected_keywords)),
            opening_balance=opening_balance,
            card_limit=card_limit,
            available_limit=available_limit,
            minimum_payment_due=minimum_payment_due,
            payment_due_date=payment_due_date,
            total_payment_due=total_payment_due,
            profit_other_charges=profit_other_charges,
            current_balance=current_balance,
        )