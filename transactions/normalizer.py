from __future__ import annotations

import calendar
import re

from datetime import date
from decimal import Decimal, InvalidOperation


class TransactionNormalizer:

    MONTHS = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    def parse_date(
        self,
        value: str,
        year: int | None = None,
        day_first: bool = True,
    ) -> date:

        value = value.strip().upper()

        # ---------------------------------------------
        # DD MMM
        # ---------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})\s+([A-Z]{3})",
            value,
        )

        if match:

            if year is None:

                raise ValueError(
                    "Year is required for DD MMM dates."
                )

            day = int(match.group(1))

            month = self.MONTHS.get(
                match.group(2)
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return date(
                year,
                month,
                day,
            )

        # ---------------------------------------------
        # DD-MMM-YYYY
        # ---------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})-([A-Z]{3})-(\d{4})",
            value,
        )

        if match:

            day = int(match.group(1))

            month = self.MONTHS.get(
                match.group(2)
            )

            parsed_year = int(
                match.group(3)
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return date(
                parsed_year,
                month,
                day,
            )

        # ---------------------------------------------
        # YYYY-MM-DD
        # ---------------------------------------------

        match = re.fullmatch(
            r"(\d{4})-(\d{1,2})-(\d{1,2})",
            value,
        )

        if match:

            return date(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )

        # ---------------------------------------------
        # DD/MM/YYYY or DD-MM-YYYY
        # ---------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})[/-]"
            r"(\d{1,2})[/-]"
            r"(\d{4})",
            value,
        )

        if match:

            first = int(match.group(1))
            second = int(match.group(2))
            parsed_year = int(
                match.group(3)
            )

            if day_first:

                day = first
                month = second

            else:

                month = first
                day = second

            return date(
                parsed_year,
                month,
                day,
            )

        raise ValueError(
            f"Unsupported date format: {value}"
        )

    def parse_month_day(
        self,
        value: str,
        year: int,
    ) -> date:

        return self.parse_date(
            value,
            year=year,
        )

    def parse_numeric_date(
        self,
        value: str,
        day_first: bool = True,
    ) -> date:

        return self.parse_date(
            value,
            day_first=day_first,
        )

    def parse_amount(
        self,
        value: str,
    ) -> dict:

        if value is None:

            raise ValueError(
                "Amount cannot be None"
            )

        raw = str(value).strip().upper()

        is_credit = bool(
            re.search(
                r"CR\s*$",
                raw,
            )
        )

        is_debit = bool(
            re.search(
                r"DR\s*$",
                raw,
            )
        )

        cleaned = re.sub(
            r"(?:CR|DR)\s*$",
            "",
            raw,
        ).strip()

        negative_parentheses = (
            cleaned.startswith("(")
            and cleaned.endswith(")")
        )

        cleaned = (
            cleaned
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )

        cleaned = re.sub(
            r"[^\d.\-]",
            "",
            cleaned,
        )

        if not cleaned:

            raise ValueError(
                f"Could not parse amount: {value}"
            )

        try:

            amount = Decimal(cleaned)

        except InvalidOperation as exc:

            raise ValueError(
                f"Invalid amount: {value}"
            ) from exc

        if negative_parentheses:

            amount = -abs(amount)

        return {
            "amount": amount,
            "is_credit": is_credit,
            "is_debit": is_debit,
        }