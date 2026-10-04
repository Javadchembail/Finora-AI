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

    # =========================================================
    # DATE PARSING
    # =========================================================

    def parse_date(
        self,
        value: str,
        year: int | None = None,
        day_first: bool = True,
    ) -> date:
        """
        Parse common financial-statement date formats.

        Supported examples:

            16 JUN
            16 JUN 19
            16 JUN 2019
            16-JUN-19
            16-JUN-2019
            16/JUN/19
            16/JUN/2019
            2019-06-16
            16/06/19
            16/06/2019
            16-06-19
            16-06-2019
            16.06.19
            16.06.2019

        For two-digit years:

            00-68 -> 2000-2068
            69-99 -> 1969-1999

        Example:

            16 JUN 19 -> 2019-06-16
        """

        if value is None:
            raise ValueError(
                "Date cannot be None"
            )

        value = str(value).strip().upper()

        if not value:
            raise ValueError(
                "Date cannot be empty"
            )

        # Normalize repeated whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        # Remove common ordinal suffixes.
        #
        # Examples:
        #   1ST JAN 2025
        #   2ND JAN 2025
        #   3RD JAN 2025
        #   4TH JAN 2025
        #
        value = re.sub(
            r"(\d{1,2})(ST|ND|RD|TH)\b",
            r"\1",
            value,
        )

        # -----------------------------------------------------
        # DD MMM
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})\s+([A-Z]{3})",
            value,
        )

        if match:

            if year is None:

                raise ValueError(
                    "Year is required for DD MMM dates."
                )

            day = int(
                match.group(1)
            )

            month = self.MONTHS.get(
                match.group(2)
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return self._safe_date(
                year=year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD MMM YY
        #
        # IMPORTANT:
        #
        # This is the format used by SS1:
        #
        #   16 JUN 19
        #   17 JUN 19
        #   18 JUN 19
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})\s+"
            r"([A-Z]{3})\s+"
            r"(\d{2})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

            month = self.MONTHS.get(
                match.group(2)
            )

            parsed_year = self._normalize_two_digit_year(
                int(match.group(3))
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD MMM YYYY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})\s+"
            r"([A-Z]{3})\s+"
            r"(\d{4})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

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

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD-MMM-YY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})-"
            r"([A-Z]{3})-"
            r"(\d{2})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

            month = self.MONTHS.get(
                match.group(2)
            )

            parsed_year = self._normalize_two_digit_year(
                int(match.group(3))
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD-MMM-YYYY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})-"
            r"([A-Z]{3})-"
            r"(\d{4})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

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

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD/MMM/YY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})/"
            r"([A-Z]{3})/"
            r"(\d{2})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

            month = self.MONTHS.get(
                match.group(2)
            )

            parsed_year = self._normalize_two_digit_year(
                int(match.group(3))
            )

            if month is None:

                raise ValueError(
                    f"Unknown month: {value}"
                )

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD/MMM/YYYY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})/"
            r"([A-Z]{3})/"
            r"(\d{4})",
            value,
        )

        if match:

            day = int(
                match.group(1)
            )

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

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # YYYY-MM-DD
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{4})-"
            r"(\d{1,2})-"
            r"(\d{1,2})",
            value,
        )

        if match:

            return self._safe_date(
                year=int(match.group(1)),
                month=int(match.group(2)),
                day=int(match.group(3)),
                original=value,
            )

        # -----------------------------------------------------
        # YYYY/MM/DD
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{4})/"
            r"(\d{1,2})/"
            r"(\d{1,2})",
            value,
        )

        if match:

            return self._safe_date(
                year=int(match.group(1)),
                month=int(match.group(2)),
                day=int(match.group(3)),
                original=value,
            )

        # -----------------------------------------------------
        # YYYY.MM.DD
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{4})\."
            r"(\d{1,2})\."
            r"(\d{1,2})",
            value,
        )

        if match:

            return self._safe_date(
                year=int(match.group(1)),
                month=int(match.group(2)),
                day=int(match.group(3)),
                original=value,
            )

        # -----------------------------------------------------
        # DD/MM/YY
        # DD-MM-YY
        # DD.MM.YY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})"
            r"([/.-])"
            r"(\d{1,2})"
            r"\2"
            r"(\d{2})",
            value,
        )

        if match:

            first = int(
                match.group(1)
            )

            second = int(
                match.group(3)
            )

            parsed_year = self._normalize_two_digit_year(
                int(match.group(4))
            )

            if day_first:

                day = first
                month = second

            else:

                month = first
                day = second

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        # -----------------------------------------------------
        # DD/MM/YYYY
        # DD-MM-YYYY
        # DD.MM.YYYY
        # -----------------------------------------------------

        match = re.fullmatch(
            r"(\d{1,2})"
            r"([/.-])"
            r"(\d{1,2})"
            r"\2"
            r"(\d{4})",
            value,
        )

        if match:

            first = int(
                match.group(1)
            )

            second = int(
                match.group(3)
            )

            parsed_year = int(
                match.group(4)
            )

            if day_first:

                day = first
                month = second

            else:

                month = first
                day = second

            return self._safe_date(
                year=parsed_year,
                month=month,
                day=day,
                original=value,
            )

        raise ValueError(
            f"Unsupported date format: {value}"
        )

    # =========================================================
    # TWO-DIGIT YEAR
    # =========================================================

    def _normalize_two_digit_year(
        self,
        value: int,
    ) -> int:
        """
        Convert a two-digit financial-statement year
        into a four-digit year.

        Examples:

            19 -> 2019
            24 -> 2024
            26 -> 2026
            68 -> 2068
            69 -> 1969
            99 -> 1999

        This follows the conventional POSIX-style
        two-digit-year interpretation.
        """

        if not 0 <= value <= 99:

            raise ValueError(
                f"Invalid two-digit year: {value}"
            )

        if value <= 68:

            return 2000 + value

        return 1900 + value

    # =========================================================
    # SAFE DATE
    # =========================================================

    def _safe_date(
        self,
        year: int,
        month: int,
        day: int,
        original: str,
    ) -> date:
        """
        Construct a date while providing a useful error
        message for invalid calendar dates.
        """

        try:

            return date(
                year,
                month,
                day,
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid calendar date: {original}"
            ) from exc

    # =========================================================
    # MONTH + DAY
    # =========================================================

    def parse_month_day(
        self,
        value: str,
        year: int,
    ) -> date:

        return self.parse_date(
            value,
            year=year,
        )

    # =========================================================
    # NUMERIC DATE
    # =========================================================

    def parse_numeric_date(
        self,
        value: str,
        day_first: bool = True,
    ) -> date:

        return self.parse_date(
            value,
            day_first=day_first,
        )

    # =========================================================
    # AMOUNT PARSING
    # =========================================================

    def parse_amount(
        self,
        value: str,
    ) -> dict:
        """
        Parse financial amounts.

        Supports:

            1,234.50
            1234.50
            1234.50 CR
            1234.50 DR
            (1234.50)
            ₹1234.50
            AED 1234.50
            INR 1234.50
        """

        if value is None:

            raise ValueError(
                "Amount cannot be None"
            )

        raw = str(value).strip().upper()

        if not raw:

            raise ValueError(
                "Amount cannot be empty"
            )

        # -----------------------------------------------------
        # Credit / debit markers
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Parentheses indicate negative amount
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Currency symbols / text
        # -----------------------------------------------------

        cleaned = (
            cleaned
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
        )

        cleaned = re.sub(
            r"\b(?:AED|INR|USD|EUR|GBP)\b",
            "",
            cleaned,
        )

        # -----------------------------------------------------
        # Keep only numeric characters
        # -----------------------------------------------------

        cleaned = re.sub(
            r"[^\d.\-]",
            "",
            cleaned,
        )

        if not cleaned:

            raise ValueError(
                f"Could not parse amount: {value}"
            )

        # -----------------------------------------------------
        # Decimal conversion
        # -----------------------------------------------------

        try:

            amount = Decimal(
                cleaned
            )

        except InvalidOperation as exc:

            raise ValueError(
                f"Invalid amount: {value}"
            ) from exc

        # -----------------------------------------------------
        # Parentheses = negative
        # -----------------------------------------------------

        if negative_parentheses:

            amount = -abs(amount)

        return {
            "amount": amount,
            "is_credit": is_credit,
            "is_debit": is_debit,
        }