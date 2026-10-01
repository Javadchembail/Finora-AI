from __future__ import annotations

import re
from typing import Dict, List, Optional


class BankTransactionExtractor:
    """
    Extract bank-account transactions from semi-structured PDF text.

    Designed to handle formats where:
    - Transaction date and value date are present
    - Particulars/description may span multiple lines
    - Transaction type and transaction ID appear on the same row
    - Withdrawals/deposits are flattened by PDF text extraction
    - Balance and CR/DR indicator appear at the end

    Important:
    The final CR/DR in the Federal Bank format represents the
    balance direction, NOT necessarily the transaction direction.
    """

    DATE_PATTERN = re.compile(
        r"^\s*"
        r"(?P<transaction_date>\d{1,2}-[A-Za-z]{3}-\d{4})\s+"
        r"(?P<value_date>\d{1,2}-[A-Za-z]{3}-\d{4})\s+"
        r"(?P<rest>.*)$",
        re.IGNORECASE,
    )

    TRANSACTION_END_PATTERN = re.compile(
        r"(?P<tran_type>[A-Z]+)\s+"
        r"(?P<tran_id>[A-Z0-9]+)\s+"
        r"(?P<amount>"
        r"\(?-?(?:\d{1,3}(?:,\d{3})*|\d+)"
        r"(?:\.\d{1,2})?\)?"
        r")\s+"
        r"(?P<balance>"
        r"\(?-?(?:\d{1,3}(?:,\d{3})*|\d+)"
        r"(?:\.\d{1,2})?\)?"
        r")\s+"
        r"(?P<drcr>DR|CR)"
        r"\s*$",
        re.IGNORECASE,
    )

    HEADER_MARKERS = (
        "date value date particulars",
        "opening balance",
        "grand total",
        "abbreviations used",
        "disclaimer",
        "end of statement",
    )

    FOOTER_MARKERS = (
        "the federal bank ltd.",
        "page ",
        "ph:",
        "website:",
    )

    def extract_from_page(
        self,
        page_number: int,
        text: str,
    ) -> List[Dict]:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        transactions: List[Dict] = []

        current: Optional[Dict] = None

        for line_number, line in enumerate(lines, start=1):

            lower = line.lower()

            # ---------------------------------------------------------
            # Skip headers / document-level markers
            # ---------------------------------------------------------
            if any(marker in lower for marker in self.HEADER_MARKERS):

                if current:
                    parsed = self._finalize(current)

                    if parsed:
                        transactions.append(parsed)

                    current = None

                continue

            # ---------------------------------------------------------
            # Skip footer lines
            # ---------------------------------------------------------
            if any(marker in lower for marker in self.FOOTER_MARKERS):
                continue

            # ---------------------------------------------------------
            # Opening balance
            # ---------------------------------------------------------
            if lower.startswith("opening balance"):

                if current:
                    parsed = self._finalize(current)

                    if parsed:
                        transactions.append(parsed)

                    current = None

                continue

            # ---------------------------------------------------------
            # Detect a new transaction
            # ---------------------------------------------------------
            date_match = self.DATE_PATTERN.match(line)

            if date_match:

                # Finish previous transaction
                if current:

                    parsed = self._finalize(current)

                    if parsed:
                        transactions.append(parsed)

                current = {
                    "page_number": page_number,
                    "line_number": line_number,
                    "transaction_date_raw": date_match.group(
                        "transaction_date"
                    ),
                    "value_date_raw": date_match.group(
                        "value_date"
                    ),
                    "description_lines": [
                        date_match.group("rest").strip()
                    ],
                }

                # Try immediately because Federal Bank usually
                # puts the transaction metadata on the same line.
                parsed = self._try_parse_transaction(current)

                if parsed:
                    current["parsed"] = parsed

                continue

            # ---------------------------------------------------------
            # Continuation line
            # ---------------------------------------------------------
            if current:

                current["description_lines"].append(line)

        # -------------------------------------------------------------
        # Final transaction
        # -------------------------------------------------------------
        if current:

            parsed = self._finalize(current)

            if parsed:
                transactions.append(parsed)

        return transactions

    # -----------------------------------------------------------------
    # Multiple pages
    # -----------------------------------------------------------------

    def extract_from_pages(
        self,
        pages: List[dict],
    ) -> List[Dict]:

        all_transactions: List[Dict] = []

        for page in pages:

            page_transactions = self.extract_from_page(
                page_number=page["page_number"],
                text=page.get("text", ""),
            )

            all_transactions.extend(page_transactions)

        return all_transactions

    # -----------------------------------------------------------------
    # Try parsing transaction metadata
    # -----------------------------------------------------------------

    def _try_parse_transaction(
        self,
        current: Dict,
    ) -> Optional[Dict]:

        combined = self._combine_description_lines(current)

        match = self.TRANSACTION_END_PATTERN.search(combined)

        if not match:
            return None

        description = combined[:match.start()].strip()

        if not description:
            return None

        amount = match.group("amount")
        balance = match.group("balance")
        balance_direction = match.group("drcr").upper()

        tran_type = match.group("tran_type").upper()
        tran_id = match.group("tran_id")

        transaction_direction = self._detect_direction(
            description=description,
            drcr=balance_direction,
        )

        return {
            "page_number": current["page_number"],
            "line_number": current["line_number"],
            "transaction_date_raw": current[
                "transaction_date_raw"
            ],
            "value_date_raw": current[
                "value_date_raw"
            ],
            "description_raw": description,
            "tran_type": tran_type,
            "tran_id": tran_id,
            "amount_raw": amount,
            "balance_raw": balance,
            "dr_cr": balance_direction,
            "direction": transaction_direction,
        }

    # -----------------------------------------------------------------
    # Finalize transaction
    # -----------------------------------------------------------------

    def _finalize(
        self,
        current: Dict,
    ) -> Optional[Dict]:

        # -------------------------------------------------------------
        # First try the original parsed result.
        #
        # This is important because the first physical line may contain
        # all transaction metadata while later lines contain additional
        # description/details.
        # -------------------------------------------------------------

        parsed = current.get("parsed")

        combined = self._combine_description_lines(current)

        match = self.TRANSACTION_END_PATTERN.search(combined)

        if not match:

            if parsed:
                return parsed

            return None

        description = combined[:match.start()].strip()

        if not description:

            if parsed:
                return parsed

            return None

        amount = match.group("amount")
        balance = match.group("balance")
        balance_direction = match.group("drcr").upper()

        tran_type = match.group("tran_type").upper()
        tran_id = match.group("tran_id")

        transaction_direction = self._detect_direction(
            description=description,
            drcr=balance_direction,
        )

        return {
            "page_number": current["page_number"],
            "line_number": current["line_number"],
            "transaction_date_raw": current[
                "transaction_date_raw"
            ],
            "value_date_raw": current[
                "value_date_raw"
            ],
            "description_raw": description,
            "tran_type": tran_type,
            "tran_id": tran_id,
            "amount_raw": amount,
            "balance_raw": balance,
            "dr_cr": balance_direction,
            "direction": transaction_direction,
        }

    # -----------------------------------------------------------------
    # Combine description lines
    # -----------------------------------------------------------------

    def _combine_description_lines(
        self,
        current: Dict,
    ) -> str:

        combined = " ".join(
            current.get("description_lines", [])
        )

        combined = re.sub(
            r"\s+",
            " ",
            combined,
        ).strip()

        return combined

    # -----------------------------------------------------------------
    # Detect transaction direction
    # -----------------------------------------------------------------

    def _detect_direction(
        self,
        description: str,
        drcr: str,
    ) -> str:

        upper = description.upper()

        # -------------------------------------------------------------
        # Federal Bank UPI outgoing
        # -------------------------------------------------------------
        if "UPIOUT" in upper:
            return "debit"

        # -------------------------------------------------------------
        # Federal Bank UPI incoming
        # -------------------------------------------------------------
        if re.search(
            r"\bUPI\s+IN\b",
            upper,
        ):
            return "credit"

        # -------------------------------------------------------------
        # IMPS incoming
        # -------------------------------------------------------------
        if re.search(
            r"\bIMPS\b.*\bINW\b",
            upper,
        ):
            return "credit"

        # -------------------------------------------------------------
        # Savings account interest
        # -------------------------------------------------------------
        if upper.startswith("SBINT"):
            return "credit"

        # -------------------------------------------------------------
        # NEFT incoming patterns
        # -------------------------------------------------------------
        if re.search(
            r"\bNEFT\b.*\bIN\b",
            upper,
        ):
            return "credit"

        # -------------------------------------------------------------
        # NEFT outgoing patterns
        # -------------------------------------------------------------
        if re.search(
            r"\bNEFT\b.*\bOUT\b",
            upper,
        ):
            return "debit"

        # -------------------------------------------------------------
        # RTGS incoming
        # -------------------------------------------------------------
        if re.search(
            r"\bRTGS\b.*\bIN\b",
            upper,
        ):
            return "credit"

        # -------------------------------------------------------------
        # RTGS outgoing
        # -------------------------------------------------------------
        if re.search(
            r"\bRTGS\b.*\bOUT\b",
            upper,
        ):
            return "debit"

        # -------------------------------------------------------------
        # ATM withdrawals
        # -------------------------------------------------------------
        if re.search(
            r"\bATM\b",
            upper,
        ):
            return "debit"

        # -------------------------------------------------------------
        # Cash withdrawal
        # -------------------------------------------------------------
        if re.search(
            r"\bCASH\s*WITHDRAW",
            upper,
        ):
            return "debit"

        # -------------------------------------------------------------
        # Do NOT blindly use the final DR/CR.
        #
        # In Federal Bank's statement format, that indicator belongs
        # to the balance column.
        # -------------------------------------------------------------

        return "unknown"