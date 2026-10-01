from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pdfplumber

from ingestion.pdf_reader import PDFReader
from ingestion.document_detector import DocumentDetector
from ingestion.universal_structure import UniversalStructureDetector
from ingestion.universal_row_extractor import UniversalRowExtractor

from transactions.normalizer import TransactionNormalizer
from transactions.merchant_normalizer import MerchantNormalizer
from transactions.validator import UniversalTransactionValidator

from transactions.models import (
    Transaction,
    StatementType,
    TransactionDirection,
    TransactionType,
)


class FinancialStatementPipeline:
    """
    Universal financial statement processing pipeline.

    Architecture:

        PDF
          |
          v
        PDF Reader
          |
          v
        Document Detection
          |
          v
        Universal Structure Detection
          |
          v
        Semantic Column Mapping
          |
          v
        Universal Row Extraction
          |
          v
        Canonical Transaction
          |
          v
        Universal Validation
          |
          v
        Merchant Normalization
          |
          v
        Validated Transactions

    The pipeline does not contain bank-specific extraction branches.

    Federal Bank and Emirates Islamic documents are regression tests,
    not special cases.
    """

    def __init__(
        self,
        file_path: str,
        password: Optional[str] = None,
    ):
        self.file_path = file_path
        self.password = password

        # ---------------------------------------------------------
        # Document reader
        # ---------------------------------------------------------

        self.reader = PDFReader(
            file_path,
            password=password,
        )

        # ---------------------------------------------------------
        # Document detector
        # ---------------------------------------------------------

        self.detector = DocumentDetector()

        # ---------------------------------------------------------
        # Universal structure detector
        # ---------------------------------------------------------

        self.structure_detector = UniversalStructureDetector()

        # ---------------------------------------------------------
        # Universal row extractor
        # ---------------------------------------------------------

        self.row_extractor = UniversalRowExtractor()

        # ---------------------------------------------------------
        # Normalization
        # ---------------------------------------------------------

        self.normalizer = TransactionNormalizer()

        # ---------------------------------------------------------
        # Merchant normalization
        # ---------------------------------------------------------

        self.merchant_normalizer = MerchantNormalizer()

        # ---------------------------------------------------------
        # Universal validation
        # ---------------------------------------------------------

        self.validator = UniversalTransactionValidator()

    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(self) -> List[Transaction]:
        """
        Process the financial document using the universal
        structure-driven extraction pipeline.
        """

        # ---------------------------------------------------------
        # 1. Read document
        # ---------------------------------------------------------

        pages = self.reader.extract_pages()

        if not pages:
            return []

        # ---------------------------------------------------------
        # 2. Detect document metadata
        # ---------------------------------------------------------

        metadata = self.detector.detect(pages)

        # ---------------------------------------------------------
        # 3. Detect universal transaction structure
        # ---------------------------------------------------------

        structure = self.structure_detector.detect(
            self.file_path,
            password=self.password,
        )

        if not structure.get("detected"):
            raise ValueError(
                "Could not detect a financial transaction table "
                "structure in the document."
            )

        columns = structure.get("columns", [])

        if not columns:
            raise ValueError(
                "Financial transaction columns could not be detected."
            )

        structure_score = float(
            structure.get("score", 0.0)
        )

        print(
            f"Universal structure detected: "
            f"page={structure.get('page')}, "
            f"columns={len(columns)}, "
            f"score={structure_score:.2f}"
        )

        for column in columns:
            print(
                "  "
                f"{column.get('header')} "
                f"-> "
                f"{column.get('semantic_type')}"
            )

        # ---------------------------------------------------------
        # 4. Extract transaction rows from PDF geometry
        # ---------------------------------------------------------

        raw_transactions = self._extract_universal_rows(
            columns=columns,
        )

        if not raw_transactions:
            raise ValueError(
                "No transaction rows could be extracted from the "
                "detected financial statement structure."
            )

        print(
            f"Universal rows extracted: "
            f"{len(raw_transactions)}"
        )

        # ---------------------------------------------------------
        # 5. Determine statement year
        # ---------------------------------------------------------

        statement_year = getattr(
            metadata,
            "statement_year",
            None,
        )

        if statement_year is None:
            statement_year = self._infer_statement_year(
                raw_transactions
            )

        # ---------------------------------------------------------
        # 6. Build canonical transactions
        # ---------------------------------------------------------

        transactions: List[Transaction] = []

        for index, raw in enumerate(
            raw_transactions,
            start=1,
        ):
            try:
                transaction = (
                    self._build_transaction(
                        raw=raw,
                        metadata=metadata,
                        index=index,
                        statement_year=statement_year,
                    )
                )

                if transaction is not None:
                    transactions.append(transaction)

            except Exception as exc:
                print(
                    f"WARNING: Could not normalize "
                    f"transaction {index}: {exc}"
                )

        # ---------------------------------------------------------
        # 7. Universal validation
        # ---------------------------------------------------------

        validated_transactions = (
            self._validate_transactions(
                transactions
            )
        )

        print(
            f"Universal transactions validated: "
            f"{len(validated_transactions)}"
        )

        return validated_transactions

    # =============================================================
    # UNIVERSAL ROW EXTRACTION
    # =============================================================

    def _extract_universal_rows(
        self,
        columns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract rows from every PDF page using the same semantic
        structure.

        No bank name is inspected here.

        The detected semantic column structure is reused across
        pages because financial statements normally repeat the same
        transaction table layout throughout the document.
        """

        transactions: List[Dict[str, Any]] = []

        with pdfplumber.open(
            self.file_path,
            password=self.password,
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):
                words = page.extract_words(
                    use_text_flow=False,
                    keep_blank_chars=False,
                )

                if not words:
                    continue

                try:
                    page_transactions = (
                        self.row_extractor.extract(
                            words=words,
                            columns=columns,
                        )
                    )
                except Exception as exc:
                    print(
                        f"WARNING: Universal row extraction "
                        f"failed on page {page_number}: {exc}"
                    )
                    continue

                for row in page_transactions:
                    row["page_number"] = page_number

                    # Keep the source file with the raw row.
                    row["source_file"] = self.file_path

                    transactions.append(row)

        return transactions

    # =============================================================
    # CANONICAL TRANSACTION BUILDING
    # =============================================================

    def _build_transaction(
        self,
        raw: Dict[str, Any],
        metadata: Any,
        index: int,
        statement_year: Optional[int],
    ) -> Optional[Transaction]:
        """
        Convert a universal extracted row into the canonical
        Transaction model.

        The method works from semantic fields rather than from a
        specific bank's column layout.
        """

        # ---------------------------------------------------------
        # Date
        # ---------------------------------------------------------

        transaction_date = self._parse_transaction_date(
            raw.get("transaction_date"),
            statement_year,
        )

        posting_date = self._parse_transaction_date(
            raw.get("posting_date"),
            statement_year,
        )

        value_date = self._parse_transaction_date(
            raw.get("value_date"),
            statement_year,
        )

        if transaction_date is None:
            transaction_date = (
                posting_date
                or value_date
            )

        if posting_date is None:
            posting_date = value_date

        if transaction_date is None:
            return None

        # ---------------------------------------------------------
        # Amount
        # ---------------------------------------------------------

        amount, direction_from_amount = (
            self._extract_amount_and_direction(
                raw
            )
        )

        if amount is None:
            return None

        amount = abs(amount)

        # ---------------------------------------------------------
        # Description
        # ---------------------------------------------------------

        description = str(
            raw.get("description")
            or ""
        ).strip()

        transaction_id = self._clean_optional_text(
            raw.get("transaction_id")
        )

        if not description and not transaction_id:
            return None

        if not description:
            description = transaction_id or "Unknown transaction"

        # ---------------------------------------------------------
        # Direction
        # ---------------------------------------------------------

        direction = (
            direction_from_amount
            or self._infer_direction(
                raw=raw,
                metadata=metadata,
            )
        )

        # ---------------------------------------------------------
        # Statement type
        # ---------------------------------------------------------

        statement_type = self._resolve_statement_type(
            metadata
        )

        # ---------------------------------------------------------
        # Currency
        # ---------------------------------------------------------

        currency = self._resolve_currency(
            metadata
        )

        # ---------------------------------------------------------
        # Merchant
        # ---------------------------------------------------------

        merchant, merchant_confidence = (
            self.merchant_normalizer.normalize(
                description
            )
        )

        # ---------------------------------------------------------
        # Transaction type
        # ---------------------------------------------------------

        transaction_type = (
            self._infer_transaction_type(
                description=description,
                direction=direction,
            )
        )

        # ---------------------------------------------------------
        # Statement amount
        # ---------------------------------------------------------

        statement_amount = self._parse_decimal(
            raw.get("amount")
        )

        if statement_amount is None:
            statement_amount = self._parse_decimal(
                raw.get("debit")
            )

        if statement_amount is None:
            statement_amount = self._parse_decimal(
                raw.get("credit")
            )

        # ---------------------------------------------------------
        # Balance
        # ---------------------------------------------------------

        running_balance = self._parse_decimal(
            raw.get("balance")
        )

        # ---------------------------------------------------------
        # Bank transaction ID
        # ---------------------------------------------------------

        bank_transaction_id = (
            self._clean_optional_text(
                raw.get("transaction_id")
            )
        )

        # ---------------------------------------------------------
        # Transaction channel
        # ---------------------------------------------------------

        transaction_channel = (
            self._infer_transaction_channel(
                description
            )
        )

        # ---------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------

        extraction_confidence = (
            self._calculate_extraction_confidence(
                raw=raw,
                structure_confidence=None,
            )
        )

        return Transaction(
            transaction_id=f"TXN-{index:06d}",

            transaction_date=transaction_date,

            posting_date=posting_date,

            description_raw=description,

            description_normalized=merchant,

            merchant=merchant,

            original_amount=amount,

            original_currency=currency,

            statement_amount=(
                abs(statement_amount)
                if statement_amount is not None
                else amount
            ),

            statement_currency=currency,

            direction=direction,

            transaction_type=transaction_type,

            statement_type=statement_type,

            merchant_confidence=merchant_confidence,

            extraction_confidence=extraction_confidence,

            account_identifier=None,

            source_file=self.file_path,

            source_page=(
                raw.get("page_number")
            ),

            bank_transaction_id=(
                bank_transaction_id
            ),

            running_balance=(
                abs(running_balance)
                if running_balance is not None
                else None
            ),

            balance_direction=None,

            transaction_channel=(
                transaction_channel
            ),
        )

    # =============================================================
    # AMOUNT
    # =============================================================

    def _extract_amount_and_direction(
        self,
        raw: Dict[str, Any],
    ):
        """
        Resolve amount and direction from semantic numeric fields.

        Priority:

            debit  -> DEBIT
            credit -> CREDIT
            amount -> infer later

        This avoids assuming that every statement has a debit/credit
        pair.
        """

        debit = self._parse_decimal(
            raw.get("debit")
        )

        credit = self._parse_decimal(
            raw.get("credit")
        )

        amount = self._parse_decimal(
            raw.get("amount")
        )

        if debit is not None:
            return (
                abs(debit),
                TransactionDirection.DEBIT,
            )

        if credit is not None:
            return (
                abs(credit),
                TransactionDirection.CREDIT,
            )

        if amount is not None:
            return (
                abs(amount),
                None,
            )

        return None, None

    # =============================================================
    # DIRECTION
    # =============================================================

    def _infer_direction(
        self,
        raw: Dict[str, Any],
        metadata: Any,
    ) -> TransactionDirection:
        """
        Infer direction when the document exposes only a generic
        amount column.

        The inference is semantic and document-driven. It does not
        inspect a bank name.
        """

        description = str(
            raw.get("description")
            or ""
        ).upper()

        # ---------------------------------------------------------
        # Explicit textual credit indicators
        # ---------------------------------------------------------

        credit_terms = (
            "CREDIT",
            "CR",
            "REFUND",
            "REVERSAL",
            "SALARY",
            "PAYMENT RECEIVED",
            "RECEIVED",
            "DEPOSIT",
            "INTEREST CREDIT",
            "CASHBACK",
            "REWARD",
        )

        for term in credit_terms:
            if term in description:
                return TransactionDirection.CREDIT

        # ---------------------------------------------------------
        # Explicit textual debit indicators
        # ---------------------------------------------------------

        debit_terms = (
            "DEBIT",
            "DR",
            "PURCHASE",
            "WITHDRAWAL",
            "ATM",
            "FEE",
            "CHARGE",
            "TRANSFER TO",
            "PAYMENT TO",
        )

        for term in debit_terms:
            if term in description:
                return TransactionDirection.DEBIT

        # ---------------------------------------------------------
        # Statement-level fallback
        # ---------------------------------------------------------

        statement_type = self._resolve_statement_type(
            metadata
        )

        if statement_type == StatementType.CREDIT_CARD:
            return TransactionDirection.DEBIT

        # Generic financial statements normally represent a
        # transaction amount as an outgoing amount when no explicit
        # direction is available. Validation can flag ambiguous
        # cases later.
        return TransactionDirection.DEBIT

    # =============================================================
    # STATEMENT TYPE
    # =============================================================

    def _resolve_statement_type(
        self,
        metadata: Any,
    ) -> StatementType:
        value = getattr(
            metadata,
            "statement_type",
            None,
        )

        if isinstance(
            value,
            StatementType,
        ):
            return value

        if value:
            try:
                return StatementType(
                    str(value)
                )
            except ValueError:
                pass

        return StatementType.UNKNOWN

    # =============================================================
    # CURRENCY
    # =============================================================

    def _resolve_currency(
        self,
        metadata: Any,
    ) -> str:
        currency = getattr(
            metadata,
            "currency",
            None,
        )

        if currency:
            return str(
                currency
            ).upper()

        return "UNKNOWN"

    # =============================================================
    # TRANSACTION TYPE
    # =============================================================

    def _infer_transaction_type(
        self,
        description: str,
        direction: TransactionDirection,
    ) -> TransactionType:
        """
        Generic semantic transaction classification.

        These rules are intentionally based on transaction meaning,
        not bank identity.
        """

        text = description.upper()

        if any(
            term in text
            for term in (
                "SALARY",
                "PAYROLL",
                "WAGES",
            )
        ):
            return TransactionType.SALARY

        if any(
            term in text
            for term in (
                "REFUND",
                "REVERSAL",
                "REVERSED",
            )
        ):
            return TransactionType.REFUND

        if any(
            term in text
            for term in (
                "INTEREST",
                "INT CREDIT",
                "INT PAID",
            )
        ):
            return TransactionType.INTEREST

        if any(
            term in text
            for term in (
                "FEE",
                "CHARGE",
                "SERVICE CHARGE",
                "BANK CHARGE",
            )
        ):
            return TransactionType.FEE

        if any(
            term in text
            for term in (
                "TRANSFER",
                "TRF",
                "WIRE",
                "ACH",
                "SEPA",
                "SWIFT",
                "UPI",
                "IMPS",
                "NEFT",
                "RTGS",
            )
        ):
            return TransactionType.TRANSFER

        if any(
            term in text
            for term in (
                "ATM",
                "CASH WITHDRAWAL",
                "CASH WITHDRAW",
            )
        ):
            return TransactionType.CASH_WITHDRAWAL

        if any(
            term in text
            for term in (
                "LOAN",
                "EMI",
                "INSTALLMENT",
            )
        ):
            return TransactionType.LOAN_PAYMENT

        if any(
            term in text
            for term in (
                "BILL",
                "UTILITY",
                "ELECTRICITY",
                "WATER",
                "INTERNET",
                "TELECOM",
            )
        ):
            return TransactionType.BILL_PAYMENT

        if any(
            term in text
            for term in (
                "TAX",
                "GOVERNMENT",
                "VAT",
            )
        ):
            return TransactionType.TAX

        if direction == TransactionDirection.CREDIT:
            return TransactionType.OTHER

        return TransactionType.PURCHASE

    # =============================================================
    # TRANSACTION CHANNEL
    # =============================================================

    def _infer_transaction_channel(
        self,
        description: str,
    ) -> Optional[str]:
        """
        Infer a generic payment channel from transaction text.

        This is enrichment only. It is not used to extract the
        transaction itself.
        """

        text = description.upper()

        channel_patterns = (
            ("UPI", "UPI"),
            ("IMPS", "IMPS"),
            ("NEFT", "NEFT"),
            ("RTGS", "RTGS"),
            ("SWIFT", "SWIFT"),
            ("SEPA", "SEPA"),
            ("ACH", "ACH"),
            ("WIRE", "WIRE"),
            ("ATM", "ATM"),
            ("POS", "POS"),
            ("CONTACTLESS", "CONTACTLESS"),
            ("ONLINE", "ONLINE"),
        )

        for marker, channel in channel_patterns:
            if marker in text:
                return channel

        return None

    # =============================================================
    # DATE PARSING
    # =============================================================

    def _parse_transaction_date(
        self,
        value: Any,
        statement_year: Optional[int],
    ) -> Optional[date]:

        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        try:
            result = self.normalizer.parse_date(
                text,
                year=statement_year,
            )

            if isinstance(result, date):
                return result

            return result

        except Exception:
            return None

    # =============================================================
    # STATEMENT YEAR INFERENCE
    # =============================================================

    def _infer_statement_year(
        self,
        rows: List[Dict[str, Any]],
    ) -> Optional[int]:

        for row in rows:

            for field in (
                "transaction_date",
                "posting_date",
                "value_date",
            ):
                value = row.get(field)

                if not value:
                    continue

                text = str(value)

                # YYYY-MM-DD
                parts = text.split("-")

                if (
                    len(parts) == 3
                    and len(parts[0]) == 4
                    and parts[0].isdigit()
                ):
                    return int(
                        parts[0]
                    )

                # DD-MMM-YYYY
                if (
                    len(parts) == 3
                    and parts[-1].isdigit()
                    and len(parts[-1]) == 4
                ):
                    return int(
                        parts[-1]
                    )

                # DD/MM/YYYY
                parts = text.split("/")

                if (
                    len(parts) == 3
                    and parts[-1].isdigit()
                    and len(parts[-1]) == 4
                ):
                    return int(
                        parts[-1]
                    )

        return None

    # =============================================================
    # DECIMAL PARSING
    # =============================================================

    def _parse_decimal(
        self,
        value: Any,
    ) -> Optional[Decimal]:

        if value is None:
            return None

        if isinstance(
            value,
            Decimal,
        ):
            return value

        text = str(value).strip()

        if not text:
            return None

        negative = False

        if text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1]

        text = (
            text
            .replace(",", "")
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("AED", "")
            .replace("INR", "")
            .replace("USD", "")
            .replace("EUR", "")
            .strip()
        )

        # Handle common trailing credit/debit markers.
        upper = text.upper()

        if upper.endswith("CR"):
            text = text[:-2].strip()

        elif upper.endswith("DR"):
            negative = True
            text = text[:-2].strip()

        try:
            number = Decimal(
                text
            )

            if negative:
                number = -abs(number)

            return number

        except (
            InvalidOperation,
            ValueError,
        ):
            return None

    # =============================================================
    # TEXT CLEANING
    # =============================================================

    def _clean_optional_text(
        self,
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(value).strip()

        return text or None

    # =============================================================
    # EXTRACTION CONFIDENCE
    # =============================================================

    def _calculate_extraction_confidence(
        self,
        raw: Dict[str, Any],
        structure_confidence: Optional[float],
    ) -> float:
        """
        Estimate confidence from the fields actually recovered.

        This is intentionally conservative and independent of bank
        identity.
        """

        score = 0.50

        if raw.get("transaction_date"):
            score += 0.10

        if raw.get("posting_date"):
            score += 0.05

        if raw.get("value_date"):
            score += 0.05

        if raw.get("description"):
            score += 0.10

        if raw.get("amount"):
            score += 0.10

        if (
            raw.get("debit")
            or raw.get("credit")
        ):
            score += 0.10

        if raw.get("transaction_id"):
            score += 0.05

        if raw.get("balance"):
            score += 0.05

        if structure_confidence is not None:
            score = (
                score * 0.70
                + float(structure_confidence) * 0.30
            )

        return min(
            round(score, 4),
            1.0,
        )

    # =============================================================
    # VALIDATION
    # =============================================================

    def _validate_transactions(
        self,
        transactions: List[Transaction],
    ) -> List[Transaction]:
        """
        Validate all canonical transactions.

        Validation is performed after universal extraction so the
        validator remains completely independent of document layout
        or bank identity.
        """

        validated_transactions: List[
            Transaction
        ] = []

        seen_transaction_keys: Set[str] = set()

        previous_transaction: Optional[
            Transaction
        ] = None

        for transaction in transactions:

            result = self.validator.validate(
                transaction=transaction,
                previous_transaction=previous_transaction,
                seen_transaction_keys=(
                    seen_transaction_keys
                ),
            )

            # -----------------------------------------------------
            # Validation confidence
            # -----------------------------------------------------

            transaction.extraction_confidence = (
                result.confidence
            )

            # -----------------------------------------------------
            # Manual review
            # -----------------------------------------------------

            transaction.requires_review = (
                result.requires_review
            )

            # -----------------------------------------------------
            # Validation notes
            # -----------------------------------------------------

            if result.issues:

                issue_messages = []

                for issue in result.issues:

                    issue_messages.append(
                        f"{issue.code}: "
                        f"{issue.message}"
                    )

                transaction.notes = (
                    " | ".join(
                        issue_messages
                    )
                )

            # -----------------------------------------------------
            # Keep transaction
            # -----------------------------------------------------

            validated_transactions.append(
                transaction
            )

            # -----------------------------------------------------
            # Duplicate tracking
            # -----------------------------------------------------

            key = (
                self.validator._transaction_key(
                    transaction
                )
            )

            seen_transaction_keys.add(
                key
            )

            # -----------------------------------------------------
            # Previous transaction
            # -----------------------------------------------------

            previous_transaction = transaction

        return validated_transactions