from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set

import pdfplumber

from ingestion.pdf_reader import PDFReader
from ingestion.document_detector import DocumentDetector
from ingestion.universal_structure import UniversalStructureDetector
from ingestion.universal_row_extractor import UniversalRowExtractor
from ingestion.ocr_fallback import OCRFallback

from transactions.normalizer import TransactionNormalizer
from transactions.merchant_normalizer import MerchantNormalizer
from transactions.validator import UniversalTransactionValidator

from transactions.models import (
    Transaction,
    StatementType,
    TransactionDirection,
    TransactionType,
)

from learning.category_memory import (
    CategoryMemory,
    HybridCategoryEngine,
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
        Merchant Normalization
          |
          v
        Transaction Type Detection
          |
          v
        Category Intelligence
          |
          v
        Category / Subcategory / Confidence
          |
          v
        Universal Validation
          |
          v
        Validated Transactions

    Important:
        This pipeline is bank-agnostic.

        Federal Bank, Emirates Islamic, SS1, or any other
        statement are treated as regression tests only.

        No bank name is used for extraction or categorization.
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

        self.structure_detector = (
            UniversalStructureDetector()
        )

        # ---------------------------------------------------------
        # Universal row extractor
        # ---------------------------------------------------------

        self.row_extractor = (
            UniversalRowExtractor()
        )

        # ---------------------------------------------------------
        # OCR fallback
        # ---------------------------------------------------------

        self.ocr_fallback = OCRFallback(
            file_path=file_path,
            password=password,
        )

        # ---------------------------------------------------------
        # Normalization
        # ---------------------------------------------------------

        self.normalizer = TransactionNormalizer()

        # ---------------------------------------------------------
        # Merchant normalization
        # ---------------------------------------------------------

        self.merchant_normalizer = (
            MerchantNormalizer()
        )

        # ---------------------------------------------------------
        # Universal validation
        # ---------------------------------------------------------

        self.validator = (
            UniversalTransactionValidator()
        )

        # ---------------------------------------------------------
        # Category memory
        #
        # User corrections are persisted here.
        # ---------------------------------------------------------

        self.category_memory = CategoryMemory()

        # ---------------------------------------------------------
        # Universal category engine
        # ---------------------------------------------------------

        self.category_engine = (
            HybridCategoryEngine(
                self.category_memory
            )
        )

    # =============================================================
    # MAIN PIPELINE
    # =============================================================

    def run(
        self,
    ) -> List[Transaction]:

        # ---------------------------------------------------------
        # 1. Read document
        # ---------------------------------------------------------

        pages = self.reader.extract_pages()

        if not pages:
            return []

        # ---------------------------------------------------------
        # 2. Detect document metadata
        # ---------------------------------------------------------

        metadata = self.detector.detect(
            pages
        )

        # ---------------------------------------------------------
        # 3. Detect universal transaction structure
        # ---------------------------------------------------------

        structure = (
            self.structure_detector.detect(
                self.file_path,
                password=self.password,
            )
        )

        raw_transactions: List[
            Dict[str, Any]
        ] = []

        # ---------------------------------------------------------
        # Normal native-PDF path
        # ---------------------------------------------------------

        if structure.get("detected"):

            columns = structure.get(
                "columns",
                [],
            )

            if columns:

                structure_score = float(
                    structure.get(
                        "score",
                        0.0,
                    )
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

                # -----------------------------------------------------
                # 4. Try existing native row extraction first
                # -----------------------------------------------------

                raw_transactions = (
                    self._extract_universal_rows(
                        columns=columns,
                    )
                )

                if raw_transactions:

                    print(
                        f"Universal rows extracted: "
                        f"{len(raw_transactions)}"
                    )

        # ---------------------------------------------------------
        # OCR fallback
        #
        # Used when:
        #   1. structure detection fails, OR
        #   2. native row extraction returns no rows.
        #
        # This keeps normal text-based PDFs on the fast path.
        # ---------------------------------------------------------

        if not raw_transactions:

            print(
                "Native PDF extraction returned no "
                "transaction rows."
            )

            print(
                "Starting Tesseract OCR fallback..."
            )

            raw_transactions = (
                self.ocr_fallback.extract()
            )

            # Scanned statements may contain currency only inside the
            # page image, so transfer OCR-detected currency into metadata.
            ocr_currency = getattr(
                self.ocr_fallback,
                "detected_currency",
                None,
            )

            if ocr_currency and not getattr(
                metadata,
                "currency",
                None,
            ):
                metadata.currency = ocr_currency
                print(
                    f"OCR currency detected: {ocr_currency}"
                )

        # ---------------------------------------------------------
        # Final extraction check
        # ---------------------------------------------------------

        if not raw_transactions:

            raise ValueError(
                "No transaction rows could be extracted "
                "from the financial statement, including OCR."
            )

        print(
            f"Transactions extracted: "
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

            statement_year = (
                self._infer_statement_year(
                    raw_transactions
                )
            )

        # ---------------------------------------------------------
        # 6. Build canonical transactions
        # ---------------------------------------------------------

        transactions: List[
            Transaction
        ] = []

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

                    transactions.append(
                        transaction
                    )

            except Exception as exc:

                print(
                    f"WARNING: Could not normalize "
                    f"transaction {index}: {exc}"
                )

        # ---------------------------------------------------------
        # 7. Category intelligence
        # ---------------------------------------------------------

        if transactions:

            print(
                "Starting universal transaction "
                "categorization..."
            )

            transactions = (
                self.category_engine.classify_transactions(
                    transactions
                )
            )

            category_counts: Dict[
                str,
                int,
            ] = {}

            category_review_count = 0

            for transaction in transactions:

                category = (
                    transaction.category
                    or "Uncategorized"
                )

                category_counts[
                    category
                ] = (
                    category_counts.get(
                        category,
                        0,
                    )
                    + 1
                )

                if getattr(
                    transaction,
                    "requires_review",
                    False,
                ):

                    category_review_count += 1

            print(
                "Category classification completed."
            )

            print(
                f"  Categorized transactions: "
                f"{len(transactions)}"
            )

            print(
                f"  Category review transactions: "
                f"{category_review_count}"
            )

            print(
                "  Category distribution:"
            )

            for (
                category,
                count,
            ) in sorted(
                category_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            ):

                print(
                    f"    {category}: "
                    f"{count}"
                )

        # ---------------------------------------------------------
        # 8. Universal validation
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

        # ---------------------------------------------------------
        # 9. Final statistics
        # ---------------------------------------------------------

        final_category_review = sum(
            1
            for transaction
            in validated_transactions
            if getattr(
                transaction,
                "requires_review",
                False,
            )
        )

        validation_issue_count = sum(
            1
            for transaction
            in validated_transactions
            if getattr(
                transaction,
                "notes",
                None,
            )
        )

        print(
            "Final pipeline status:"
        )

        print(
            f"  Total transactions: "
            f"{len(validated_transactions)}"
        )

        print(
            f"  Category review: "
            f"{final_category_review}"
        )

        print(
            f"  Transactions with validation notes: "
            f"{validation_issue_count}"
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
        Extract rows from every PDF page using the same
        semantic structure.

        No bank name is inspected here.
        """

        transactions: List[
            Dict[str, Any]
        ] = []

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

                    row[
                        "page_number"
                    ] = page_number

                    row[
                        "source_file"
                    ] = self.file_path

                    transactions.append(
                        row
                    )

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

        # ---------------------------------------------------------
        # Date
        # ---------------------------------------------------------

        transaction_date = (
            self._parse_transaction_date(
                raw.get(
                    "transaction_date"
                ),
                statement_year,
            )
        )

        posting_date = (
            self._parse_transaction_date(
                raw.get(
                    "posting_date"
                ),
                statement_year,
            )
        )

        value_date = (
            self._parse_transaction_date(
                raw.get(
                    "value_date"
                ),
                statement_year,
            )
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

        (
            amount,
            direction_from_amount,
        ) = self._extract_amount_and_direction(
            raw
        )

        if amount is None:
            return None

        amount = abs(
            amount
        )

        # ---------------------------------------------------------
        # Description
        # ---------------------------------------------------------

        description = str(
            raw.get(
                "description"
            )
            or ""
        ).strip()

        transaction_id = (
            self._clean_optional_text(
                raw.get(
                    "transaction_id"
                )
            )
        )

        if (
            not description
            and not transaction_id
        ):

            return None

        if not description:

            description = (
                transaction_id
                or "Unknown transaction"
            )

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

        statement_type = (
            self._resolve_statement_type(
                metadata
            )
        )

        # ---------------------------------------------------------
        # Currency
        # ---------------------------------------------------------

        currency = (
            self._resolve_currency(
                metadata
            )
        )

        # ---------------------------------------------------------
        # Merchant
        # ---------------------------------------------------------

        (
            merchant,
            merchant_confidence,
        ) = self.merchant_normalizer.normalize(
            description
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

        statement_amount = (
            self._parse_decimal(
                raw.get(
                    "amount"
                )
            )
        )

        if statement_amount is None:

            statement_amount = (
                self._parse_decimal(
                    raw.get(
                        "debit"
                    )
                )
            )

        if statement_amount is None:

            statement_amount = (
                self._parse_decimal(
                    raw.get(
                        "credit"
                    )
                )
            )

        # ---------------------------------------------------------
        # Balance
        # ---------------------------------------------------------

        running_balance = (
            self._parse_decimal(
                raw.get(
                    "balance"
                )
            )
        )

        # ---------------------------------------------------------
        # Bank transaction ID
        # ---------------------------------------------------------

        bank_transaction_id = (
            self._clean_optional_text(
                raw.get(
                    "transaction_id"
                )
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
        # Extraction confidence
        # ---------------------------------------------------------

        extraction_confidence = (
            self._calculate_extraction_confidence(
                raw=raw,
                structure_confidence=None,
            )
        )

        return Transaction(

            transaction_id=(
                f"TXN-{index:06d}"
            ),

            transaction_date=(
                transaction_date
            ),

            posting_date=(
                posting_date
            ),

            description_raw=(
                description
            ),

            description_normalized=(
                merchant
            ),

            merchant=(
                merchant
            ),

            original_amount=(
                amount
            ),

            original_currency=(
                currency
            ),

            statement_amount=(
                abs(
                    statement_amount
                )
                if statement_amount
                is not None
                else amount
            ),

            statement_currency=(
                currency
            ),

            direction=(
                direction
            ),

            transaction_type=(
                transaction_type
            ),

            statement_type=(
                statement_type
            ),

            merchant_confidence=(
                merchant_confidence
            ),

            extraction_confidence=(
                extraction_confidence
            ),

            account_identifier=None,

            source_file=(
                self.file_path
            ),

            source_page=(
                raw.get(
                    "page_number"
                )
            ),

            bank_transaction_id=(
                bank_transaction_id
            ),

            running_balance=(
                abs(
                    running_balance
                )
                if running_balance
                is not None
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

        debit = self._parse_decimal(
            raw.get(
                "debit"
            )
        )

        credit = self._parse_decimal(
            raw.get(
                "credit"
            )
        )

        amount = self._parse_decimal(
            raw.get(
                "amount"
            )
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

        return (
            None,
            None,
        )

    # =============================================================
    # DIRECTION
    # =============================================================

    def _infer_direction(
        self,
        raw: Dict[str, Any],
        metadata: Any,
    ) -> TransactionDirection:

        description = str(
            raw.get(
                "description"
            )
            or ""
        ).upper()

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

                return (
                    TransactionDirection.CREDIT
                )

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

                return (
                    TransactionDirection.DEBIT
                )

        statement_type = (
            self._resolve_statement_type(
                metadata
            )
        )

        if (
            statement_type
            == StatementType.CREDIT_CARD
        ):

            return (
                TransactionDirection.DEBIT
            )

        return (
            TransactionDirection.DEBIT
        )

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

        text = " ".join(
            str(
                description or ""
            ).upper().split()
        )

        # ---------------------------------------------------------
        # Salary
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "SALARY",
                "PAYROLL",
                "WAGES",
                "MONTHLY SALARY",
                "SALARY CREDIT",
                "SALARY CR",
                "PAYROLL CREDIT",
                "EMPLOYEE SALARY",
            )
        ):

            return TransactionType.SALARY

        # ---------------------------------------------------------
        # Refund / reversal
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "REFUND",
                "REVERSAL",
                "REVERSED",
                "REVERTED",
                "CHARGEBACK",
                "RETURNED PAYMENT",
                "PAYMENT RETURN",
                "CREDIT REVERSAL",
            )
        ):

            return TransactionType.REFUND

        # ---------------------------------------------------------
        # Interest
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "INTEREST",
                "INT CREDIT",
                "INT CR",
                "INT PAID",
                "INTEREST CREDIT",
                "INTEREST DEBIT",
                "INTEREST PAYMENT",
            )
        ):

            return TransactionType.INTEREST

        # ---------------------------------------------------------
        # Fees
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "FEE",
                "FEES",
                "SERVICE CHARGE",
                "BANK CHARGE",
                "PROCESSING CHARGE",
                "PROCESSING FEE",
                "CONVENIENCE FEE",
                "TRANSACTION FEE",
                "ANNUAL FEE",
                "MAINTENANCE FEE",
                "ATM FEE",
                "CARD FEE",
                "LATE FEE",
            )
        ):

            return TransactionType.FEE

        if (
            text.startswith("CHARGE ")
            or text.endswith(" CHARGE")
        ):

            return TransactionType.FEE

        # ---------------------------------------------------------
        # Cash withdrawal
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "ATM WITHDRAWAL",
                "ATM CASH",
                "ATM WDL",
                "ATM WD",
                "CASH WITHDRAWAL",
                "CASH WITHDRAW",
                "CASH WDL",
                "CASH WD",
                "CASH DISPENSE",
            )
        ) or text.startswith("ATM"):

            return (
                TransactionType.CASH_WITHDRAWAL
            )

        # ---------------------------------------------------------
        # Loan
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "LOAN",
                "EMI",
                "INSTALLMENT",
                "INSTALMENT",
                "LOAN REPAYMENT",
                "LOAN PAYMENT",
                "FINANCE PAYMENT",
                "MORTGAGE PAYMENT",
            )
        ):

            return (
                TransactionType.LOAN_PAYMENT
            )

        # ---------------------------------------------------------
        # Tax
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "INCOME TAX",
                "PROPERTY TAX",
                "SALES TAX",
                "GST",
                "VAT",
                "TAX PAYMENT",
                "TAX PAID",
                "GOVERNMENT FEE",
                "GOVERNMENT PAYMENT",
                "MUNICIPAL TAX",
                "CUSTOMS DUTY",
                "DUTY PAYMENT",
            )
        ):

            return TransactionType.TAX

        if (
            text == "TAX"
            or text.startswith("TAX ")
            or text.startswith("GOVERNMENT ")
        ):

            return TransactionType.TAX

        # ---------------------------------------------------------
        # Bills
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "BILL PAYMENT",
                "BILL PAY",
                "UTILITY PAYMENT",
                "ELECTRICITY",
                "ELECTRIC BILL",
                "WATER BILL",
                "GAS BILL",
                "GAS PAYMENT",
                "INTERNET BILL",
                "INTERNET PAYMENT",
                "BROADBAND",
                "TELECOM BILL",
                "TELECOM PAYMENT",
                "MOBILE BILL",
                "MOBILE PAYMENT",
                "PHONE BILL",
                "PHONE PAYMENT",
                "UTILITY",
            )
        ):

            return (
                TransactionType.BILL_PAYMENT
            )

        # ---------------------------------------------------------
        # Investments
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "STOCK PURCHASE",
                "BUY STOCK",
                "SHARE PURCHASE",
                "BUY SHARES",
                "MUTUAL FUND",
                "MUTUAL FUNDS",
                "SYSTEMATIC INVESTMENT",
                "SIP",
                "BOND PURCHASE",
                "BROKERAGE",
                "BROKER",
                "SECURITIES",
                "INVESTMENT",
                "INVESTMENTS",
                "DEMAT",
            )
        ):

            return (
                TransactionType.INVESTMENT
            )

        # ---------------------------------------------------------
        # Card payments
        # ---------------------------------------------------------

        if any(
            term in text
            for term in (
                "CREDIT CARD PAYMENT",
                "CARD PAYMENT",
                "CARD BILL PAYMENT",
                "CREDIT CARD BILL",
                "CARD BILL",
                "CARD SETTLEMENT",
                "PAYMENT RECEIVED",
                "PAYMENT RECEIVED FROM",
            )
        ):

            return (
                TransactionType.PAYMENT
            )

        # ---------------------------------------------------------
        # Explicit transfers
        # ---------------------------------------------------------

        explicit_transfer_terms = (
            "TRANSFER",
            "TRANSFER TO",
            "TRANSFER FROM",
            "BANK TRANSFER",
            "INTERNAL TRANSFER",
            "SELF TRANSFER",
            "OWN ACCOUNT",
            "OWN A/C",
            "BETWEEN ACCOUNTS",
            "BENEFICIARY",
            "P2P TRANSFER",
            "PERSON TO PERSON",
            "ACCOUNT TRANSFER",
            "TRF TO",
            "TRF FROM",
            "TRANSFER CR",
            "TRANSFER DR",
            "WIRE TRANSFER",
            "WIRE PAYMENT",
            "SWIFT TRANSFER",
            "SEPA TRANSFER",
            "ACH TRANSFER",
            "DIRECT DEBIT",
        )

        if any(
            term in text
            for term in explicit_transfer_terms
        ):

            return (
                TransactionType.TRANSFER
            )

        tokens = set(
            text
            .replace("/", " ")
            .replace("-", " ")
            .split()
        )

        if tokens.intersection(
            {
                "TRF",
                "XFER",
                "TRANSFER",
            }
        ):

            return (
                TransactionType.TRANSFER
            )

        # ---------------------------------------------------------
        # Payment rail context
        # ---------------------------------------------------------

        rails = {
            "UPI",
            "IMPS",
            "NEFT",
            "RTGS",
            "ACH",
            "SEPA",
            "SWIFT",
        }

        has_rail = bool(
            tokens.intersection(
                rails
            )
        )

        if (
            has_rail
            and direction
            == TransactionDirection.CREDIT
        ):

            return (
                TransactionType.TRANSFER
            )

        # ---------------------------------------------------------
        # Generic fallback
        # ---------------------------------------------------------

        if (
            direction
            == TransactionDirection.CREDIT
        ):

            return TransactionType.OTHER

        return TransactionType.PURCHASE

    # =============================================================
    # TRANSACTION CHANNEL
    # =============================================================

    def _infer_transaction_channel(
        self,
        description: str,
    ) -> Optional[str]:

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

        for (
            marker,
            channel,
        ) in channel_patterns:

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

        text = str(
            value
        ).strip()

        if not text:
            return None

        try:

            result = (
                self.normalizer.parse_date(
                    text,
                    year=statement_year,
                )
            )

            if isinstance(
                result,
                date,
            ):

                return result

            return result

        except Exception:

            return None

    # =============================================================
    # STATEMENT YEAR
    # =============================================================

    def _infer_statement_year(
        self,
        rows: List[
            Dict[str, Any]
        ],
    ) -> Optional[int]:

        for row in rows:

            for field in (
                "transaction_date",
                "posting_date",
                "value_date",
            ):

                value = row.get(
                    field
                )

                if not value:
                    continue

                text = str(
                    value
                )

                parts = text.split(
                    "-"
                )

                if (
                    len(parts) == 3
                    and len(parts[0]) == 4
                    and parts[0].isdigit()
                ):

                    return int(
                        parts[0]
                    )

                if (
                    len(parts) == 3
                    and parts[-1].isdigit()
                    and len(parts[-1]) == 4
                ):

                    return int(
                        parts[-1]
                    )

                parts = text.split(
                    "/"
                )

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

        text = str(
            value
        ).strip()

        if not text:
            return None

        negative = False

        if (
            text.startswith("(")
            and text.endswith(")")
        ):

            negative = True

            text = text[
                1:-1
            ]

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

        upper = text.upper()

        if upper.endswith(
            "CR"
        ):

            text = text[
                :-2
            ].strip()

        elif upper.endswith(
            "DR"
        ):

            negative = True

            text = text[
                :-2
            ].strip()

        try:

            number = Decimal(
                text
            )

            if negative:

                number = -abs(
                    number
                )

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

        text = str(
            value
        ).strip()

        return (
            text
            or None
        )

    # =============================================================
    # EXTRACTION CONFIDENCE
    # =============================================================

    def _calculate_extraction_confidence(
        self,
        raw: Dict[str, Any],
        structure_confidence: Optional[float],
    ) -> float:

        score = 0.50

        if raw.get(
            "transaction_date"
        ):

            score += 0.10

        if raw.get(
            "posting_date"
        ):

            score += 0.05

        if raw.get(
            "value_date"
        ):

            score += 0.05

        if raw.get(
            "description"
        ):

            score += 0.10

        if raw.get(
            "amount"
        ):

            score += 0.10

        if (
            raw.get("debit")
            or raw.get("credit")
        ):

            score += 0.10

        if raw.get(
            "transaction_id"
        ):

            score += 0.05

        if raw.get(
            "balance"
        ):

            score += 0.05

        if structure_confidence is not None:

            score = (
                score * 0.70
                + float(
                    structure_confidence
                ) * 0.30
            )

        return min(
            round(
                score,
                4,
            ),
            1.0,
        )

    # =============================================================
    # VALIDATION
    # =============================================================

    def _validate_transactions(
        self,
        transactions: List[
            Transaction
        ],
    ) -> List[
        Transaction
    ]:

        validated_transactions: List[
            Transaction
        ] = []

        seen_transaction_keys: Set[
            str
        ] = set()

        previous_transaction: Optional[
            Transaction
        ] = None

        for transaction in transactions:

            # Save the category-review state BEFORE validation.
            #
            # Validation should not destroy category intelligence.

            category_requires_review = (
                getattr(
                    transaction,
                    "requires_review",
                    False,
                )
            )

            result = (
                self.validator.validate(
                    transaction=transaction,
                    previous_transaction=(
                        previous_transaction
                    ),
                    seen_transaction_keys=(
                        seen_transaction_keys
                    ),
                )
            )

            # -----------------------------------------------------
            # Validation confidence
            # -----------------------------------------------------

            transaction.extraction_confidence = (
                result.confidence
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
            # CATEGORY REVIEW
            #
            # IMPORTANT:
            #
            # The validator's requires_review flag is NOT merged
            # into the category review flag.
            #
            # This keeps:
            #
            #   category confidence
            #
            # separate from:
            #
            #   extraction / validation confidence.
            # -----------------------------------------------------

            transaction.requires_review = (
                category_requires_review
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

            previous_transaction = (
                transaction
            )

        return validated_transactions