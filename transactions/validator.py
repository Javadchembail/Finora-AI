from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Set

from transactions.models import Transaction


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"


@dataclass
class ValidationResult:
    is_valid: bool
    requires_review: bool
    confidence: float
    issues: List[ValidationIssue] = field(default_factory=list)


class UniversalTransactionValidator:
    """
    Bank-agnostic validation and confidence engine.

    This validator works on the canonical Transaction model
    and does not contain rules for any particular bank.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.70,
        review_confidence: float = 0.85,
    ):
        self.minimum_confidence = minimum_confidence
        self.review_confidence = review_confidence

    # =========================================================
    # PUBLIC API
    # =========================================================

    def validate(
        self,
        transaction: Transaction,
        previous_transaction: Optional[Transaction] = None,
        seen_transaction_keys: Optional[Set[str]] = None,
    ) -> ValidationResult:

        issues: List[ValidationIssue] = []

        self._validate_date(
            transaction,
            issues,
        )

        self._validate_amount(
            transaction,
            issues,
        )

        self._validate_currency(
            transaction,
            issues,
        )

        self._validate_description(
            transaction,
            issues,
        )

        self._validate_direction(
            transaction,
            issues,
        )

        self._validate_statement_amount(
            transaction,
            issues,
        )

        self._validate_balance(
            transaction,
            previous_transaction,
            issues,
        )

        self._validate_duplicate(
            transaction,
            seen_transaction_keys,
            issues,
        )

        confidence = self._calculate_confidence(
            transaction,
            issues,
        )

        has_error = any(
            issue.severity == "error"
            for issue in issues
        )

        requires_review = (
            has_error
            or confidence < self.review_confidence
            or any(
                issue.severity == "warning"
                for issue in issues
            )
        )

        return ValidationResult(
            is_valid=not has_error,
            requires_review=requires_review,
            confidence=confidence,
            issues=issues,
        )

    # =========================================================
    # DATE VALIDATION
    # =========================================================

    def _validate_date(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        transaction_date = getattr(
            transaction,
            "transaction_date",
            None,
        )

        posting_date = getattr(
            transaction,
            "posting_date",
            None,
        )

        if transaction_date is None:

            issues.append(
                ValidationIssue(
                    code="MISSING_TRANSACTION_DATE",
                    message="Transaction date is missing.",
                    severity="error",
                )
            )

        if (
            transaction_date is not None
            and posting_date is not None
            and posting_date < transaction_date
        ):

            issues.append(
                ValidationIssue(
                    code="POSTING_DATE_BEFORE_TRANSACTION_DATE",
                    message=(
                        "Posting date occurs before "
                        "transaction date."
                    ),
                    severity="warning",
                )
            )

    # =========================================================
    # AMOUNT VALIDATION
    # =========================================================

    def _validate_amount(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        value = getattr(
            transaction,
            "original_amount",
            None,
        )

        if value is None:

            issues.append(
                ValidationIssue(
                    code="MISSING_AMOUNT",
                    message="Transaction amount is missing.",
                    severity="error",
                )
            )

            return

        try:

            amount = Decimal(
                str(value)
            )

        except Exception:

            issues.append(
                ValidationIssue(
                    code="INVALID_AMOUNT",
                    message="Transaction amount is not numeric.",
                    severity="error",
                )
            )

            return

        if amount < 0:

            issues.append(
                ValidationIssue(
                    code="NEGATIVE_NORMALIZED_AMOUNT",
                    message=(
                        "Canonical transaction amount is negative."
                    ),
                    severity="warning",
                )
            )

        if amount == 0:

            issues.append(
                ValidationIssue(
                    code="ZERO_AMOUNT",
                    message="Transaction amount is zero.",
                    severity="warning",
                )
            )

    # =========================================================
    # CURRENCY VALIDATION
    # =========================================================

    def _validate_currency(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        value = getattr(
            transaction,
            "original_currency",
            None,
        )

        if not value:

            issues.append(
                ValidationIssue(
                    code="MISSING_CURRENCY",
                    message="Transaction currency is missing.",
                    severity="warning",
                )
            )

            return

        currency = str(
            value
        ).strip().upper()

        if currency in {
            "UNKNOWN",
            "N/A",
            "NA",
            "NULL",
            "NONE",
        }:

            issues.append(
                ValidationIssue(
                    code="UNKNOWN_CURRENCY",
                    message=(
                        "Transaction currency could not "
                        "be identified."
                    ),
                    severity="warning",
                )
            )

    # =========================================================
    # DESCRIPTION VALIDATION
    # =========================================================

    def _validate_description(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        value = getattr(
            transaction,
            "description_raw",
            "",
        )

        description = str(
            value or ""
        ).strip()

        if not description:

            issues.append(
                ValidationIssue(
                    code="MISSING_DESCRIPTION",
                    message="Transaction description is missing.",
                    severity="error",
                )
            )

            return

        if len(description) < 3:

            issues.append(
                ValidationIssue(
                    code="SHORT_DESCRIPTION",
                    message=(
                        "Transaction description is "
                        "unusually short."
                    ),
                    severity="warning",
                )
            )

    # =========================================================
    # DIRECTION VALIDATION
    # =========================================================

    def _validate_direction(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        direction = getattr(
            transaction,
            "direction",
            None,
        )

        if direction is None:

            issues.append(
                ValidationIssue(
                    code="MISSING_DIRECTION",
                    message=(
                        "Transaction debit/credit direction "
                        "is missing."
                    ),
                    severity="error",
                )
            )

    # =========================================================
    # STATEMENT AMOUNT VALIDATION
    # =========================================================

    def _validate_statement_amount(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> None:

        statement_amount = getattr(
            transaction,
            "statement_amount",
            None,
        )

        original_amount = getattr(
            transaction,
            "original_amount",
            None,
        )

        if statement_amount is None:
            return

        if original_amount is None:
            return

        try:

            statement = abs(
                Decimal(
                    str(statement_amount)
                )
            )

            original = abs(
                Decimal(
                    str(original_amount)
                )
            )

        except Exception:

            issues.append(
                ValidationIssue(
                    code="INVALID_STATEMENT_AMOUNT",
                    message=(
                        "Statement amount could not be "
                        "interpreted numerically."
                    ),
                    severity="warning",
                )
            )

            return

        if statement == original:
            return

        exchange_rate = getattr(
            transaction,
            "exchange_rate",
            None,
        )

        if exchange_rate is not None:
            return

        issues.append(
            ValidationIssue(
                code="AMOUNT_MISMATCH",
                message=(
                    "Original amount and statement amount "
                    "differ without an exchange rate."
                ),
                severity="warning",
            )
        )

    # =========================================================
    # BALANCE VALIDATION
    # =========================================================

    def _validate_balance(
        self,
        transaction: Transaction,
        previous_transaction: Optional[Transaction],
        issues: List[ValidationIssue],
    ) -> None:
        """
        Validate running balance when the canonical model
        provides balance information.

        Balance is optional because many financial statements
        do not contain a running balance.
        """

        current_balance = getattr(
            transaction,
            "running_balance",
            None,
        )

        if current_balance is None:
            return

        if previous_transaction is None:
            return

        previous_balance = getattr(
            previous_transaction,
            "running_balance",
            None,
        )

        if previous_balance is None:
            return

        amount = getattr(
            transaction,
            "original_amount",
            None,
        )

        if amount is None:
            return

        try:

            current = Decimal(
                str(current_balance)
            )

            previous = Decimal(
                str(previous_balance)
            )

            transaction_amount = abs(
                Decimal(
                    str(amount)
                )
            )

        except Exception:

            issues.append(
                ValidationIssue(
                    code="INVALID_BALANCE_DATA",
                    message=(
                        "Running balance could not be "
                        "interpreted as a numeric value."
                    ),
                    severity="warning",
                )
            )

            return

        direction = getattr(
            transaction,
            "direction",
            None,
        )

        if direction is None:
            return

        direction_value = getattr(
            direction,
            "value",
            str(direction),
        )

        if direction_value == "debit":

            expected_balance = (
                previous - transaction_amount
            )

        elif direction_value == "credit":

            expected_balance = (
                previous + transaction_amount
            )

        else:

            return

        tolerance = Decimal(
            "0.01"
        )

        difference = abs(
            current - expected_balance
        )

        if difference > tolerance:

            issues.append(
                ValidationIssue(
                    code="BALANCE_MISMATCH",
                    message=(
                        "Running balance does not match "
                        "the transaction amount and direction."
                    ),
                    severity="warning",
                )
            )

    # =========================================================
    # DUPLICATE VALIDATION
    # =========================================================

    def _validate_duplicate(
        self,
        transaction: Transaction,
        seen_transaction_keys: Optional[Set[str]],
        issues: List[ValidationIssue],
    ) -> None:

        if seen_transaction_keys is None:
            return

        key = self._transaction_key(
            transaction
        )

        if key in seen_transaction_keys:

            issues.append(
                ValidationIssue(
                    code="POSSIBLE_DUPLICATE",
                    message=(
                        "A transaction with the same date, "
                        "amount, direction and description "
                        "already exists."
                    ),
                    severity="warning",
                )
            )

    # =========================================================
    # CONFIDENCE CALCULATION
    # =========================================================

    def _calculate_confidence(
        self,
        transaction: Transaction,
        issues: List[ValidationIssue],
    ) -> float:

        raw_confidence = getattr(
            transaction,
            "extraction_confidence",
            None,
        )

        try:

            confidence = (
                float(raw_confidence)
                if raw_confidence is not None
                else 0.50
            )

        except Exception:

            confidence = 0.50

        merchant_confidence = getattr(
            transaction,
            "merchant_confidence",
            None,
        )

        if merchant_confidence is not None:

            try:

                merchant_value = float(
                    merchant_confidence
                )

                confidence = (
                    confidence * 0.80
                    + merchant_value * 0.20
                )

            except Exception:
                pass

        for issue in issues:

            if issue.severity == "error":

                confidence -= 0.20

            elif issue.severity == "warning":

                confidence -= 0.05

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return round(
            confidence,
            4,
        )

    # =========================================================
    # TRANSACTION KEY
    # =========================================================

    @staticmethod
    def _transaction_key(
        transaction: Transaction,
    ) -> str:

        transaction_date = getattr(
            transaction,
            "transaction_date",
            None,
        )

        date_value = (
            transaction_date.isoformat()
            if transaction_date
            else ""
        )

        amount_value = str(
            getattr(
                transaction,
                "original_amount",
                "",
            )
        )

        direction = getattr(
            transaction,
            "direction",
            None,
        )

        direction_value = getattr(
            direction,
            "value",
            str(direction),
        ) if direction is not None else ""

        description = (
            getattr(
                transaction,
                "description_normalized",
                None,
            )
            or getattr(
                transaction,
                "description_raw",
                None,
            )
            or ""
        )

        description_value = str(
            description
        ).strip().upper()

        return "|".join(
            [
                date_value,
                amount_value,
                direction_value,
                description_value,
            ]
        )