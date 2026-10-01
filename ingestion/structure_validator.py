"""
Universal financial statement structure validator.

Validates candidate column mappings by inspecting the actual
transaction rows underneath the detected headers.

This module is bank-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional


@dataclass
class ColumnValidation:
    """Validation result for one semantic column."""

    semantic_type: str
    header: str
    score: float
    samples_checked: int
    valid_samples: int
    evidence: List[str] = field(default_factory=list)


@dataclass
class StructureValidationResult:
    """Overall structure validation result."""

    confidence: float
    columns: Dict[str, ColumnValidation]
    valid: bool
    issues: List[str] = field(default_factory=list)


class StructureValidator:
    """
    Validate detected financial-statement columns against actual rows.

    Supported semantic column types:

    - transaction_date
    - posting_date
    - value_date
    - description
    - amount
    - debit
    - credit
    - balance
    - transaction_id
    """

    DATE_TYPES = {
        "transaction_date",
        "posting_date",
        "value_date",
    }

    NUMERIC_TYPES = {
        "amount",
        "debit",
        "credit",
        "balance",
    }

    TEXT_TYPES = {
        "description",
        "transaction_id",
    }

    DATE_PATTERNS = [
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$",
        r"^\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}$",
        r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4}$",
        r"^[A-Za-z]{3}\s+\d{1,2},?\s+\d{2,4}$",
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",
    ]

    def validate(
        self,
        column_candidates: Dict[str, str],
        rows: List[Dict[str, Any]],
    ) -> StructureValidationResult:
        """
        Validate candidate semantic mappings.

        Example:

            column_candidates = {
                "Date": "transaction_date",
                "Description": "description",
                "Amount": "amount",
                "Balance": "balance",
            }

            rows = [
                {
                    "Date": "31/08/2026",
                    "Description": "Amazon",
                    "Amount": "450.00",
                    "Balance": "5500.00",
                }
            ]
        """

        results: Dict[str, ColumnValidation] = {}
        issues: List[str] = []

        if not column_candidates:
            return StructureValidationResult(
                confidence=0.0,
                columns={},
                valid=False,
                issues=["No column candidates supplied."],
            )

        if not rows:
            return StructureValidationResult(
                confidence=0.0,
                columns={},
                valid=False,
                issues=["No transaction rows available for validation."],
            )

        for header, semantic_type in column_candidates.items():
            validation = self._validate_column(
                header=header,
                semantic_type=semantic_type,
                rows=rows,
            )

            results[semantic_type] = validation

            if validation.score < 0.50:
                issues.append(
                    f"{header} does not strongly match "
                    f"{semantic_type}."
                )

        confidence = self._overall_confidence(results)

        valid = (
            confidence >= 0.60
            and len(results) > 0
        )

        return StructureValidationResult(
            confidence=confidence,
            columns=results,
            valid=valid,
            issues=issues,
        )

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------

    def _validate_column(
        self,
        header: str,
        semantic_type: str,
        rows: List[Dict[str, Any]],
    ) -> ColumnValidation:

        samples = []

        for row in rows:
            if header not in row:
                continue

            value = row.get(header)

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            samples.append(value)

        # Limit validation to useful samples.
        samples = samples[:25]

        if not samples:
            return ColumnValidation(
                semantic_type=semantic_type,
                header=header,
                score=0.0,
                samples_checked=0,
                valid_samples=0,
                evidence=["No non-empty samples found."],
            )

        if semantic_type in self.DATE_TYPES:
            return self._validate_dates(
                header,
                semantic_type,
                samples,
            )

        if semantic_type in self.NUMERIC_TYPES:
            return self._validate_numeric(
                header,
                semantic_type,
                samples,
            )

        if semantic_type in self.TEXT_TYPES:
            return self._validate_text(
                header,
                semantic_type,
                samples,
            )

        return ColumnValidation(
            semantic_type=semantic_type,
            header=header,
            score=0.0,
            samples_checked=len(samples),
            valid_samples=0,
            evidence=[
                f"Unknown semantic type: {semantic_type}"
            ],
        )

    # ------------------------------------------------------------------
    # Date validation
    # ------------------------------------------------------------------

    def _validate_dates(
        self,
        header: str,
        semantic_type: str,
        samples: List[str],
    ) -> ColumnValidation:

        valid = 0
        evidence = []

        for value in samples:
            if self._looks_like_date(value):
                valid += 1

        score = valid / len(samples)

        if score >= 0.80:
            evidence.append(
                f"{valid}/{len(samples)} samples look like dates."
            )
        else:
            evidence.append(
                f"Only {valid}/{len(samples)} samples look like dates."
            )

        return ColumnValidation(
            semantic_type=semantic_type,
            header=header,
            score=score,
            samples_checked=len(samples),
            valid_samples=valid,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------

    def _validate_numeric(
        self,
        header: str,
        semantic_type: str,
        samples: List[str],
    ) -> ColumnValidation:

        valid = 0
        evidence = []

        for value in samples:
            if self._looks_like_number(value):
                valid += 1

        score = valid / len(samples)

        evidence.append(
            f"{valid}/{len(samples)} samples look numeric."
        )

        return ColumnValidation(
            semantic_type=semantic_type,
            header=header,
            score=score,
            samples_checked=len(samples),
            valid_samples=valid,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Text validation
    # ------------------------------------------------------------------

    def _validate_text(
        self,
        header: str,
        semantic_type: str,
        samples: List[str],
    ) -> ColumnValidation:

        valid = 0
        evidence = []

        for value in samples:
            if self._looks_like_text(value):
                valid += 1

        score = valid / len(samples)

        evidence.append(
            f"{valid}/{len(samples)} samples look like text."
        )

        return ColumnValidation(
            semantic_type=semantic_type,
            header=header,
            score=score,
            samples_checked=len(samples),
            valid_samples=valid,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Primitive checks
    # ------------------------------------------------------------------

    def _looks_like_date(self, value: str) -> bool:

        value = value.strip()

        for pattern in self.DATE_PATTERNS:
            if re.match(pattern, value, re.IGNORECASE):
                return True

        formats = [
            "%d-%b-%Y",
            "%d-%b-%y",
            "%d/%m/%Y",
            "%d/%m/%y",
            "%d-%m-%Y",
            "%d-%m-%y",
            "%Y-%m-%d",
            "%d %b %Y",
            "%b %d %Y",
        ]

        for fmt in formats:
            try:
                datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue

        return False

    def _looks_like_number(self, value: str) -> bool:

        value = value.strip()

        if not value:
            return False

        # Remove common financial formatting.
        cleaned = (
            value
            .replace(",", "")
            .replace(" ", "")
            .replace("₹", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .replace("AED", "")
            .replace("INR", "")
            .replace("USD", "")
            .replace("EUR", "")
            .replace("GBP", "")
        )

        # Handle CR / DR suffixes.
        cleaned = re.sub(
            r"(CR|DR)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Handle parentheses.
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1]

        try:
            Decimal(cleaned)
            return True
        except (InvalidOperation, ValueError):
            return False

    def _looks_like_text(self, value: str) -> bool:

        value = value.strip()

        if not value:
            return False

        # Pure numbers are unlikely to be descriptions.
        if self._looks_like_number(value):
            return False

        # Dates are unlikely to be descriptions.
        if self._looks_like_date(value):
            return False

        return True

    # ------------------------------------------------------------------
    # Overall confidence
    # ------------------------------------------------------------------

    def _overall_confidence(
        self,
        results: Dict[str, ColumnValidation],
    ) -> float:

        if not results:
            return 0.0

        scores = [
            result.score
            for result in results.values()
        ]

        return round(sum(scores) / len(scores), 4)