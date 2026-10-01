from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pdfplumber

from ingestion.header_grouping import HeaderGrouper
from ingestion.semantic_mapper import SemanticHeaderMapper


@dataclass
class DetectedColumn:
    header: str
    semantic_type: str
    x0: float
    x1: float
    top: float
    confidence: float = 1.0


class UniversalStructureDetector:

    def __init__(self):
        self.grouper = HeaderGrouper()
        self.mapper = SemanticHeaderMapper()

    def detect(
        self,
        file_path: str,
        password: Optional[str] = None,
    ) -> Dict:

        with pdfplumber.open(
            file_path,
            password=password,
        ) as pdf:

            best_result = None

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                words = page.extract_words(
                    use_text_flow=False,
                    keep_blank_chars=False,
                )

                result = self._detect_page(
                    words,
                    page_number,
                )

                if result is None:
                    continue

                if (
                    best_result is None
                    or result["score"] > best_result["score"]
                ):
                    best_result = result

            if best_result is None:
                return {
                    "detected": False,
                    "page": None,
                    "columns": [],
                    "score": 0.0,
                }

            return best_result

    # ---------------------------------------------------------
    # Page
    # ---------------------------------------------------------

    def _detect_page(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:

        if not words:
            return None

        groups = self.grouper.group(words)

        candidates = []

        for group in groups:

            semantic_type = self.mapper.map_header(
                group.text
            )

            if semantic_type is None:
                continue

            # Ignore isolated "DR" / "CR".
            #
            # These are ambiguous because they can represent
            # balance direction, transaction direction,
            # or part of another header.
            if group.text.strip().upper() in {
                "DR",
                "CR",
            }:
                continue

            candidates.append(
                DetectedColumn(
                    header=group.text,
                    semantic_type=semantic_type,
                    x0=group.x0,
                    x1=group.x1,
                    top=group.top,
                )
            )

        if not candidates:
            return None

        # -----------------------------------------------------
        # Find the transaction-header region.
        #
        # A real transaction table normally has multiple
        # semantic columns close together.
        # -----------------------------------------------------

        header_region = self._find_header_region(
            candidates
        )

        if not header_region:
            return None

        selected = self._select_columns(
            header_region
        )

        if not selected:
            return None

        score = self._structure_score(
            selected
        )

        return {
            "detected": True,
            "page": page_number,
            "columns": [
                {
                    "header": column.header,
                    "semantic_type": column.semantic_type,
                    "x0": round(column.x0, 2),
                    "x1": round(column.x1, 2),
                    "top": round(column.top, 2),
                    "confidence": column.confidence,
                }
                for column in selected
            ],
            "score": round(score, 4),
        }

    # ---------------------------------------------------------
    # Header region detection
    # ---------------------------------------------------------

    def _find_header_region(
        self,
        candidates: List[DetectedColumn],
    ) -> List[DetectedColumn]:

        candidates = sorted(
            candidates,
            key=lambda column: column.top
        )

        best_region = []

        # A transaction header normally contains at least
        # three meaningful fields located reasonably close
        # vertically.
        for candidate in candidates:

            region = [
                item
                for item in candidates
                if abs(
                    item.top - candidate.top
                ) <= 12
            ]

            if len(region) > len(best_region):
                best_region = region

        return best_region

    # ---------------------------------------------------------
    # Remove duplicate semantic candidates
    # ---------------------------------------------------------

    def _select_columns(
        self,
        candidates: List[DetectedColumn],
    ) -> List[DetectedColumn]:

        selected: Dict[str, DetectedColumn] = {}

        for candidate in candidates:

            semantic = candidate.semantic_type

            existing = selected.get(
                semantic
            )

            if existing is None:
                selected[semantic] = candidate
                continue

            # Prefer the candidate that is part of the
            # denser transaction-table header.
            #
            # For now, prefer the one with a more useful
            # horizontal position rather than arbitrary text
            # elsewhere on the page.

            if self._header_priority(candidate) > self._header_priority(existing):
                selected[semantic] = candidate

        result = list(
            selected.values()
        )

        result.sort(
            key=lambda column: column.x0
        )

        return result

    def _header_priority(
        self,
        column: DetectedColumn,
    ) -> float:

        score = 0.0

        text = column.header.lower().strip()

        # Strong transaction-table names.
        strong_names = {
            "date",
            "value date",
            "post date",
            "trxn. date",
            "transaction date",
            "description",
            "particulars",
            "amount",
            "withdrawals",
            "deposits",
            "balance",
            "tran id",
        }

        if text in strong_names:
            score += 10

        # Penalize known metadata fields.
        metadata_names = {
            "current balance",
            "effective available balance",
            "available balance",
            "opening balance",
            "closing balance",
        }

        if text in metadata_names:
            score -= 5

        return score

    # ---------------------------------------------------------
    # Structure confidence
    # ---------------------------------------------------------

    def _structure_score(
        self,
        columns: List[DetectedColumn],
    ) -> float:

        if not columns:
            return 0.0

        score = 0.0

        semantic_types = {
            column.semantic_type
            for column in columns
        }

        if "transaction_date" in semantic_types:
            score += 0.25

        if "description" in semantic_types:
            score += 0.25

        if (
            "amount" in semantic_types
            or "debit" in semantic_types
            or "credit" in semantic_types
        ):
            score += 0.25

        if "balance" in semantic_types:
            score += 0.15

        if (
            "transaction_id" in semantic_types
            or "posting_date" in semantic_types
            or "value_date" in semantic_types
        ):
            score += 0.10

        return min(score, 1.0)