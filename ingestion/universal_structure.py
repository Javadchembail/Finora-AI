from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import re
import pdfplumber

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
    """
    Universal financial-statement structure detector.

    Designed for arbitrary:
        - banks
        - countries
        - currencies
        - statement layouts
        - column orders
        - header terminology

    Pipeline:

        PDF words
            ↓
        Header candidate detection
            ↓
        Header-band detection
            ↓
        Multi-word header reconstruction
            ↓
        Semantic mapping
            ↓
        Duplicate/conflict resolution
            ↓
        Transaction structure
    """

    WORD_ROW_TOLERANCE = 4.0

    MULTILINE_MAX_VERTICAL_GAP = 18.0

    HEADER_BAND_TOLERANCE = 14.0

    MIN_HORIZONTAL_OVERLAP_RATIO = 0.55

    HORIZONTAL_HEADER_GAP = 24.0

    CORE_DATE_TYPES = {
        "transaction_date",
        "posting_date",
        "value_date",
    }

    AMOUNT_TYPES = {
        "amount",
        "debit",
        "credit",
    }

    ARTIFACT_HEADERS = {
        "DR",
        "CR",
        "/CR",
        "/DR",
        "DR/CR",
        "CR/DR",
    }

    COMPOSITE_HEADERS = {
        "value date": "value_date",
        "posting date": "posting_date",
        "post date": "posting_date",
        "transaction date": "transaction_date",
        "tran date": "transaction_date",
        "txn date": "transaction_date",
        "trans date": "transaction_date",
        "entry date": "transaction_date",
        "booking date": "transaction_date",
        "effective date": "value_date",

        "transaction id": "transaction_id",
        "transaction no": "transaction_id",
        "transaction number": "transaction_id",
        "txn id": "transaction_id",
        "txn no": "transaction_id",
        "txn number": "transaction_id",
        "tran id": "transaction_id",
        "tran no": "transaction_id",
        "reference number": "transaction_id",
        "reference no": "transaction_id",
        "ref no": "transaction_id",
        "trace id": "transaction_id",
        "trace number": "transaction_id",
        "confirmation number": "transaction_id",

        "transaction type": "transaction_type",
        "tran type": "transaction_type",
        "txn type": "transaction_type",

        "transaction description": "description",
        "transaction details": "description",
        "account description": "description",
        "account details": "description",

        "transaction amount": "amount",

        "amount out": "debit",
        "money out": "debit",
        "payment out": "debit",
        "paid out": "debit",

        "amount in": "credit",
        "money in": "credit",
        "payment in": "credit",
        "paid in": "credit",

        "closing balance": "balance",
        "running balance": "balance",
        "available balance": "balance",
        "current balance": "balance",
        "account balance": "balance",
        "ledger balance": "balance",
        "book balance": "balance",
    }

    def __init__(self):
        self.mapper = SemanticHeaderMapper()

    # ================================================================
    # PUBLIC API
    # ================================================================

    def detect(
        self,
        file_path: str,
        password: Optional[str] = None,
    ) -> Dict:

        with pdfplumber.open(
            file_path,
            password=password,
        ) as pdf:

            page_results = []

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

                result = self._detect_page(
                    words,
                    page_number,
                )

                if result:
                    page_results.append(result)

            if not page_results:
                return {
                    "detected": False,
                    "page": None,
                    "columns": [],
                    "score": 0.0,
                }

            page_results.sort(
                key=lambda result: (
                    result["score"],
                    len(result["columns"]),
                    result.get(
                        "transaction_rows",
                        0,
                    ),
                ),
                reverse=True,
            )

            return page_results[0]

    # ================================================================
    # PAGE DETECTION
    # ================================================================

    def _detect_page(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:

        normalized_words = self._normalize_words(
            words
        )

        if not normalized_words:
            return None

        primary = self._detect_page_primary(
            normalized_words,
            page_number,
        )

        if primary is not None:
            return primary

        return self._detect_fragmented_table(
            normalized_words,
            page_number,
        )

    # ================================================================
    # PRIMARY DETECTOR
    # ================================================================

    def _detect_page_primary(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:

        semantic_words = self._map_individual_words(
            words
        )

        if not semantic_words:
            return None

        header_words = self._find_header_band(
            semantic_words
        )

        if not header_words:
            return None

        header_top = min(
            word["top"]
            for word in header_words
        )

        expanded_header_words = []

        for word in words:

            if abs(
                word["top"] - header_top
            ) <= self.HEADER_BAND_TOLERANCE:

                expanded_header_words.append(
                    word
                )

        if not expanded_header_words:
            expanded_header_words = header_words

        logical_headers = (
            self._reconstruct_headers(
                expanded_header_words
            )
        )

        candidates = []

        for item in logical_headers:

            header = self._clean_header(
                item["header"]
            )

            if not header:
                continue

            if self._is_artifact_header(
                header
            ):
                continue

            semantic_type, confidence = (
                self._map_header(
                    header
                )
            )

            if not semantic_type:
                continue

            candidates.append(
                DetectedColumn(
                    header=header,
                    semantic_type=semantic_type,
                    x0=item["x0"],
                    x1=item["x1"],
                    top=item["top"],
                    confidence=float(
                        confidence
                    ),
                )
            )

        if not candidates:
            return None

        region = (
            self._select_transaction_columns(
                candidates
            )
        )

        if not region:
            return None

        score = self._structure_score(
            region
        )

        return {
            "detected": True,
            "page": page_number,
            "columns": [
                {
                    "header": column.header,
                    "semantic_type": column.semantic_type,
                    "x0": round(
                        column.x0,
                        2,
                    ),
                    "x1": round(
                        column.x1,
                        2,
                    ),
                    "top": round(
                        column.top,
                        2,
                    ),
                    "confidence": round(
                        column.confidence,
                        4,
                    ),
                }
                for column in sorted(
                    region,
                    key=lambda column: column.x0,
                )
            ],
            "score": round(
                score,
                4,
            ),
        }

    # ================================================================
    # HEADER MAPPING
    # ================================================================

    def _map_header(
        self,
        header: str,
    ) -> Tuple[Optional[str], float]:

        normalized = self._normalize_header(
            header
        )

        if not normalized:
            return None, 0.0

        composite = self.COMPOSITE_HEADERS.get(
            normalized
        )

        if composite:
            return composite, 0.99

        semantic_type, confidence = (
            self.mapper.map_header_with_confidence(
                header
            )
        )

        if semantic_type:
            return (
                semantic_type,
                float(confidence),
            )

        alias = self._generic_header_alias(
            normalized
        )

        if alias:
            return alias, 0.90

        return None, 0.0

    def _generic_header_alias(
        self,
        normalized: str,
    ) -> Optional[str]:

        aliases = {

            # Description
            "particulars": "description",
            "narration": "description",
            "narrative": "description",
            "details": "description",
            "description": "description",
            "remarks": "description",
            "memo": "description",
            "merchant": "description",
            "payee": "description",
            "beneficiary": "description",

            # Debit
            "withdrawal": "debit",
            "withdrawals": "debit",
            "debit": "debit",
            "debits": "debit",

            # Credit
            "deposit": "credit",
            "deposits": "credit",
            "credit": "credit",
            "credits": "credit",

            # Balance
            "balance": "balance",

            # Dates
            "date": "transaction_date",
            "transaction date": "transaction_date",
            "tran date": "transaction_date",
            "txn date": "transaction_date",
            "posting date": "posting_date",
            "post date": "posting_date",
            "value date": "value_date",

            # Amount
            "amount": "amount",

            # Transaction ID
            "reference": "transaction_id",
            "reference no": "transaction_id",
            "reference number": "transaction_id",
            "transaction id": "transaction_id",
            "transaction no": "transaction_id",
            "transaction number": "transaction_id",
            "txn id": "transaction_id",
            "txn no": "transaction_id",
            "tran id": "transaction_id",

            # Transaction type
            "transaction type": "transaction_type",
            "tran type": "transaction_type",
            "txn type": "transaction_type",
        }

        return aliases.get(
            normalized
        )

    # ================================================================
    # HEADER BAND
    # ================================================================

    def _find_header_band(
        self,
        semantic_words: List[Dict],
    ) -> List[Dict]:

        if not semantic_words:
            return []

        clusters: List[List[Dict]] = []

        ordered = sorted(
            semantic_words,
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

        for word in ordered:

            placed = False

            for cluster in clusters:

                average_top = (
                    sum(
                        item["top"]
                        for item in cluster
                    )
                    / len(cluster)
                )

                if abs(
                    word["top"]
                    - average_top
                ) <= self.HEADER_BAND_TOLERANCE:

                    cluster.append(
                        word
                    )

                    placed = True
                    break

            if not placed:

                clusters.append(
                    [word]
                )

        if not clusters:
            return []

        scored = []

        for cluster in clusters:

            score = (
                self._header_band_score(
                    cluster
                )
            )

            scored.append(
                (
                    score,
                    len(cluster),
                    cluster,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return sorted(
            scored[0][2],
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

    def _header_band_score(
        self,
        words: List[Dict],
    ) -> float:

        semantics = {
            word["semantic_type"]
            for word in words
        }

        score = 0.0

        score += min(
            len(words) * 0.12,
            0.60,
        )

        if semantics.intersection(
            self.CORE_DATE_TYPES
        ):
            score += 0.40

        if "description" in semantics:
            score += 0.40

        if semantics.intersection(
            self.AMOUNT_TYPES
        ):
            score += 0.40

        if "balance" in semantics:
            score += 0.20

        if "transaction_id" in semantics:
            score += 0.10

        if words:

            average_confidence = (
                sum(
                    word["confidence"]
                    for word in words
                )
                / len(words)
            )

            score += (
                average_confidence
                * 0.20
            )

        return score

    # ================================================================
    # HEADER RECONSTRUCTION
    # ================================================================

    def _reconstruct_headers(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        if not words:
            return []

        words = sorted(
            words,
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

        horizontal_groups = (
            self._build_horizontal_header_groups(
                words
            )
        )

        result = []

        for group in horizontal_groups:

            group = sorted(
                group,
                key=lambda word: (
                    word["top"],
                    word["x0"],
                ),
            )

            texts = [
                str(
                    word["text"]
                ).strip()
                for word in group
                if str(
                    word["text"]
                ).strip()
            ]

            if not texts:
                continue

            combined = " ".join(
                texts
            )

            semantic_type, _ = (
                self._map_header(
                    combined
                )
            )

            if semantic_type:

                result.append(
                    {
                        "header": combined,
                        "x0": min(
                            word["x0"]
                            for word in group
                        ),
                        "x1": max(
                            word["x1"]
                            for word in group
                        ),
                        "top": min(
                            word["top"]
                            for word in group
                        ),
                        "bottom": max(
                            word["bottom"]
                            for word in group
                        ),
                    }
                )

                continue

            for word in group:

                text = str(
                    word["text"]
                ).strip()

                if not text:
                    continue

                if self._is_artifact_header(
                    text
                ):
                    continue

                result.append(
                    {
                        "header": text,
                        "x0": word["x0"],
                        "x1": word["x1"],
                        "top": word["top"],
                        "bottom": word["bottom"],
                    }
                )

        return result

    def _build_horizontal_header_groups(
        self,
        words: List[Dict],
    ) -> List[List[Dict]]:

        if not words:
            return []

        groups: List[List[Dict]] = []

        ordered = sorted(
            words,
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

        for word in ordered:

            placed = False

            for group in groups:

                group_top = min(
                    item["top"]
                    for item in group
                )

                group_bottom = max(
                    item["bottom"]
                    for item in group
                )

                vertical_overlap = (
                    min(
                        word["bottom"],
                        group_bottom,
                    )
                    - max(
                        word["top"],
                        group_top,
                    )
                )

                if vertical_overlap <= 0:
                    continue

                group_x1 = max(
                    item["x1"]
                    for item in group
                )

                gap = (
                    word["x0"]
                    - group_x1
                )

                if gap < -5:
                    continue

                if gap > self.HORIZONTAL_HEADER_GAP:
                    continue

                candidate_text = " ".join(
                    [
                        str(
                            item["text"]
                        ).strip()
                        for item in group
                    ]
                    + [
                        str(
                            word["text"]
                        ).strip()
                    ]
                )

                normalized = (
                    self._normalize_header(
                        candidate_text
                    )
                )

                if (
                    normalized
                    in self.COMPOSITE_HEADERS
                ):

                    group.append(
                        word
                    )

                    placed = True
                    break

                semantic_type, _ = (
                    self._map_header(
                        candidate_text
                    )
                )

                if semantic_type:

                    group.append(
                        word
                    )

                    placed = True
                    break

            if not placed:

                groups.append(
                    [word]
                )

        # ------------------------------------------------------------
        # Vertical reconstruction
        # ------------------------------------------------------------

        final_groups: List[
            List[Dict]
        ] = []

        used = set()

        for index, group in enumerate(
            groups
        ):

            if index in used:
                continue

            current = list(
                group
            )

            group_center = (
                min(
                    word["x0"]
                    for word in current
                )
                + max(
                    word["x1"]
                    for word in current
                )
            ) / 2.0

            group_top = min(
                word["top"]
                for word in current
            )

            best_index = None
            best_score = 0.0

            for other_index in range(
                index + 1,
                len(groups),
            ):

                if other_index in used:
                    continue

                other = groups[
                    other_index
                ]

                other_center = (
                    min(
                        word["x0"]
                        for word in other
                    )
                    + max(
                        word["x1"]
                        for word in other
                    )
                ) / 2.0

                other_top = min(
                    word["top"]
                    for word in other
                )

                vertical_gap = (
                    other_top
                    - group_top
                )

                if vertical_gap <= 0:
                    continue

                if (
                    vertical_gap
                    > self.MULTILINE_MAX_VERTICAL_GAP
                ):
                    continue

                center_distance = abs(
                    group_center
                    - other_center
                )

                if center_distance > 25:
                    continue

                combined_words = (
                    current
                    + other
                )

                combined_text = " ".join(
                    str(
                        word["text"]
                    ).strip()
                    for word in sorted(
                        combined_words,
                        key=lambda word: (
                            word["top"],
                            word["x0"],
                        ),
                    )
                )

                semantic_type, _ = (
                    self._map_header(
                        combined_text
                    )
                )

                if not semantic_type:
                    continue

                score = (
                    1.0
                    - min(
                        vertical_gap / 20.0,
                        1.0,
                    )
                )

                score += (
                    1.0
                    - min(
                        center_distance / 25.0,
                        1.0,
                    )
                )

                if score > best_score:

                    best_score = score

                    best_index = (
                        other_index
                    )

            if (
                best_index is not None
                and best_score >= 1.20
            ):

                current.extend(
                    groups[
                        best_index
                    ]
                )

                used.add(
                    best_index
                )

            final_groups.append(
                current
            )

        return final_groups

    # ================================================================
    # COLUMN SELECTION
    # ================================================================

    def _select_transaction_columns(
        self,
        candidates: List[DetectedColumn],
    ) -> List[DetectedColumn]:

        if not candidates:
            return []

        filtered = []

        for candidate in candidates:

            if self._is_artifact_header(
                candidate.header
            ):
                continue

            filtered.append(
                candidate
            )

        candidates = filtered

        selected: List[
            DetectedColumn
        ] = []

        for candidate in sorted(
            candidates,
            key=lambda column: (
                column.x0,
                column.top,
            ),
        ):

            duplicate_index = None

            candidate_center = (
                candidate.x0
                + candidate.x1
            ) / 2.0

            for index, existing in enumerate(
                selected
            ):

                existing_center = (
                    existing.x0
                    + existing.x1
                ) / 2.0

                if (
                    existing.semantic_type
                    != candidate.semantic_type
                ):
                    continue

                if abs(
                    existing_center
                    - candidate_center
                ) <= 10.0:

                    duplicate_index = index
                    break

            if duplicate_index is None:

                selected.append(
                    candidate
                )

            else:

                existing = selected[
                    duplicate_index
                ]

                if (
                    candidate.confidence
                    > existing.confidence
                ):

                    selected[
                        duplicate_index
                    ] = candidate

        final_columns = []

        date_roles = {
            "transaction_date",
            "posting_date",
            "value_date",
        }

        seen_roles = set()

        for column in sorted(
            selected,
            key=lambda item: item.x0,
        ):

            role = column.semantic_type

            if role in date_roles:

                final_columns.append(
                    column
                )

                continue

            if role in seen_roles:

                existing_index = None

                for index, existing in enumerate(
                    final_columns
                ):

                    if (
                        existing.semantic_type
                        == role
                    ):

                        existing_index = index
                        break

                if existing_index is not None:

                    existing = final_columns[
                        existing_index
                    ]

                    if (
                        column.confidence
                        > existing.confidence
                    ):

                        final_columns[
                            existing_index
                        ] = column

                continue

            seen_roles.add(
                role
            )

            final_columns.append(
                column
            )

        final_columns = (
            self._resolve_amount_date_conflicts(
                final_columns
            )
        )

        semantics = {
            column.semantic_type
            for column in final_columns
        }

        has_date = bool(
            semantics.intersection(
                self.CORE_DATE_TYPES
            )
        )

        has_description = (
            "description"
            in semantics
        )

        has_amount = bool(
            semantics.intersection(
                self.AMOUNT_TYPES
            )
        )

        if not has_date:
            return []

        if not (
            has_description
            or has_amount
        ):
            return []

        return sorted(
            final_columns,
            key=lambda column: column.x0,
        )

    def _resolve_amount_date_conflicts(
        self,
        columns: List[DetectedColumn],
    ) -> List[DetectedColumn]:

        result = list(
            columns
        )

        value_date_columns = [
            column
            for column in result
            if column.semantic_type
            == "value_date"
        ]

        if not value_date_columns:
            return result

        cleaned = []

        for column in result:

            if (
                column.semantic_type
                != "amount"
            ):

                cleaned.append(
                    column
                )

                continue

            amount_center = (
                column.x0
                + column.x1
            ) / 2.0

            conflict = False

            for value_date in (
                value_date_columns
            ):

                value_center = (
                    value_date.x0
                    + value_date.x1
                ) / 2.0

                if abs(
                    amount_center
                    - value_center
                ) <= 20:

                    conflict = True
                    break

            if not conflict:

                cleaned.append(
                    column
                )

        return cleaned

    # ================================================================
    # STRUCTURE SCORE
    # ================================================================

    def _structure_score(
        self,
        columns: List[DetectedColumn],
    ) -> float:

        if not columns:
            return 0.0

        semantics = {
            column.semantic_type
            for column in columns
        }

        score = 0.0

        if semantics.intersection(
            self.CORE_DATE_TYPES
        ):
            score += 0.25

        if "description" in semantics:
            score += 0.25

        if semantics.intersection(
            self.AMOUNT_TYPES
        ):
            score += 0.25

        if "balance" in semantics:
            score += 0.15

        if semantics.intersection(
            {
                "transaction_id",
                "transaction_type",
                "posting_date",
                "value_date",
            }
        ):
            score += 0.10

        average_confidence = (
            sum(
                column.confidence
                for column in columns
            )
            / len(columns)
        )

        score *= (
            0.80
            + (
                average_confidence
                * 0.20
            )
        )

        return min(
            score,
            1.0,
        )

    # ================================================================
    # FRAGMENTED TABLE FALLBACK
    # ================================================================

    def _detect_fragmented_table(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:

        rows = self._group_words_by_top(
            words
        )

        if len(rows) < 3:
            return None

        candidate_bands = []

        for start in range(
            len(rows)
        ):

            for end in range(
                start,
                min(
                    len(rows),
                    start + 4,
                ),
            ):

                band_words = []

                for index in range(
                    start,
                    end + 1,
                ):

                    band_words.extend(
                        rows[index]
                    )

                mapped = (
                    self._map_fragmented_header_words(
                        band_words
                    )
                )

                if not mapped:
                    continue

                columns = (
                    self._build_fragmented_columns(
                        mapped
                    )
                )

                if not columns:
                    continue

                semantics = {
                    column.semantic_type
                    for column in columns
                }

                if not semantics.intersection(
                    self.CORE_DATE_TYPES
                ):
                    continue

                if not (
                    semantics.intersection(
                        self.AMOUNT_TYPES
                    )
                    or
                    "description"
                    in semantics
                ):
                    continue

                candidate_bands.append(
                    (
                        start,
                        end,
                        columns,
                    )
                )

        if not candidate_bands:
            return None

        best = None

        for (
            start,
            end,
            columns,
        ) in candidate_bands:

            transaction_rows = (
                self._count_transaction_evidence(
                    rows,
                    end,
                    columns,
                )
            )

            if transaction_rows < 2:
                continue

            columns = (
                self._select_transaction_columns(
                    columns
                )
            )

            if not columns:
                continue

            score = (
                self._fragmented_structure_score(
                    columns,
                    transaction_rows,
                )
            )

            candidate = (
                score,
                transaction_rows,
                len(columns),
                start,
                end,
                columns,
            )

            if (
                best is None
                or candidate[:5]
                > best[:5]
            ):

                best = candidate

        if best is None:
            return None

        (
            score,
            transaction_rows,
            _,
            start,
            end,
            columns,
        ) = best

        return {
            "detected": True,
            "page": page_number,
            "columns": [
                {
                    "header": column.header,
                    "semantic_type": column.semantic_type,
                    "x0": round(
                        column.x0,
                        2,
                    ),
                    "x1": round(
                        column.x1,
                        2,
                    ),
                    "top": round(
                        column.top,
                        2,
                    ),
                    "confidence": round(
                        column.confidence,
                        4,
                    ),
                }
                for column in sorted(
                    columns,
                    key=lambda item: item.x0,
                )
            ],
            "score": round(
                score,
                4,
            ),
            "structure_score": round(
                self._structure_score(
                    columns
                ),
                4,
            ),
            "transaction_evidence": round(
                min(
                    transaction_rows / 10.0,
                    1.0,
                ),
                4,
            ),
            "transaction_rows": transaction_rows,
            "repeated_rows": transaction_rows,
            "header_top": round(
                min(
                    rows[index][0]["top"]
                    for index in range(
                        start,
                        end + 1,
                    )
                ),
                2,
            ),
        }

    # ================================================================
    # FRAGMENTED HELPERS
    # ================================================================

    def _map_fragmented_header_words(
        self,
        row: List[Dict],
    ) -> List[Dict]:

        mapped = []

        for word in row:

            text = str(
                word["text"]
            ).strip()

            if not text:
                continue

            if self._is_artifact_header(
                text
            ):
                continue

            semantic_type, confidence = (
                self._map_header(
                    text
                )
            )

            if not semantic_type:
                continue

            mapped.append(
                {
                    **word,
                    "semantic_type": semantic_type,
                    "confidence": float(
                        confidence
                    ),
                }
            )

        return mapped

    def _build_fragmented_columns(
        self,
        mapped_words: List[Dict],
    ) -> List[DetectedColumn]:

        if not mapped_words:
            return []

        ordered = sorted(
            mapped_words,
            key=lambda item: (
                item["x0"],
                item["top"],
            ),
        )

        groups: List[
            List[Dict]
        ] = []

        for word in ordered:

            best_group = None
            best_distance = None

            center = (
                word["x0"]
                + word["x1"]
            ) / 2.0

            for group in groups:

                group_x0 = min(
                    item["x0"]
                    for item in group
                )

                group_x1 = max(
                    item["x1"]
                    for item in group
                )

                group_center = (
                    group_x0
                    + group_x1
                ) / 2.0

                distance = abs(
                    center
                    - group_center
                )

                tolerance = max(
                    18.0,
                    (
                        group_x1
                        - group_x0
                    ) * 1.5,
                )

                if (
                    distance
                    <= tolerance
                ):

                    if (
                        best_distance
                        is None
                        or distance
                        < best_distance
                    ):

                        best_group = group
                        best_distance = distance

            if best_group is None:

                groups.append(
                    [word]
                )

            else:

                best_group.append(
                    word
                )

        columns = []

        for group in groups:

            group.sort(
                key=lambda item: (
                    item["top"],
                    item["x0"],
                )
            )

            texts = [
                str(
                    item["text"]
                ).strip()
                for item in group
            ]

            combined = " ".join(
                texts
            ).strip()

            semantic_type, confidence = (
                self._map_header(
                    combined
                )
            )

            if not semantic_type:

                ranked = sorted(
                    group,
                    key=lambda item:
                    item["confidence"],
                    reverse=True,
                )

                chosen = ranked[0]

                semantic_type = (
                    chosen[
                        "semantic_type"
                    ]
                )

                confidence = (
                    chosen[
                        "confidence"
                    ]
                )

                header = (
                    chosen[
                        "text"
                    ]
                )

            else:

                header = combined

            columns.append(
                DetectedColumn(
                    header=header,
                    semantic_type=semantic_type,
                    x0=min(
                        item["x0"]
                        for item in group
                    ),
                    x1=max(
                        item["x1"]
                        for item in group
                    ),
                    top=min(
                        item["top"]
                        for item in group
                    ),
                    confidence=float(
                        confidence
                        or 0.90
                    ),
                )
            )

        return (
            self._select_transaction_columns(
                columns
            )
        )

    # ================================================================
    # TRANSACTION EVIDENCE
    # ================================================================

    def _count_transaction_evidence(
        self,
        rows: List[List[Dict]],
        header_end: int,
        columns: List[DetectedColumn],
    ) -> int:

        if header_end >= (
            len(rows) - 1
        ):
            return 0

        count = 0

        for row in rows[
            header_end + 1:
        ]:

            joined = " ".join(
                str(
                    item["text"]
                )
                for item in row
            ).strip()

            if not joined:
                continue

            lower = joined.casefold()

            if any(
                term in lower
                for term in (
                    "opening balance",
                    "closing balance",
                    "balance forward",
                    "brought forward",
                    "carried forward",
                    "statement summary",
                    "grand total",
                    "total transactions",
                )
            ):
                continue

            has_date = (
                self._row_has_date(
                    row,
                    [],
                )
            )

            numeric_count = sum(
                1
                for item in row
                if self._looks_numeric(
                    item["text"]
                )
            )

            if (
                has_date
                and numeric_count >= 1
            ):

                count += 1

                if count >= 100:
                    break

        return count

    def _row_has_date(
        self,
        row: List[Dict],
        date_columns: List[DetectedColumn],
    ) -> bool:

        for item in row:

            text = str(
                item["text"]
            ).strip()

            if self._looks_like_date_token(
                text
            ):

                if not date_columns:
                    return True

                center = (
                    item["x0"]
                    + item["x1"]
                ) / 2.0

                nearest = min(
                    date_columns,
                    key=lambda column:
                    abs(
                        (
                            (
                                column.x0
                                + column.x1
                            )
                            / 2.0
                        )
                        - center
                    ),
                )

                nearest_center = (
                    nearest.x0
                    + nearest.x1
                ) / 2.0

                if abs(
                    nearest_center
                    - center
                ) <= 85:

                    return True

        return False

    # ================================================================
    # FRAGMENTED SCORE
    # ================================================================

    def _fragmented_structure_score(
        self,
        columns: List[DetectedColumn],
        transaction_rows: int,
    ) -> float:

        base = self._structure_score(
            columns
        )

        evidence = min(
            transaction_rows / 10.0,
            1.0,
        )

        return min(
            (
                base * 0.65
            )
            + (
                evidence * 0.35
            ),
            1.0,
        )

    # ================================================================
    # WORD NORMALIZATION
    # ================================================================

    def _normalize_words(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        result = []

        for word in words:

            text = str(
                word.get(
                    "text",
                    "",
                )
            ).strip()

            if not text:
                continue

            try:

                x0 = float(
                    word["x0"]
                )

                x1 = float(
                    word["x1"]
                )

                top = float(
                    word["top"]
                )

                bottom = float(
                    word.get(
                        "bottom",
                        top,
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

            result.append(
                {
                    "text": text,
                    "x0": x0,
                    "x1": x1,
                    "top": top,
                    "bottom": bottom,
                }
            )

        return result

    # ================================================================
    # INDIVIDUAL WORD MAPPING
    # ================================================================

    def _map_individual_words(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        result = []

        for word in words:

            text = word[
                "text"
            ]

            if self._is_artifact_header(
                text
            ):
                continue

            # "Tran" / "Txn" alone is ambiguous.
            if (
                self._normalize_header(
                    text
                )
                in {
                    "tran",
                    "txn",
                    "trans",
                }
            ):
                continue

            semantic_type, confidence = (
                self._map_header(
                    text
                )
            )

            if not semantic_type:
                continue

            result.append(
                {
                    **word,
                    "semantic_type": semantic_type,
                    "confidence": float(
                        confidence
                    ),
                }
            )

        return result

    # ================================================================
    # WORD GROUPING
    # ================================================================

    def _group_words_by_top(
        self,
        words: List[Dict],
    ) -> List[List[Dict]]:

        rows: List[
            List[Dict]
        ] = []

        for word in sorted(
            words,
            key=lambda item: (
                item["top"],
                item["x0"],
            ),
        ):

            placed = False

            for row in rows:

                average_top = (
                    sum(
                        item["top"]
                        for item in row
                    )
                    / len(row)
                )

                if abs(
                    word["top"]
                    - average_top
                ) <= self.WORD_ROW_TOLERANCE:

                    row.append(
                        word
                    )

                    placed = True
                    break

            if not placed:

                rows.append(
                    [word]
                )

        for row in rows:

            row.sort(
                key=lambda item:
                item["x0"]
            )

        return rows

    # ================================================================
    # DATE DETECTION
    # ================================================================

    def _looks_like_date_token(
        self,
        text: str,
    ) -> bool:

        value = str(
            text
        ).strip().upper()

        if not value:
            return False

        months = (
            "JAN|FEB|MAR|APR|MAY|JUN|"
            "JUL|AUG|SEP|OCT|NOV|DEC"
        )

        patterns = [
            rf"^\d{{1,2}}[-/]({months})[-/]\d{{2,4}}$",
            rf"^\d{{4}}[-/]({months})[-/]\\d{{1,2}}$",
            r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$",
            r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
        ]

        if any(
            re.match(
                pattern,
                value,
            )
            for pattern in patterns
        ):
            return True

        if re.fullmatch(
            r"\d{1,2}",
            value,
        ):

            number = int(
                value
            )

            return (
                1
                <= number
                <= 31
            )

        if (
            value[:3]
            in months.split("|")
            and len(value) <= 9
        ):

            return True

        if re.fullmatch(
            r"\d{4}",
            value,
        ):

            year = int(
                value
            )

            return (
                1900
                <= year
                <= 2100
            )

        return False

    # ================================================================
    # NUMERIC DETECTION
    # ================================================================

    def _looks_numeric(
        self,
        text: str,
    ) -> bool:

        value = str(
            text
        ).strip()

        value = value.replace(
            ",",
            "",
        )

        value = value.replace(
            "₹",
            "",
        ).replace(
            "$",
            "",
        ).replace(
            "€",
            "",
        ).replace(
            "£",
            "",
        )

        if (
            value.startswith("(")
            and value.endswith(")")
        ):

            value = value[
                1:-1
            ]

        value = value.strip()

        try:

            float(
                value
            )

            return True

        except (
            TypeError,
            ValueError,
        ):

            return False

    # ================================================================
    # HEADER HELPERS
    # ================================================================

    def _normalize_header(
        self,
        text: str,
    ) -> str:

        value = str(
            text
        ).strip().lower()

        value = value.replace(
            "&",
            " and ",
        )

        value = re.sub(
            r"[/_.:#\-]+",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    def _clean_header(
        self,
        text: str,
    ) -> str:

        value = str(
            text
        ).strip()

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # PDF extraction can attach a CR/DR marker to a neighboring
        # header.
        #
        # Examples:
        #
        #   Balance /CR -> Balance
        #   Balance /DR -> Balance
        #   Amount CR   -> Amount
        #   Amount DR   -> Amount
        #
        # These are formatting artifacts, not separate columns.
        # ------------------------------------------------------------

        value = re.sub(
            r"\s*/\s*(?:CR|DR)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"\s+(?:CR|DR)\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    def _is_artifact_header(
        self,
        text: str,
    ) -> bool:

        normalized = str(
            text
        ).strip().upper()

        if normalized in (
            self.ARTIFACT_HEADERS
        ):
            return True

        if normalized.startswith(
            "/"
        ) and normalized[
            1:
        ] in {
            "CR",
            "DR",
        }:

            return True

        return False