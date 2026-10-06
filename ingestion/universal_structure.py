from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    Architecture:

        PDF words
            ↓
        Candidate semantic words
            ↓
        Header-band detection
            ↓
        Multi-line header reconstruction
            ↓
        Semantic mapping
            ↓
        Transaction structure
    """

    WORD_ROW_TOLERANCE = 4.0

    # Maximum distance between stacked words belonging to the
    # same logical header.
    MULTILINE_MAX_VERTICAL_GAP = 18.0

    # Header-band tolerance around the strongest header row.
    HEADER_BAND_TOLERANCE = 14.0

    # Horizontal overlap required for vertically stacked words.
    MIN_HORIZONTAL_OVERLAP_RATIO = 0.55

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

    def __init__(self):
        self.mapper = SemanticHeaderMapper()
        # Cache semantic header lookups because statement PDFs repeat the
        # same header vocabulary across pages and rows. Both successful and
        # unsuccessful lookups are cached.
        self._semantic_cache = {}

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
        # Normalize once per page. The previous implementation normalized
        # the same word list again inside the primary detector.
        normalized_words = self._normalize_words(words)
        if not normalized_words:
            return None

        primary = self._detect_page_primary(
            normalized_words,
            page_number,
        )
        if primary is not None:
            return primary

        # Fragmented detection is deliberately expensive. Do not run it on
        # pages that contain no plausible financial-header vocabulary.
        if not self._has_financial_header_signal(normalized_words):
            return None

        return self._detect_fragmented_table(
            normalized_words,
            page_number,
        )

    def _detect_page_primary(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:

        normalized_words = words

        if not normalized_words:
            return None

        # ------------------------------------------------------------
        # STEP 1
        # Find candidate semantic words throughout the page.
        # ------------------------------------------------------------

        semantic_words = (
            self._map_individual_words(
                normalized_words
            )
        )

        if not semantic_words:
            return None

        # ------------------------------------------------------------
        # STEP 2
        # Identify the most likely transaction-header band.
        #
        # IMPORTANT:
        # Multiline reconstruction happens ONLY inside this band.
        #
        # This prevents a header word from being accidentally combined
        # with a transaction-row word.
        # ------------------------------------------------------------

        header_words = (
            self._find_header_band(
                semantic_words
            )
        )

        if not header_words:
            return None

        # ------------------------------------------------------------
        # STEP 3
        # Reconstruct vertically stacked headers.
        # ------------------------------------------------------------

        logical_headers = (
            self._reconstruct_headers(
                header_words
            )
        )

        # ------------------------------------------------------------
        # STEP 4
        # Map reconstructed headers.
        # ------------------------------------------------------------

        candidates = []

        for item in logical_headers:

            header = item["header"].strip()

            if not header:
                continue

            if header.upper() in {
                "DR",
                "CR",
            }:
                continue

            semantic_type, confidence = (
                self._map_header_cached(
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

        # ------------------------------------------------------------
        # STEP 5
        # Select strongest transaction structure.
        # ------------------------------------------------------------

        region = (
            self._select_transaction_columns(
                candidates
            )
        )

        if not region:
            return None

        # ------------------------------------------------------------
        # STEP 6
        # Final score.
        # ------------------------------------------------------------

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
    # FRAGMENTED TABLE FALLBACK
    # ================================================================

    def _detect_fragmented_table(
        self,
        words: List[Dict],
        page_number: int,
    ) -> Optional[Dict]:
        """Generic fallback for physically fragmented financial table headers.

        The fallback deliberately uses document geometry plus generic financial
        vocabulary. It never checks a bank name or a bank-specific layout.
        """
        rows = self._group_words_by_top(words)
        if len(rows) < 3:
            return None

        # Build candidate header bands from nearby physical rows. A PDF can
        # place a header on one line, two lines, or several aligned lines.
        candidate_bands = []
        for start in range(len(rows)):
            for end in range(start, min(len(rows), start + 4)):
                band_words = []
                for idx in range(start, end + 1):
                    band_words.extend(rows[idx])
                mapped = self._map_fragmented_header_words(band_words)
                if not mapped:
                    continue
                columns = self._build_fragmented_columns(mapped)
                if not columns:
                    continue
                semantics = {c.semantic_type for c in columns}
                if not semantics.intersection(self.CORE_DATE_TYPES):
                    continue
                if not (semantics.intersection(self.AMOUNT_TYPES) or "description" in semantics):
                    continue
                candidate_bands.append((start, end, columns))

        if not candidate_bands:
            return None

        best = None
        for start, end, columns in candidate_bands:
            transaction_rows = self._count_transaction_evidence(rows, end, columns)
            if transaction_rows < 2:
                continue

            score = self._fragmented_structure_score(columns, transaction_rows)
            candidate = (score, transaction_rows, len(columns), start, end, columns)
            if best is None or candidate[:5] > best[:5]:
                best = candidate

        if best is None:
            return None

        score, transaction_rows, _, start, end, columns = best
        return {
            "detected": True,
            "page": page_number,
            "columns": [
                {
                    "header": c.header,
                    "semantic_type": c.semantic_type,
                    "x0": round(c.x0, 2),
                    "x1": round(c.x1, 2),
                    "top": round(c.top, 2),
                    "confidence": round(c.confidence, 4),
                }
                for c in sorted(columns, key=lambda item: item.x0)
            ],
            "score": round(score, 4),
            "structure_score": round(self._structure_score(columns), 4),
            "transaction_evidence": round(min(transaction_rows / 5.0, 1.0), 4),
            "transaction_rows": transaction_rows,
            "repeated_rows": transaction_rows,
            "header_top": round(min(rows[i][0]["top"] for i in range(start, end + 1)), 2),
        }

    def _group_words_by_top(
        self,
        words: List[Dict],
    ) -> List[List[Dict]]:
        rows: List[List[Dict]] = []
        for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
            placed = False
            for row in rows:
                average_top = sum(item["top"] for item in row) / len(row)
                if abs(word["top"] - average_top) <= self.WORD_ROW_TOLERANCE:
                    row.append(word)
                    placed = True
                    break
            if not placed:
                rows.append([word])
        for row in rows:
            row.sort(key=lambda item: item["x0"])
        return rows

    def _map_fragmented_header_words(
        self,
        row: List[Dict],
    ) -> List[Dict]:
        mapped = []
        for word in row:
            text = str(word["text"]).strip()
            if not text or text.upper() in {"DR", "CR", "/CR"}:
                continue

            semantic_type, confidence = self._map_header_cached(text)
            semantic_type = semantic_type or self._generic_header_alias(text)
            if not semantic_type:
                continue

            mapped.append({
                **word,
                "semantic_type": semantic_type,
                "confidence": float(confidence if confidence else 0.90),
            })
        return mapped

    def _generic_header_alias(self, text: str) -> Optional[str]:
        """Small universal vocabulary for common statement terminology."""
        normalized = " ".join(str(text).lower().replace("/", " ").split())
        aliases = {
            "particulars": "description",
            "narration": "description",
            "narrative": "description",
            "details": "description",
            "description": "description",
            "withdrawal": "debit",
            "withdrawals": "debit",
            "debit": "debit",
            "debits": "debit",
            "deposit": "credit",
            "deposits": "credit",
            "credit": "credit",
            "credits": "credit",
            "balance": "balance",
            "closing balance": "balance",
            "running balance": "balance",
            "available balance": "balance",
            "date": "transaction_date",
            "transaction date": "transaction_date",
            "tran date": "transaction_date",
            "txn date": "transaction_date",
            "value date": "value_date",
            "posting date": "posting_date",
            "post date": "posting_date",
            "tran id": "transaction_id",
            "txn id": "transaction_id",
            "transaction id": "transaction_id",
            "reference": "transaction_id",
            "reference no": "transaction_id",
            "reference number": "transaction_id",
            "amount": "amount",
            "transaction amount": "amount",
            "money out": "debit",
            "money in": "credit",
        }
        return aliases.get(normalized)

    def _fragmented_row_score(self, semantics: set) -> float:
        score = 0.0
        if semantics.intersection(self.CORE_DATE_TYPES):
            score += 2.0
        if "description" in semantics:
            score += 2.0
        if semantics.intersection(self.AMOUNT_TYPES):
            score += 2.0
        if "balance" in semantics:
            score += 1.5
        if "transaction_id" in semantics:
            score += 0.75
        if "value_date" in semantics or "posting_date" in semantics:
            score += 0.5
        return score

    def _build_fragmented_columns(
        self,
        mapped_words: List[Dict],
    ) -> List[DetectedColumn]:
        """
        Merge only adjacent header words that share a column region.
        This specifically handles headers such as Value + Date and
        Tran + ID without hardcoding a bank layout.
        """
        if not mapped_words:
            return []

        ordered = sorted(mapped_words, key=lambda item: (item["x0"], item["top"]))
        groups: List[List[Dict]] = []

        for word in ordered:
            best_group = None
            best_distance = None
            center = (word["x0"] + word["x1"]) / 2.0

            for group in groups:
                group_x0 = min(item["x0"] for item in group)
                group_x1 = max(item["x1"] for item in group)
                group_center = (group_x0 + group_x1) / 2.0
                distance = abs(center - group_center)

                # A multi-line header normally stays inside the same
                # horizontal column. The tolerance is deliberately
                # generous enough for different PDF font metrics.
                tolerance = max(18.0, (group_x1 - group_x0) * 1.5)
                if distance <= tolerance and (
                    best_distance is None or distance < best_distance
                ):
                    best_group = group
                    best_distance = distance

            if best_group is None:
                groups.append([word])
            else:
                best_group.append(word)

        columns: List[DetectedColumn] = []
        for group in groups:
            group.sort(key=lambda item: (item["top"], item["x0"]))
            texts = [item["text"] for item in group]

            # Try the combined phrase first. If it is not recognized,
            # choose the strongest semantic token in that physical
            # column.
            combined = " ".join(texts).strip()
            semantic_type, confidence = self._map_header_cached(combined)
            semantic_type = semantic_type or self._generic_header_alias(combined)

            if semantic_type is None:
                ranked = sorted(
                    group,
                    key=lambda item: item["confidence"],
                    reverse=True,
                )
                chosen = ranked[0]
                semantic_type = chosen["semantic_type"]
                confidence = chosen["confidence"]
                header = chosen["text"]
            else:
                header = combined

            columns.append(
                DetectedColumn(
                    header=header,
                    semantic_type=semantic_type,
                    x0=min(item["x0"] for item in group),
                    x1=max(item["x1"] for item in group),
                    top=min(item["top"] for item in group),
                    confidence=float(confidence or 0.90),
                )
            )

        # Keep one physical column per semantic role at nearby x
        # positions. Distinct debit/credit/balance columns are retained.
        deduped: List[DetectedColumn] = []
        for column in sorted(columns, key=lambda item: item.x0):
            duplicate = None
            center = (column.x0 + column.x1) / 2.0
            for index, existing in enumerate(deduped):
                if existing.semantic_type != column.semantic_type:
                    continue
                existing_center = (existing.x0 + existing.x1) / 2.0
                if abs(existing_center - center) <= 15.0:
                    duplicate = index
                    break
            if duplicate is None:
                deduped.append(column)
            elif column.confidence > deduped[duplicate].confidence:
                deduped[duplicate] = column

        semantics = {column.semantic_type for column in deduped}
        has_date = bool(semantics.intersection(self.CORE_DATE_TYPES))
        has_money = bool(semantics.intersection(self.AMOUNT_TYPES))
        has_description = "description" in semantics

        if not has_date or not (has_money or has_description):
            return []

        return deduped

    def _count_transaction_evidence(
        self,
        rows: List[List[Dict]],
        header_end: int,
        columns: List[DetectedColumn],
    ) -> int:
        """Count repeated transaction-like rows after the header.

        Financial PDFs frequently wrap descriptions onto continuation lines,
        so this method only requires a date plus at least one numeric amount on
        the first physical line of a transaction.
        """
        if header_end >= len(rows) - 1:
            return 0

        count = 0
        for row in rows[header_end + 1:]:
            joined = " ".join(str(item["text"]) for item in row).strip()
            if not joined:
                continue

            lower = joined.casefold()
            if any(term in lower for term in (
                "opening balance",
                "closing balance",
                "balance forward",
                "brought forward",
                "carried forward",
                "statement summary",
            )):
                continue

            has_date = self._row_has_date(row, [])
            numeric_count = sum(
                1 for item in row if self._looks_numeric(item["text"])
            )

            if has_date and numeric_count >= 1:
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
            text = str(item["text"]).strip()
            if self._looks_like_date_token(text):
                if not date_columns:
                    return True
                center = (item["x0"] + item["x1"]) / 2.0
                nearest = min(
                    date_columns,
                    key=lambda column: abs(
                        ((column.x0 + column.x1) / 2.0) - center
                    ),
                )
                if abs(
                    ((nearest.x0 + nearest.x1) / 2.0) - center
                ) <= 85.0:
                    return True
        return False

    def _looks_like_date_token(self, text: str) -> bool:
        import re

        value = str(text).strip().upper()
        if not value:
            return False

        months = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
        patterns = [
            rf"^\d{{1,2}}[-/]({months})[-/]\d{{2,4}}$",
            rf"^\d{{4}}[-/]({months})[-/]\d{{1,2}}$",
            r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$",
            r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$",
        ]
        if any(re.match(pattern, value) for pattern in patterns):
            return True

        # Individual tokens used by split PDF dates, e.g. 31 + AUG + 2026.
        if re.fullmatch(r"\d{1,2}", value):
            number = int(value)
            return 1 <= number <= 31
        if value[:3] in months.split("|") and len(value) <= 9:
            return True
        if re.fullmatch(r"\d{4}", value):
            year = int(value)
            return 1900 <= year <= 2100
        if re.fullmatch(r"\d{2}", value):
            year = int(value)
            return 0 <= year <= 99
        return False

    def _looks_numeric(self, text: str) -> bool:
        value = str(text).strip().replace(",", "")
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _fragmented_structure_score(
        self,
        columns: List[DetectedColumn],
        transaction_rows: int,
    ) -> float:
        base = self._structure_score(columns)
        evidence = min(transaction_rows / 10.0, 1.0)
        return min((base * 0.65) + (evidence * 0.35), 1.0)

    # ================================================================
    # SEMANTIC MAPPING CACHE / FAST SCREENING
    # ================================================================

    def _map_header_cached(
        self,
        text: str,
    ) -> Tuple[Optional[str], float]:
        """Cache semantic mapper calls for repeated PDF vocabulary."""
        key = " ".join(str(text).strip().casefold().split())
        if not key:
            return None, 0.0

        cached = self._semantic_cache.get(key)
        if cached is not None:
            return cached

        result = self.mapper.map_header_with_confidence(text)
        if result is None:
            result = (None, 0.0)
        else:
            semantic_type, confidence = result
            result = (semantic_type, float(confidence or 0.0))

        self._semantic_cache[key] = result
        return result

    def _has_financial_header_signal(
        self,
        words: List[Dict],
    ) -> bool:
        """Cheap universal vocabulary gate for fragmented-table fallback.

        This gate never decides that a page *is* a statement. It only decides
        whether the expensive fragmented fallback is worth attempting.
        """
        signals = {
            "date",
            "value",
            "posting",
            "post",
            "transaction",
            "trans",
            "tran",
            "txn",
            "reference",
            "ref",
            "description",
            "details",
            "particulars",
            "narration",
            "amount",
            "debit",
            "credit",
            "withdrawal",
            "withdrawals",
            "deposit",
            "deposits",
            "balance",
            "opening",
            "closing",
        }

        for word in words:
            text = " ".join(
                str(word.get("text", "")).casefold().split()
            )
            if text in signals:
                return True

        return False

    # ================================================================
    # NORMALIZE PDF WORDS
    # ================================================================

    def _normalize_words(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        result = []

        for word in words:

            text = str(
                word.get("text", "")
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
    # INDIVIDUAL SEMANTIC WORD MAPPING
    # ================================================================

    def _map_individual_words(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        result = []

        for word in words:

            text = word["text"]

            semantic_type, confidence = (
                self._map_header_cached(
                    text
                )
            )

            if not semantic_type:
                continue

            # Very short generic words such as "Dr" and "Cr" are
            # intentionally ignored here.
            if text.upper() in {
                "DR",
                "CR",
            }:
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
    # HEADER BAND DETECTION
    # ================================================================

    def _find_header_band(
        self,
        semantic_words: List[Dict],
    ) -> List[Dict]:

        """
        Find the densest horizontal semantic band.

        A transaction header usually contains several semantic fields
        close together:

            Date
            Description
            Debit
            Credit
            Balance

        This method identifies that band BEFORE multiline
        reconstruction.

        Therefore:

            Value
            Date

        can be combined, while a transaction-row Date cannot.
        """

        if not semantic_words:
            return []

        # ------------------------------------------------------------
        # Build top clusters.
        # ------------------------------------------------------------

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

                    cluster.append(word)
                    placed = True
                    break

            if not placed:
                clusters.append(
                    [word]
                )

        if not clusters:
            return []

        # ------------------------------------------------------------
        # Score clusters.
        # ------------------------------------------------------------

        scored_clusters = []

        for cluster in clusters:

            score = self._header_band_score(
                cluster
            )

            scored_clusters.append(
                (
                    score,
                    cluster,
                )
            )

        scored_clusters.sort(
            key=lambda item: (
                item[0],
                len(item[1]),
            ),
            reverse=True,
        )

        best_cluster = (
            scored_clusters[0][1]
        )

        return sorted(
            best_cluster,
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

    # ================================================================
    # HEADER BAND SCORE
    # ================================================================

    def _header_band_score(
        self,
        words: List[Dict],
    ) -> float:

        semantics = {
            word["semantic_type"]
            for word in words
        }

        score = 0.0

        # More semantic columns = stronger table-header evidence.
        score += min(
            len(words) * 0.12,
            0.60,
        )

        # Date evidence.
        if semantics.intersection(
            self.CORE_DATE_TYPES
        ):
            score += 0.40

        # Description evidence.
        if "description" in semantics:
            score += 0.40

        # Monetary evidence.
        if semantics.intersection(
            self.AMOUNT_TYPES
        ):
            score += 0.40

        # Balance evidence.
        if "balance" in semantics:
            score += 0.20

        # Transaction ID evidence.
        if "transaction_id" in semantics:
            score += 0.10

        # Average mapper confidence.
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
    # MULTI-LINE HEADER RECONSTRUCTION
    # ================================================================

    def _reconstruct_headers(
        self,
        words: List[Dict],
    ) -> List[Dict]:

        """
        Reconstruct only words inside the already-selected
        transaction-header band.

        This is the critical difference from the previous version.

        Example:

            Value
            Date

        becomes:

            Value Date

        while:

            Date
            17 Jun 19

        does NOT get combined because the transaction row is outside
        the selected header band.
        """

        words = sorted(
            words,
            key=lambda word: (
                word["top"],
                word["x0"],
            ),
        )

        used = set()
        result = []

        for index, first in enumerate(words):

            if index in used:
                continue

            best_index = None
            best_score = 0.0

            first_width = max(
                first["x1"]
                - first["x0"],
                1.0,
            )

            first_center = (
                first["x0"]
                + first["x1"]
            ) / 2

            for candidate_index in range(
                index + 1,
                len(words),
            ):

                if candidate_index in used:
                    continue

                second = words[
                    candidate_index
                ]

                vertical_gap = (
                    second["top"]
                    - first["top"]
                )

                if vertical_gap <= 0:
                    continue

                if (
                    vertical_gap
                    > self.MULTILINE_MAX_VERTICAL_GAP
                ):
                    break

                second_width = max(
                    second["x1"]
                    - second["x0"],
                    1.0,
                )

                second_center = (
                    second["x0"]
                    + second["x1"]
                ) / 2

                # ----------------------------------------------------
                # Horizontal overlap.
                # ----------------------------------------------------

                overlap_left = max(
                    first["x0"],
                    second["x0"],
                )

                overlap_right = min(
                    first["x1"],
                    second["x1"],
                )

                overlap = max(
                    0.0,
                    overlap_right
                    - overlap_left,
                )

                overlap_ratio = (
                    overlap
                    / min(
                        first_width,
                        second_width,
                    )
                )

                if (
                    overlap_ratio
                    < self.MIN_HORIZONTAL_OVERLAP_RATIO
                ):
                    continue

                # ----------------------------------------------------
                # Center alignment.
                # ----------------------------------------------------

                center_distance = abs(
                    first_center
                    - second_center
                )

                max_width = max(
                    first_width,
                    second_width,
                )

                center_ratio = (
                    center_distance
                    / max_width
                )

                if center_ratio > 0.80:
                    continue

                # ----------------------------------------------------
                # Score.
                # ----------------------------------------------------

                vertical_score = max(
                    0.0,
                    1.0
                    - (
                        vertical_gap
                        / self.MULTILINE_MAX_VERTICAL_GAP
                    ),
                )

                center_score = max(
                    0.0,
                    1.0
                    - min(
                        center_ratio,
                        1.0,
                    ),
                )

                score = (
                    overlap_ratio
                    * 0.60
                    + center_score
                    * 0.25
                    + vertical_score
                    * 0.15
                )

                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            # --------------------------------------------------------
            # Strong geometric relationship.
            # --------------------------------------------------------

            if (
                best_index is not None
                and best_score >= 0.70
            ):

                second = words[
                    best_index
                ]

                # Only combine if doing so creates a meaningful
                # semantic header.
                combined_text = (
                    f"{first['text']} "
                    f"{second['text']}"
                ).strip()

                combined_semantic, combined_confidence = (
                    self._map_header_cached(
                        combined_text
                    )
                )

                # If the combined phrase is understood by the
                # semantic mapper, use it.
                if combined_semantic:

                    result.append(
                        {
                            "header": combined_text,
                            "x0": min(
                                first["x0"],
                                second["x0"],
                            ),
                            "x1": max(
                                first["x1"],
                                second["x1"],
                            ),
                            "top": min(
                                first["top"],
                                second["top"],
                            ),
                            "bottom": max(
                                first["bottom"],
                                second["bottom"],
                            ),
                        }
                    )

                    used.add(index)
                    used.add(best_index)

                    continue

            # --------------------------------------------------------
            # No valid combination.
            # --------------------------------------------------------

            result.append(
                {
                    "header": first["text"],
                    "x0": first["x0"],
                    "x1": first["x1"],
                    "top": first["top"],
                    "bottom": first["bottom"],
                }
            )

            used.add(index)

        return result

    # ================================================================
    # SELECT TRANSACTION COLUMNS
    # ================================================================

    def _select_transaction_columns(
        self,
        candidates: List[DetectedColumn],
    ) -> List[DetectedColumn]:

        if not candidates:
            return []

        # ------------------------------------------------------------
        # Deduplicate only physically identical columns.
        # ------------------------------------------------------------

        selected = []

        for candidate in sorted(
            candidates,
            key=lambda column: (
                column.x0,
                column.top,
            ),
        ):

            duplicate_index = None

            for index, existing in enumerate(
                selected
            ):

                if (
                    existing.semantic_type
                    != candidate.semantic_type
                ):
                    continue

                existing_center = (
                    existing.x0
                    + existing.x1
                ) / 2

                candidate_center = (
                    candidate.x0
                    + candidate.x1
                ) / 2

                if abs(
                    existing_center
                    - candidate_center
                ) <= 12.0:

                    duplicate_index = index
                    break

            if duplicate_index is None:

                selected.append(
                    candidate
                )

            elif (
                candidate.confidence
                > selected[
                    duplicate_index
                ].confidence
            ):

                selected[
                    duplicate_index
                ] = candidate

        # ------------------------------------------------------------
        # Require enough transaction structure.
        # ------------------------------------------------------------

        semantics = {
            column.semantic_type
            for column in selected
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

        if not (
            has_date
            and (
                has_description
                or has_amount
            )
        ):
            return []

        return sorted(
            selected,
            key=lambda column: column.x0,
        )

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