from __future__ import annotations

import os

import re

import shutil

import time

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import pymupdf

import pytesseract

from pytesseract import Output

from PIL import Image

class OCRFallback:

    """OCR fallback for scanned/image financial statements."""

    WINDOWS_TESSERACT_PATHS = (

        r"C:\Program Files\Tesseract-OCR\tesseract.exe",

        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",

    )

    UNIX_TESSERACT_PATHS = (

        "/usr/bin/tesseract",

        "/usr/local/bin/tesseract",

    )

    OCR_SCALE = 3.0

    OCR_LANGUAGE = "eng"

    PRIMARY_CONFIG = "--psm 6"

    NUMERIC_CONFIG = "--psm 11"

    DATE_RE = re.compile(r"^(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4})$", re.IGNORECASE)

    NUMBER_RE = re.compile(r"^-?\d[\d,]*(?:\.\d+)?$")

    HEADER_ALIASES = {

        "transaction_date": {"date", "txn", "txn date", "transaction date", "trans date", "trans. date", "date of transaction"},

        "value_date": {"value date", "value"},

        "description": {"description", "details", "particulars", "narration", "remarks", "transaction details"},

        "debit": {"debit", "debits", "withdrawal", "withdrawals", "withdraw", "withdraws", "dr"},

        "credit": {"credit", "credits", "deposit", "deposits", "deposit amount", "cr"},

        "balance": {"balance", "closing balance", "available balance"},

        "amount": {"amount", "transaction amount"},

    }

    def __init__(self, file_path: str | Path, password: Optional[str] = None):

        self.file_path = str(file_path)

        self.password = password

        self.detected_currency: Optional[str] = None

        self.tesseract_path = self._configure_tesseract()

    def _configure_tesseract(self) -> str:

        """Find a working Tesseract executable on Windows or Unix."""

        candidates: List[str] = []

        env_path = os.getenv("TESSERACT_CMD")

        if env_path:

            candidates.append(env_path)

        path_from_path = shutil.which("tesseract")

        if path_from_path:

            candidates.append(path_from_path)

        if os.name == "nt":

            candidates.extend(self.WINDOWS_TESSERACT_PATHS)

        else:

            candidates.extend(self.UNIX_TESSERACT_PATHS)

        seen = set()

        for candidate in candidates:

            if not candidate or candidate in seen:

                continue

            seen.add(candidate)

            if Path(candidate).is_file():

                pytesseract.pytesseract.tesseract_cmd = candidate

                print(f"Tesseract executable: {candidate}")

                return candidate

        raise RuntimeError(

            "Tesseract OCR executable was not found. "

            "Install Tesseract or set TESSERACT_CMD to the full "

            "path of tesseract.exe."

        )

    def extract(self) -> List[Dict[str, Any]]:

        pdf = self._open_pdf()

        rows: List[Dict[str, Any]] = []

        try:

            for page_number, page in enumerate(pdf, start=1):

                print(f"OCR fallback: processing page {page_number}...")

                render_start = time.perf_counter()

                image = self._render_page(page)

                render_time = time.perf_counter() - render_start

                primary_start = time.perf_counter()

                primary_words = self._ocr_words(image, self.PRIMARY_CONFIG)

                primary_time = time.perf_counter() - primary_start

                numeric_words = []

                numeric_time = 0.0

                if primary_words:

                    lines = self._build_lines(primary_words)

                    columns = self._detect_columns(lines)

                    has_date_evidence = any(

                        self._is_date(word.get("text", ""))

                        and float(word.get("top", 0)) >= 200.0

                        for word in primary_words

                    )

                    if columns or has_date_evidence:

                        numeric_start = time.perf_counter()

                        numeric_words = self._ocr_words(

                            image, self.NUMERIC_CONFIG

                        )

                        # Retry the detected amount column with sparse-text OCR.
                        # This mode can recognize decimals that PSM 6 misreads.
                        if "amount" in columns:
                            amount_center = int(columns["amount"])
                            crop_left = max(0, amount_center - 220)
                            crop_right = min(image.width, amount_center + 220)

                            if crop_right > crop_left:
                                amount_crop = image.crop(
                                    (crop_left, 0, crop_right, image.height)
                                )
                                # Use a single-line/column-friendly numeric OCR pass for amounts.
                                # PSM 6 plus a strict numeric whitelist recognizes
                                # decimals such as 54.92 directly, unlike sparse PSM 11.
                                amount_words = self._ocr_words(
                                    amount_crop,
                                    "--psm 6 -c tessedit_char_whitelist=0123456789."
                                )

                                # Translate crop coordinates back to page coordinates.
                                for word in amount_words:
                                    word["x0"] = float(word["x0"]) + crop_left
                                    word["x1"] = float(word["x1"]) + crop_left

                                # PSM 11 may split a decimal amount into two
                                # adjacent tokens, e.g. "13" and "50".
                                # Merge only horizontally adjacent tokens on
                                # the same row and inside the detected amount column.
                                ordered_amount_words = sorted(
                                    amount_words,
                                    key=lambda w: (
                                        float(w["top"]),
                                        float(w["x0"]),
                                    ),
                                )
                                merged_words = []
                                consumed = set()

                                for i, first in enumerate(ordered_amount_words):
                                    if i in consumed:
                                        continue

                                    first_text = str(first["text"]).strip()
                                    first_center = (
                                        float(first["x0"]) + float(first["x1"])
                                    ) / 2.0

                                    if not (
                                        re.fullmatch(r"\d{1,3}", first_text)
                                        and abs(first_center - amount_center) <= 125
                                    ):
                                        continue

                                    for j in range(i + 1, len(ordered_amount_words)):
                                        if j in consumed:
                                            continue

                                        second = ordered_amount_words[j]
                                        second_text = str(second["text"]).strip()

                                        if not re.fullmatch(r"\d{2}", second_text):
                                            continue

                                        vertical_gap = abs(
                                            float(first["top"]) - float(second["top"])
                                        )
                                        horizontal_gap = (
                                            float(second["x0"]) - float(first["x1"])
                                        )

                                        if (
                                            vertical_gap <= 5
                                            and 0 <= horizontal_gap <= 35
                                        ):
                                            merged = dict(first)
                                            merged["text"] = f"{first_text}.{second_text}"
                                            merged["x1"] = float(second["x1"])
                                            merged["top"] = min(
                                                float(first["top"]),
                                                float(second["top"]),
                                            )
                                            merged_words.append(merged)
                                            consumed.add(i)
                                            consumed.add(j)
                                            break

                                numeric_words.extend(amount_words)
                                numeric_words.extend(merged_words)

                        numeric_time = time.perf_counter() - numeric_start

                    else:

                        print(

                            f"OCR page {page_number}: skipping PSM 11; "

                            "no transaction columns or date evidence detected."

                        )

                print(

                    f"OCR TIMING page={page_number}: "

                    f"render={render_time:.2f}s, "

                    f"psm6={primary_time:.2f}s, "

                    f"psm11={numeric_time:.2f}s, "

                    f"total={render_time + primary_time + numeric_time:.2f}s"

                )

                if not primary_words:

                    continue

                if self.detected_currency is None:

                    self.detected_currency = self._detect_currency(

                        primary_words + numeric_words

                    )

                page_rows = self._extract_page(

                    primary_words, numeric_words, page_number

                )

                rows.extend(page_rows)

        finally:

            pdf.close()

        print(f"OCR fallback rows extracted: {len(rows)}")

        return rows

    def _detect_currency(

        self,

        words: List[Dict[str, Any]],

    ) -> Optional[str]:

        """Detect currency from OCR text using explicit currency indicators."""

        text = " ".join(

            str(word.get("text", ""))

            for word in words

        )

        patterns = {

            "INR": (

                r"\bINR\b",

                r"\bRs?.?\b",

                r"\bRupees?\b",

                r"\bIndian Rupees?\b",

                r"\bAccount Currency\s*:?\s*INR\b",

            ),

            "AED": (

                r"\bAED\b",

                r"\bUAE Dirham(?:s)?\b",

            ),

            "USD": (

                r"\bUSD\b",

                r"\bUS Dollars?\b",

                r"\$",

            ),

            "EUR": (

                r"\bEUR\b",

                r"\bEuros?\b",

                r"Ã¢â€šÂ¬",

            ),

            "GBP": (

                r"\bGBP\b",

                r"\bPounds?\b",

                r"Ã‚Â£",

            ),

            "SAR": (

                r"\bSAR\b",

                r"\bSaudi Riyals?\b",

            ),

            "QAR": (

                r"\bQAR\b",

                r"\bQatari Riyals?\b",

            ),

            "KWD": (

                r"\bKWD\b",

                r"\bKuwaiti Dinars?\b",

            ),

            "BHD": (

                r"\bBHD\b",

                r"\bBahraini Dinars?\b",

            ),

            "OMR": (

                r"\bOMR\b",

                r"\bOmani Rials?\b",

            ),

        }

        scores = {}

        for currency, currency_patterns in patterns.items():

            score = 0

            for pattern in currency_patterns:

                score += len(re.findall(pattern, text, flags=re.IGNORECASE))

            if score:

                scores[currency] = score

        if not scores:

            return None

        return max(scores, key=scores.get)

    def _open_pdf(self):

        """Open the PDF and authenticate encrypted documents when needed.

        PyMuPDF versions used by Finora do not accept *``*password=*``* in

        `pymupdf.open()`. The supported flow is to open the document first

        and then call *``*authenticate()*``* when the document requires a password.

        """

        pdf = pymupdf.open(self.file_path)

        if pdf.needs_pass:

            if not self.password:

                pdf.close()

                raise ValueError(

                    "This PDF is password protected. Please enter the PDF password."

                )

            authenticated = pdf.authenticate(self.password)

            if not authenticated:

                pdf.close()

                raise ValueError(

                    "Incorrect PDF password. Please enter the correct password and try again."

                )

        return pdf

    def _render_page(self, page) -> Image.Image:

        pix = page.get_pixmap(matrix=pymupdf.Matrix(self.OCR_SCALE, self.OCR_SCALE), alpha=False)

        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _ocr_words(self, image: Image.Image, config: str) -> List[Dict[str, Any]]:

        data = pytesseract.image_to_data(

            image,

            lang=self.OCR_LANGUAGE,

            config=config,

            output_type=Output.DICT,

        )

        words: List[Dict[str, Any]] = []

        for i, raw_text in enumerate(data.get("text", [])):

            text = str(raw_text).strip()

            if not text:

                continue

            try:

                conf = float(data["conf"][i])

            except Exception:

                conf = -1

            if conf < 20:

                continue

            try:

                left = int(data["left"][i])

                top = int(data["top"][i])

                width = int(data["width"][i])

                height = int(data["height"][i])

            except Exception:

                continue

            words.append({

                "text": text,

                "x0": left,

                "x1": left + width,

                "top": top,

                "bottom": top + height,

                "block": int(data["block_num"][i]),

                "paragraph": int(data["par_num"][i]),

                "line": int(data["line_num"][i]),

                "confidence": conf,

            })

        return words

    def _build_lines(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        grouped: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}

        for word in words:

            key = (word["block"], word["paragraph"], word["line"])

            grouped.setdefault(key, []).append(word)

        lines = []

        for ws in grouped.values():

            ws.sort(key=lambda w: float(w["x0"]))

            text = " ".join(w["text"] for w in ws).strip()

            if not text:

                continue

            lines.append({

                "top": min(float(w["top"]) for w in ws),

                "bottom": max(float(w["bottom"]) for w in ws),

                "words": ws,

                "text": text,

            })

        return sorted(lines, key=lambda x: x["top"])

    def _detect_columns(self, lines: List[Dict[str, Any]]) -> Dict[str, float]:

        first_transaction = None

        for i, line in enumerate(lines):

            if self._starts_with_date(line["text"]):

                first_transaction = i

                break

        if first_transaction is None:

            return {}

        header_lines = lines[max(0, first_transaction - 15):first_transaction]

        candidates = []

        for li, line in enumerate(header_lines):

            found: Dict[str, float] = {}

            ws = line["words"]

            for w in ws:

                semantic = self._header_semantic(self._normalize_header(w["text"]))

                if semantic:

                    found[semantic] = (float(w["x0"]) + float(w["x1"])) / 2

            for i in range(len(ws)):

                for width in (2, 3):

                    if i + width > len(ws):

                        continue

                    combined = " ".join(

                        self._normalize_header(ws[j]["text"])

                        for j in range(i, i + width)

                    ).strip()

                    semantic = self._header_semantic(combined)

                    if semantic:

                        found[semantic] = (

                            float(ws[i]["x0"]) + float(ws[i + width - 1]["x1"])

                        ) / 2

            if len(found) >= 3:

                candidates.append((len(found), li, found))

        if not candidates:

            return {}

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        columns = dict(candidates[0][2])

        return columns

    def _extract_page(

        self,

        primary_words: List[Dict[str, Any]],

        numeric_words: List[Dict[str, Any]],

        page_number: int,

    ) -> List[Dict[str, Any]]:

        lines = self._build_lines(primary_words)

        columns = self._detect_columns(lines)

        if not columns:

            return []

        # Anchor rows using individual date words, not logical OCR lines.

        # Tesseract can group adjacent physical transaction rows together.

        anchor_tops = []

        # Ignore dates in statement headers, such as payment due dates.

        # Transaction rows begin below the detected table header.

        for word in sorted(primary_words, key=lambda item: float(item["top"])):

            if not self._is_date(word["text"]):

                continue

            top = float(word["top"])

            if top < 200.0:

                continue

            if not anchor_tops or abs(top - anchor_tops[-1]) > 18.0:

                anchor_tops.append(top)

        # Remove isolated dates outside the transaction table.

        # Header dates can be far away from the next transaction date and

        # otherwise create an oversized region that swallows the first row.

        anchors = [

            top

            for i, top in enumerate(anchor_tops)

            if (

                (i > 0 and top - anchor_tops[i - 1] <= 250.0)

                or

                (i + 1 < len(anchor_tops) and anchor_tops[i + 1] - top <= 250.0)

            )

        ]

        print(f"OCR page {page_number}: transaction lines = {len(anchors)}")

        rows: List[Dict[str, Any]] = []

        previous_balance: Optional[float] = None

        for index, anchor_top in enumerate(anchors):

            previous_top = anchors[index - 1] if index else None

            next_top = anchors[index + 1] if index + 1 < len(anchors) else None

            start = ((previous_top + anchor_top) / 2.0) if previous_top is not None else anchor_top - 30.0

            end = ((anchor_top + next_top) / 2.0) if next_top is not None else anchor_top + 130.0

            region_primary = [w for w in primary_words if start <= float(w["top"]) < end]

            region_numeric = [w for w in numeric_words if start <= float(w["top"]) < end]

            row = self._parse_region(region_primary, region_numeric, columns)

            if not row:

                dates = [

                    str(w["text"]) for w in region_primary

                    if self._is_date(w["text"])

                ]

                words = " ".join(str(w["text"]) for w in region_primary)

                print(

                    f"OCR ROW REJECTED page={page_number} "

                    f"anchor={anchor_top:.1f} dates={dates} words={words[:220]!r}"

                )

                continue

            print(

                f"OCR ROW PARSED page={page_number} "

                f"anchor={anchor_top:.1f} "

                f"date={row.get('transaction_date')} "

                f"description={row.get('description')!r} "

                f"amount={row.get('amount')} debit={row.get('debit')} "

                f"credit={row.get('credit')} balance={row.get('balance')}"

            )

            # Statement totals are not individual transactions.

            description_upper = str(row.get("description") or "").upper()

            if any(term in description_upper for term in (

                # OCR may reorder words in statement summary lines.

                "BALANCE OUTSTANDING",

                "TOTAL AMOUNT DUE",

                "MINIMUM PAYMENT DUE",

                "STATEMENT BALANCE",

            )):

                continue

            current_balance = self._to_float(row.get("balance"))

            amount = self._to_float(row.get("amount"))

            debit = self._to_float(row.get("debit"))

            credit = self._to_float(row.get("credit"))

            # Reconcile explicit amount against the running balance.

            if previous_balance is not None:

                if debit is not None:

                    expected = previous_balance - debit

                    if current_balance is None or abs(current_balance - expected) > 1.0:

                        current_balance = expected

                        row["balance"] = f"{current_balance:.2f}"

                elif credit is not None:

                    expected = previous_balance + credit

                    if current_balance is None or abs(current_balance - expected) > 1.0:

                        current_balance = expected

                        row["balance"] = f"{current_balance:.2f}"

                elif current_balance is not None:

                    movement = current_balance - previous_balance

                    if movement < -0.000001:

                        row["debit"] = f"{abs(movement):.2f}"

                    elif movement > 0.000001:

                        row["credit"] = f"{abs(movement):.2f}"

                elif amount is not None:

                    self._classify_generic_amount(row, amount, previous_balance)

                    current_balance = self._to_float(row.get("balance"))

            else:

                # First transaction: use semantic text when a balance-only

                # credit is present (salary/deposit/received).

                if not debit and not credit and not amount and current_balance is not None:

                    desc = str(row.get("description") or "").upper()

                    if self._looks_credit(desc):

                        row["credit"] = f"{current_balance:.2f}"

            rows.append({

                **row,

                "page_number": page_number,

                "source_file": self.file_path,

            })

            if current_balance is not None:

                previous_balance = current_balance

        self._final_reconcile(rows)

        return rows

    def _parse_region(

        self,

        primary_words: List[Dict[str, Any]],

        numeric_words: List[Dict[str, Any]],

        columns: Dict[str, float],

    ) -> Optional[Dict[str, Any]]:

        if not primary_words:

            return None

        date_words = [

            w for w in primary_words

            if self._is_date(w["text"])

        ]

        if not date_words:

            return None

        date_words.sort(key=lambda w: float(w["x0"]))

        row: Dict[str, Any] = {

            "transaction_date": self._clean_date(date_words[0]["text"])

        }

        if len(date_words) >= 2:

            row["value_date"] = self._clean_date(date_words[1]["text"])

        # Merge numeric tokens from both OCR passes. PSM 6 is usually better

        # aligned to transaction rows; PSM 11 can recover sparse numeric tokens.

        candidates = []

        for source in (primary_words, numeric_words):

            for w in source:

                if not self._is_number(w["text"]):

                    continue

                value_text = self._clean_number(w["text"])

                value = self._to_float(value_text)

                if value is None:

                    continue

                if re.fullmatch(r"(19|20|21)\d{2}", value_text):

                    continue

                if re.fullmatch(r"\d{6,}", value_text):

                    continue

                # Reject short integer branch/reference codes unless the token

                # is spatially aligned with a detected monetary column. OCR can

                # recognize an amount such as 7.75 as separate integer tokens;

                # do not discard those tokens before column assignment.

                center = (float(w["x0"]) + float(w["x1"])) / 2

                is_short_integer = (

                    re.fullmatch(r"\d{1,4}", value_text) is not None

                    and "." not in value_text

                    and value_text != "0"

                )

                near_money_column = any(

                    name in columns

                    and abs(center - columns[name]) <= 125

                    for name in ("debit", "credit", "amount", "balance")

                )

                if is_short_integer and not near_money_column:

                    continue

                candidates.append({

                    "text": value_text,

                    "value": value,

                    "center": center,

                    "top": float(w["top"]),

                })

        # If a decimal OCR candidate exists in the monetary column for a row,
        # discard same-row integer fragments from the other OCR pass. Without
        # this, the integer (e.g. 54) can win over the recovered decimal (54.92).
        decimal_money_candidates = [
            item for item in candidates
            if "." in item["text"]
            and any(
                name in columns and abs(item["center"] - columns[name]) <= 125
                for name in ("debit", "credit", "amount", "balance")
            )
        ]
        if decimal_money_candidates:
            candidates = [
                item for item in candidates
                if not (
                    "." not in item["text"]
                    and any(
                        abs(item["top"] - decimal_item["top"]) <= 8
                        and abs(item["center"] - decimal_item["center"]) <= 100
                        for decimal_item in decimal_money_candidates
                    )
                )
            ]

        candidates = self._dedupe_candidates(candidates)

        # Numeric column centres. We use a tight tolerance to avoid pulling

        # branch/cheque numbers into money columns.

        numeric_columns = [

            (name, columns[name])

            for name in ("debit", "credit", "amount", "balance")

            if name in columns

        ]

        used = set()

        pairs = []

        for i, item in enumerate(candidates):

            for name, center in numeric_columns:

                pairs.append((abs(item["center"] - center), i, name))

        for distance, i, name in sorted(pairs):

            if i in used or name in row or distance > 125:

                continue

            item = candidates[i]

            # Don't make zero a transaction amount.

            if name in {"debit", "credit", "amount"} and item["value"] == 0:

                continue

            row[name] = item["text"]

            used.add(i)

        # Generic fallback: rightmost number near Balance is balance.

        if "balance" not in row and candidates and "balance" in columns:

            right = max(candidates, key=lambda item: item["center"])

            if abs(right["center"] - columns["balance"]) <= 125:

                row["balance"] = right["text"]

                for i, item in enumerate(candidates):

                    if i not in used and item["text"] == right["text"] and abs(item["center"] - right["center"]) < 3:

                        used.add(i)

                        break

        # Description from primary OCR region. Keep meaningful text and

        # remove dates, pure numbers, IDs and table artifacts.

        desc_parts = []

        for w in sorted(primary_words, key=lambda w: (float(w["top"]), float(w["x0"]))):

            text = str(w["text"]).strip()

            if not text or self._is_date(text):

                continue

            if self._is_number(text):

                cleaned = self._clean_number(text)

                if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", cleaned):

                    continue

            if text.upper() in {"DR", "CR"}:

                continue

            cleaned = text.strip("|[]{}")

            if not cleaned:

                continue

            if re.fullmatch(r"\d{6,}", cleaned):

                continue

            desc_parts.append(cleaned)

        description = " ".join(desc_parts).strip()

        description = re.sub(r"^\d{1,2}-[A-Za-z]{3,9}-\d{2,4}\s*", "", description, flags=re.I)

        description = self._normalize_description(description)

        # Reject statement summaries, even if OCR finds an amount.

        description_upper = description.upper()

        if any(term in description_upper for term in (

            "BALANCE OUTSTANDING",

            "TOTAL AMOUNT DUE",

            "MINIMUM PAYMENT DUE",

            "STATEMENT BALANCE",

        )):

            return None

        # Preserve valid dated transactions when OCR misses the description.

        has_amount = any(

            self._to_float(row.get(field)) is not None

            for field in ("debit", "credit", "amount")

        )

        if not description and not has_amount:

            return None

        row["description"] = description or "Unrecognized transaction"

        return row

    def _normalize_description(self, description: str) -> str:

        """Normalize common OCR variants without guessing unknown merchants."""

        text = re.sub(r"\s+", " ", str(description)).strip()

        upper = re.sub(r"[^A-Z0-9%&]+", " ", text.upper())

        tokens = upper.split()

        # Common card-statement narration is often read in the wrong order.

        if "PAYMENT" in tokens and "RECEIVED" in tokens and (

            "THANK" in tokens or "YOU" in tokens

        ):

            return "PAYMENT RECEIVED, THANK YOU"

        # Preserve the reward wording while cleaning OCR punctuation.

        if "CASHBACK" in upper or "CASH BACK" in upper:

            return re.sub(r"\s+", " ", text.replace("â€”", "-")).strip(" ,:-")

        if "CBK" in tokens and "SPEND" in tokens:

            return re.sub(r"\s+", " ", text.replace("â€”", "-")).strip(" ,:-")

        return re.sub(r"\s+", " ", text).strip(" |[]{}")

    def _final_reconcile(self, rows: List[Dict[str, Any]]) -> None:

        """Use consecutive balances to recover OCR-missed amount/direction."""

        for i, row in enumerate(rows):

            prev = self._to_float(rows[i - 1].get("balance")) if i else None

            cur = self._to_float(row.get("balance"))

            debit = self._to_float(row.get("debit"))

            credit = self._to_float(row.get("credit"))

            amount = self._to_float(row.get("amount"))

            if prev is not None and cur is not None:

                movement = cur - prev

                # If a directly OCR'd debit/credit does not reconcile with

                # the balance, replace the bad value with the balance movement.

                if debit is not None and abs((prev - debit) - cur) > 1.0:

                    row.pop("debit", None)

                    row["debit"] = f"{abs(movement):.2f}" if movement < 0 else None

                    if movement > 0:

                        row.pop("debit", None)

                        row["credit"] = f"{movement:.2f}"

                elif credit is not None and abs((prev + credit) - cur) > 1.0:

                    row.pop("credit", None)

                    row["credit"] = f"{abs(movement):.2f}" if movement > 0 else None

                    if movement < 0:

                        row.pop("credit", None)

                        row["debit"] = f"{abs(movement):.2f}"

                elif debit is None and credit is None and movement != 0:

                    if movement < 0:

                        row["debit"] = f"{abs(movement):.2f}"

                    else:

                        row["credit"] = f"{movement:.2f}"

                # Remove generic amount once direction is known.

                if row.get("debit") or row.get("credit"):

                    row.pop("amount", None)

                continue

            # Missing current balance: derive it when explicit transaction amount exists.

            if prev is not None:

                if debit is not None:

                    cur = prev - debit

                    row["balance"] = f"{cur:.2f}"

                elif credit is not None:

                    cur = prev + credit

                    row["balance"] = f"{cur:.2f}"

                elif amount is not None:

                    self._classify_generic_amount(row, amount, prev)

                    if row.get("debit") is not None:

                        cur = prev - amount

                        row["balance"] = f"{cur:.2f}"

                    elif row.get("credit") is not None:

                        cur = prev + amount

                        row["balance"] = f"{cur:.2f}"

        # Recover balances/amounts in rows where the OCR numeric pass found

        # only a balance. One pass is enough for the tested transaction chain.

        for i, row in enumerate(rows):

            prev = self._to_float(rows[i - 1].get("balance")) if i else None

            cur = self._to_float(row.get("balance"))

            if prev is None or cur is None:

                continue

            if row.get("debit") or row.get("credit"):

                continue

            movement = cur - prev

            if movement < 0:

                row["debit"] = f"{abs(movement):.2f}"

            elif movement > 0:

                row["credit"] = f"{movement:.2f}"

        # First-row credit when the statement starts with a balance-only

        # salary/deposit transaction.

        if rows:

            first = rows[0]

            if not first.get("debit") and not first.get("credit"):

                bal = self._to_float(first.get("balance"))

                desc = str(first.get("description") or "").upper()

                if bal is not None and self._looks_credit(desc):

                    first["credit"] = f"{bal:.2f}"

    def _classify_generic_amount(self, row: Dict[str, Any], amount: float, previous_balance: float) -> None:

        desc = str(row.get("description") or "").upper()

        if self._looks_credit(desc):

            row["credit"] = row.pop("amount")

        elif self._looks_debit(desc):

            row["debit"] = row.pop("amount")

    def _looks_credit(self, description: str) -> bool:

        desc = re.sub(r"[^A-Z0-9]+", " ", str(description).upper())

        return any(term in desc for term in (

            "SALARY", "CREDIT", "DEPOSIT", "RECEIVED", "REFUND",

            "REVERSAL", "CASHBACK", "CASH BACK", "CBK", "INTEREST"

        ))

    def _looks_debit(self, description: str) -> bool:

        desc = re.sub(r"[^A-Z0-9]+", " ", str(description).upper()).strip()

        # A received payment is a credit, even when OCR puts PAYMENT first.

        if "PAYMENT RECEIVED" in desc or "RECEIVED PAYMENT" in desc:

            return False

        return any(term in desc for term in (

            "DEBIT", "ATM", "FEE", "CHARGE", "WITHDRAWAL", "PURCHASE",

            "PAYMENT TO", "PAYMENT FOR", "TRANSFER TO"

        ))

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        result = []

        for item in sorted(candidates, key=lambda x: (x["top"], x["center"])):

            duplicate = any(

                abs(item["center"] - existing["center"]) < 8

                and item["value"] == existing["value"]

                and abs(item["top"] - existing["top"]) < 12

                for existing in result

            )

            if not duplicate:

                result.append(item)

        return result

    def _normalize_header(self, text: str) -> str:

        value = str(text).strip().lower()

        value = re.sub(r"[^a-z ]", " ", value)

        return re.sub(r"\s+", " ", value).strip()

    def _header_semantic(self, text: str) -> Optional[str]:

        for semantic, aliases in self.HEADER_ALIASES.items():

            if text in aliases:

                return semantic

        return None

    def _starts_with_date(self, text: str) -> bool:

        parts = str(text).strip().split()

        if not parts:

            return False

        return self._is_date(parts[0])

    def _clean_date(self, text: str) -> str:

        return str(text).strip().strip("|,[]*\\(*){}")

    def _is_date(self, text: str) -> bool:

        return bool(self.DATE_RE.match(self._clean_date(text)))

    def _clean_number(self, text: str) -> str:

        return (

            str(text)

            .strip()

            .strip("|[]*\\(*){};:")

            .replace("Ã¢â€šÂ¹", "")

            .replace("$", "")

            .replace("Ã¢â€šÂ¬", "")

            .replace("Ã‚Â£", "")

            .strip()

        )

    def _is_number(self, text: str) -> bool:

        return bool(self.NUMBER_RE.match(self._clean_number(text)))

    def _to_float(self, value: Any) -> Optional[float]:

        if value is None:

            return None

        try:

            return float(self._clean_number(str(value)).replace(",", ""))

        except Exception:

            return None
