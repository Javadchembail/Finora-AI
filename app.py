
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from decimal import Decimal
from difflib import SequenceMatcher
from html import escape

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from ingestion.pipeline import FinancialStatementPipeline
from transactions.models import Transaction, TransactionDirection
from ai.rag_service import StatementRAG
from learning.category_memory import HybridCategoryEngine, CategoryMemory, merchant_key
from reports.financial_report import build_finora_report

load_dotenv()

st.set_page_config(
    page_title="Finora AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "page": "Home",
    "transactions": [],
    "file_name": None,
    "chat_history": [],
    "ai_summary": None,
    "transaction_focus": None,
    "category_engine": None,
    "transactions_backup": [],
    "category_review_skipped": False,
    "upload_mode": False,
    "open_category_editor": False,
    "statement_metadata": {},
    "category_icon_overrides": {},
    "pending_category_campaigns": [],
    "category_editor_version": 0,
    "statement_id": None,
    "statement_source_path": None,
    "review_cursor": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def _statement_context_path(statement_id=None):
    """Return a statement-scoped context file so tabs/statements cannot overwrite each other."""
    os.makedirs("storage", exist_ok=True)
    sid = str(statement_id or st.session_state.get("statement_id") or "").strip()
    if not sid:
        return os.path.join("storage", "finora_statement_context.json")
    return os.path.join("storage", f"finora_statement_context_{sid}.json")


def _json_safe(value):
    """Convert common metadata values into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_statement_context():
    """Persist only statement context needed to rebuild the live dashboard."""
    payload = {
        "statement_id": st.session_state.get("statement_id"),
        "file_name": st.session_state.get("file_name"),
        "statement_source_path": st.session_state.get("statement_source_path"),
        "statement_metadata": _json_safe(
            st.session_state.get("statement_metadata") or {}
        ),
        "category_icon_overrides": _json_safe(
            st.session_state.get("category_icon_overrides") or {}
        ),
    }
    try:
        with open(_statement_context_path(), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_statement_context():
    try:
        path = _statement_context_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _parse_decimal_text(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d{1,2})?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _extract_bank_balance_metadata(file_path, password=None):
    """
    Recover authoritative bank-statement opening/closing balances directly
    from the source PDF.

    Important distinction:
        net movement    = total credits - total debits
        closing balance = opening balance + net movement

    A bank statement may also print a running balance on every transaction
    row. When available, the final transaction-row balance is treated as the
    statement's authoritative closing balance and is checked against the
    calculated balance.

    The parser intentionally does NOT trust an arbitrary ``closing_balance``
    value produced by the transaction pipeline because different PDF layouts
    can cause a generic extractor to map the wrong numeric field.
    """
    result = {}

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            if not password:
                return result
            if reader.decrypt(str(password)) == 0:
                return result

        page_texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        print(f"Finora bank-balance parser warning: {exc}")
        return result

    amount_re = re.compile(
        r"(?<!\d)(?:₹|rs\.?|inr|aed|usd|eur|gbp)?\s*"
        r"-?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d{1,2})?\b",
        re.IGNORECASE,
    )

    date_start_re = re.compile(
        r"^\d{1,2}[-/]?[A-Z]{3}[-/]?\d{4}\b|"
        r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        re.IGNORECASE,
    )
    credit_debit_tail_re = re.compile(
        r"(?P<amount>-?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d{1,2})?)\s*"
        r"(?P<side>Cr|Dr)\s*$",
        re.IGNORECASE,
    )

    def parse_amounts(text):
        values = []
        for match in amount_re.finditer(str(text or "")):
            raw = match.group(0).strip()
            clean = raw.replace(",", "")
            # Do not treat years as money.
            if re.fullmatch(r"\d{4}", clean):
                continue
            value = _parse_decimal_text(raw)
            if value is not None:
                values.append(value)
        return values

    lines = []
    for page_text in page_texts:
        for raw_line in str(page_text).splitlines():
            line = " ".join(raw_line.split()).strip()
            if line:
                lines.append(line)

    opening_patterns = (
        re.compile(r"\bopening\s+(?:account\s+)?balance\b", re.I),
        re.compile(r"\bbeginning\s+(?:account\s+)?balance\b", re.I),
        re.compile(r"\bbalance\s+forward\b", re.I),
        re.compile(r"\bbrought\s+forward\b", re.I),
    )
    closing_patterns = (
        re.compile(r"\bclosing\s+(?:account\s+)?balance\b", re.I),
        re.compile(r"\bending\s+(?:account\s+)?balance\b", re.I),
        re.compile(r"\bfinal\s+(?:account\s+)?balance\b", re.I),
        re.compile(r"\bbalance\s+at\s+(?:the\s+)?end\b", re.I),
        re.compile(r"\bend(?:ing)?\s+balance\b", re.I),
    )

    # 1. Explicit opening/closing labels.
    for i, line in enumerate(lines):
        if "opening_balance" not in result and any(p.search(line) for p in opening_patterns):
            values = parse_amounts(line)
            if values:
                result["opening_balance"] = values[-1]
            else:
                for nearby in lines[i + 1:i + 4]:
                    values = parse_amounts(nearby)
                    if values:
                        result["opening_balance"] = values[-1]
                        break

        if "closing_balance" not in result and any(p.search(line) for p in closing_patterns):
            values = parse_amounts(line)
            if values:
                result["closing_balance"] = values[-1]
            else:
                for nearby in lines[i + 1:i + 4]:
                    values = parse_amounts(nearby)
                    if values:
                        result["closing_balance"] = values[-1]
                        break

    # 2. Opening Balance + Closing Balance compact summary layouts.
    for i, line in enumerate(lines):
        if not any(p.search(line) for p in opening_patterns):
            continue
        if not any(p.search(line) for p in closing_patterns):
            continue

        values = parse_amounts(line)
        for nearby in lines[i + 1:i + 4]:
            values.extend(parse_amounts(nearby))

        if len(values) >= 2:
            result.setdefault("opening_balance", values[-2])
            result.setdefault("closing_balance", values[-1])

    # 3. Authoritative final transaction-row balance.
    #    Typical Federal/Indian bank layout ends a row with:
    #       ... 2339.71 Cr
    #    We only accept a value when the same line starts like a transaction
    #    date. This avoids accidentally taking a GRAND TOTAL amount.
    final_row_balance = None
    # Scan backwards from GRAND TOTAL (or the end of the statement). This is
    # more reliable than requiring the date and balance to be on the same
    # extracted PDF line because many PDFs split transaction rows across
    # multiple text lines.
    scan_lines = lines
    for idx, line in enumerate(lines):
        if re.search(r"\bGRAND\s+TOTAL\b", line, re.I):
            scan_lines = lines[:idx]
            break

    for line in reversed(scan_lines):
        tail = credit_debit_tail_re.search(line)
        if tail:
            value = _parse_decimal_text(tail.group("amount"))
            if value is not None:
                final_row_balance = value
                break

    if final_row_balance is not None:
        result["final_transaction_balance"] = final_row_balance
        # The final transaction balance is stronger evidence than a generic
        # pipeline metadata field. It can be promoted to closing_balance later
        # after reconciliation with opening + net movement.
        result["closing_balance"] = final_row_balance

    # 4. Some statements print an explicit available/current balance. Keep it
    # as a separate fallback signal; do not blindly call it closing balance.
    for line in lines:
        if re.search(r"\beffective\s+available\s+balance\b", line, re.I):
            values = parse_amounts(line)
            if values:
                result["effective_available_balance"] = values[-1]
                break

    return result

def _extract_credit_card_summary(file_path, password=None):
    """
    Extract common credit-card statement summary fields from a wide range of
    text-based PDF layouts.

    The parser is intentionally independent from FinancialStatementPipeline:
    it reads the statement itself, detects the summary-header columns, then
    maps the compact numeric summary row to those columns. It also has
    label/value and explicit opening-balance fallbacks.
    """
    field_aliases = [
        ("card_limit", (
            r"\bcard\s+limit\b",
            r"\bcredit\s+limit\b",
        )),
        ("available_limit", (
            r"\bavailable\s+limit\b",
            r"\bavailable\s+credit\b",
            r"\bcredit\s+available\b",
        )),
        ("minimum_payment_due", (
            r"\bminimum\s+(?:payment|amount)\s+due\b",
            r"\bminimum\s+due\b",
        )),
        ("payment_due_date", (
            r"\bpayment\s+due\s+date\b",
            r"\bdue\s+date\b",
        )),
        ("total_payment_due", (
            r"\btotal\s+(?:payment|amount)\s+due\b",
            r"\btotal\s+due\b",
        )),
        ("profit_other_charges", (
            r"\bprofit\s*/?\s*other\s+charges\b",
            r"\binterest\s*/?\s*other\s+charges\b",
            r"\bfinance\s+charges\b",
            r"\binterest\s+charges\b",
        )),
        ("current_balance", (
            r"\bcurrent\s+balance\b",
            r"\bclosing\s+balance\b",
            r"\bstatement\s+balance\b",
            r"\boutstanding\s+balance\b",
            r"\btotal\s+outstanding\b",
        )),
        ("opening_balance", (
            r"\bopening\s+balance\b",
            r"\bprevious\s+balance\b",
            r"\bprior\s+balance\b",
            r"\bbeginning\s+balance\b",
        )),
    ]

    amount_or_date_re = re.compile(
        r"(?<!\d)"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|[-(]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\)?"
        r"|[-(]?\d+(?:\.\d{1,2})?\)?)"
        r"(?!\d)"
    )
    date_re = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")

    def _normalise_lines(text):
        return [
            " ".join(str(line).split())
            for line in str(text or "").splitlines()
            if str(line).strip()
        ]

    def _field_hits(line):
        low = str(line).casefold()
        hits = []
        for key, patterns in field_aliases:
            if any(re.search(pattern, low) for pattern in patterns):
                hits.append(key)
        return hits

    def _tokens(line):
        return amount_or_date_re.findall(str(line))

    def _parse_token(token):
        token = str(token).strip()
        negative = token.startswith("(") and token.endswith(")")
        token = token.strip("()").replace(",", "")
        try:
            value = float(token)
            return -value if negative else value
        except Exception:
            return None

    def _compact_numeric_row(line):
        """
        Return tokens only when the line is a table-like numeric row.

        This deliberately rejects prose such as 'AED 261.45 per month...'
        and transaction rows such as '23 JUL ... 25.50'.
        """
        raw = str(line or "").strip()
        tokens = _tokens(raw)
        if len(tokens) < 2:
            return []

        residual = raw
        for token in tokens:
            residual = residual.replace(token, " ", 1)

        residual = re.sub(r"[\(\)\[\]:,./-]", " ", residual)
        words = re.findall(r"[A-Za-z\u0600-\u06ff]+", residual)

        # A summary row is almost entirely numbers/dates. Allow a currency
        # marker such as AED/USD but reject normal prose.
        allowed_words = {
            "aed", "usd", "eur", "gbp", "qar", "sar", "kwd", "bhd",
            "omr", "inr", "jpy", "cad", "aud", "chf",
        }
        meaningful_words = [
            word.casefold()
            for word in words
            if word.casefold() not in allowed_words
        ]

        if len(meaningful_words) > 0:
            # A date-bearing summary row may contain a small amount of text,
            # but a continuation amount row must be numbers only.
            has_date = any(
                date_re.fullmatch(str(token))
                for token in tokens
            )
            if has_date and len(meaningful_words) <= 2:
                return tokens
            return []

        return tokens

    def _assign_row(summary, header_order, tokens):
        if not header_order or not tokens:
            return 0

        # Normal case: same number of columns.
        candidate_orders = [list(header_order)]

        # Opening balance is commonly printed elsewhere rather than in the
        # summary row. Treat it as an optional column when one value is absent.
        optional_drop_order = (
            "opening_balance",
            "profit_other_charges",
            "minimum_payment_due",
        )
        for optional_key in optional_drop_order:
            if optional_key in header_order:
                reduced = [
                    key for key in header_order
                    if key != optional_key
                ]
                if len(reduced) == len(tokens):
                    candidate_orders.insert(0, reduced)

        # Also allow a row containing only the first N columns.
        if len(tokens) < len(header_order):
            candidate_orders.append(header_order[:len(tokens)])

        best = None
        best_score = -1

        for order in candidate_orders:
            if len(order) != len(tokens):
                continue

            date_positions = [
                idx for idx, token in enumerate(tokens)
                if date_re.fullmatch(str(token))
            ]
            expected_date_positions = [
                idx for idx, key in enumerate(order)
                if key == "payment_due_date"
            ]

            score = 0
            if expected_date_positions and date_positions:
                distance = abs(
                    expected_date_positions[0] - date_positions[0]
                )
                score += 20 if distance == 0 else max(0, 10 - distance)

            # Prefer mappings that produce valid numeric fields.
            numeric_count = sum(
                1
                for key, token in zip(order, tokens)
                if key != "payment_due_date"
                and not date_re.fullmatch(str(token))
                and _parse_token(token) is not None
            )
            score += numeric_count * 2

            if score > best_score:
                best_score = score
                best = order

        if best is None:
            return 0

        added = 0
        for key, token in zip(best, tokens):
            if key == "payment_due_date":
                if date_re.fullmatch(str(token)):
                    summary[key] = str(token).replace("-", "/")
                    added += 1
            else:
                value = _parse_token(token)
                if value is not None:
                    summary[key] = value
                    added += 1

        return added

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            if not password:
                return {}
            decrypt_result = reader.decrypt(str(password))
            if decrypt_result == 0:
                return {}

        page_texts = [
            page.extract_text() or ""
            for page in reader.pages
        ]
    except Exception as exc:
        print(f"Finora credit-card summary parser warning: {exc}")
        return {}

    summary = {"statement_type": "credit_card"}

    # Parse page-by-page. This prevents a summary row from one page being
    # accidentally paired with headers or transactions from another page.
    for page_text in page_texts:
        lines = _normalise_lines(page_text)
        if not lines:
            continue

        # ------------------------------------------------------------
        # Strategy 1: detect a table header and its compact value row.
        # ------------------------------------------------------------
        # Find local clusters of English summary labels. PDF extractors may
        # place Arabic and English labels on alternating lines, so we collect
        # all recognised labels first and then group nearby occurrences.
        label_occurrences = []
        for line_index, line in enumerate(lines):
            for key in _field_hits(line):
                label_occurrences.append((line_index, key))

        for occurrence_index, (first_label_index, _) in enumerate(
            label_occurrences
        ):
            cluster = []
            for label_index, key in label_occurrences[occurrence_index:]:
                if label_index - first_label_index > 31:
                    break
                cluster.append((label_index, key))

            header_order = []
            label_indices = []
            for label_index, key in cluster:
                if key not in header_order:
                    header_order.append(key)
                    label_indices.append(label_index)

            if len(header_order) < 3:
                continue

            search_start = label_indices[-1] + 1

            # Look for a table-like numeric row. Give a date-bearing row
            # preference because payment-due date is a strong column anchor.
            candidates = []
            for j in range(
                search_start,
                min(search_start + 90, len(lines)),
            ):
                tokens = _compact_numeric_row(lines[j])
                if len(tokens) < 3:
                    continue

                has_date = any(
                    date_re.fullmatch(str(token))
                    for token in tokens
                )

                # Ignore short transaction-like rows. Summary rows usually
                # contain at least 4 values, or a date plus 2+ values.
                if len(tokens) >= 4 or (has_date and len(tokens) >= 3):
                    score = len(tokens) * 2 + (20 if has_date else 0)
                    candidates.append((score, j, tokens))

            candidates.sort(
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )

            assigned_row_index = None

            for _, row_index, tokens in candidates[:8]:
                before = set(summary.keys())
                _assign_row(summary, header_order, tokens)
                added = len(set(summary.keys()) - before)

                if added >= 3:
                    assigned_row_index = row_index
                    break

            # Some statements split the summary across two compact rows.
            # Example: the first row contains limit/available/minimum/date/
            # total, while the next numeric-only row contains profit/current.
            if assigned_row_index is not None:
                remaining_keys = [
                    key for key in header_order
                    if key not in summary
                    and key != "opening_balance"
                ]

                if remaining_keys:
                    for j in range(
                        assigned_row_index + 1,
                        min(assigned_row_index + 12, len(lines)),
                    ):
                        continuation = _compact_numeric_row(lines[j])
                        if not continuation:
                            continue
                        if any(
                            date_re.fullmatch(str(token))
                            for token in continuation
                        ):
                            continue

                        before = set(summary.keys())
                        for key, token in zip(
                            remaining_keys,
                            continuation,
                        ):
                            value = _parse_token(token)
                            if value is not None:
                                summary[key] = value
                        if len(set(summary.keys()) - before) > 0:
                            break

            if len(summary) >= 4:
                break

        # ------------------------------------------------------------
        # Strategy 2: explicit label/value pairs.
        # ------------------------------------------------------------
        for key, patterns in field_aliases:
            if key in summary:
                continue

            for i, line in enumerate(lines):
                low = line.casefold()
                if not any(re.search(pattern, low) for pattern in patterns):
                    continue

                # First accept a value on the same line as the label.
                same_line_tokens = _tokens(line)

                if key == "payment_due_date":
                    date_match = next(
                        (
                            token for token in same_line_tokens
                            if date_re.fullmatch(str(token))
                        ),
                        None,
                    )
                    if date_match:
                        summary[key] = str(date_match).replace("-", "/")
                        break
                else:
                    numeric_tokens = [
                        token for token in same_line_tokens
                        if not date_re.fullmatch(str(token))
                    ]
                    for token in numeric_tokens:
                        value = _parse_token(token)
                        if value is not None:
                            summary[key] = value
                            break

                    if key in summary:
                        break

                # Otherwise accept only a nearby numeric-only line. Never
                # scan arbitrary prose, because warning/fee text can contain
                # unrelated numbers such as penalty amounts.
                for next_index in range(
                    i + 1,
                    min(i + 5, len(lines)),
                ):
                    nearby = _compact_numeric_row(lines[next_index])
                    if not nearby:
                        continue

                    if key == "payment_due_date":
                        date_match = next(
                            (
                                token for token in nearby
                                if date_re.fullmatch(str(token))
                            ),
                            None,
                        )
                        if date_match:
                            summary[key] = str(date_match).replace("-", "/")
                            break
                    else:
                        numeric_tokens = [
                            token for token in nearby
                            if not date_re.fullmatch(str(token))
                        ]
                        if numeric_tokens:
                            value = _parse_token(numeric_tokens[0])
                            if value is not None:
                                summary[key] = value
                                break

                if key in summary:
                    break

        # ------------------------------------------------------------
        # Strategy 3: explicit opening/previous balance line.
        # ------------------------------------------------------------
        for key in (
            "opening_balance",
            "current_balance",
        ):
            if key in summary:
                continue

            for i, line in enumerate(lines):
                hits = _field_hits(line)
                if key not in hits:
                    continue

                tokens = _tokens(line)
                numeric_tokens = [
                    token for token in tokens
                    if not date_re.fullmatch(str(token))
                ]
                if numeric_tokens:
                    value = _parse_token(numeric_tokens[-1])
                    if value is not None:
                        summary[key] = value
                        break

    # A credit-card statement should expose at least a few of the summary
    # fields before we classify it as such. Never invent zeroes.
    real_fields = [
        key for key in summary
        if key != "statement_type"
    ]

    return summary if len(real_fields) >= 3 else {}

def _restore_statement_context():
    """Restore filename/metadata after a Streamlit reconnect."""
    if st.session_state.get("upload_mode"):
        return

    payload = _load_statement_context()
    if payload:
        if not st.session_state.get("statement_id"):
            st.session_state.statement_id = payload.get("statement_id")
        if not st.session_state.get("file_name"):
            st.session_state.file_name = payload.get("file_name")
        if not st.session_state.get("statement_source_path"):
            st.session_state.statement_source_path = payload.get("statement_source_path")
        if not st.session_state.get("statement_metadata"):
            metadata = payload.get("statement_metadata")
            if isinstance(metadata, dict):
                st.session_state.statement_metadata = metadata
        overrides = payload.get("category_icon_overrides")
        if isinstance(overrides, dict) and not st.session_state.get("category_icon_overrides"):
            st.session_state.category_icon_overrides = overrides

    # Recover a missing filename from the canonical Transaction.source_file.
    if not st.session_state.get("file_name") and st.session_state.get("transactions"):
        first = st.session_state.transactions[0]
        source_file = str(getattr(first, "source_file", "") or "").strip()
        if source_file:
            st.session_state.file_name = os.path.basename(source_file)

    # If the current session has transactions and a local source PDF, rebuild
    # missing card metadata from the document without touching the pipeline.
    if st.session_state.get("file_name"):
        existing_metadata = dict(st.session_state.get("statement_metadata") or {})
        source_path = st.session_state.get("statement_source_path")
        if not source_path:
            source_path = os.path.join(
                "storage",
                os.path.basename(st.session_state.file_name),
            )

        # Always give the PDF parser a chance to refresh credit-card values.
        # Older sessions may have persisted pipeline defaults such as 0.0.
        if os.path.exists(source_path):
            parsed = _extract_credit_card_summary(source_path, password=None)
            if parsed:
                st.session_state.statement_metadata = {
                    **existing_metadata,
                    **parsed,
                }
                _save_statement_context()


def _ensure_restored_categories():
    """Rehydrate missing category values from the existing category engine."""
    transactions = list(st.session_state.get("transactions") or [])
    if not transactions:
        return
    needs_rebuild = any(
        not str(getattr(tx, "category", "") or "").strip()
        for tx in transactions
    )
    if not needs_rebuild:
        return
    try:
        engine = HybridCategoryEngine(CategoryMemory())
        engine.classify_transactions(transactions)
        st.session_state.transactions = transactions
        st.session_state.category_engine = engine
        _save_transaction_backup(transactions)
    except Exception as exc:
        print(f"Finora category restore warning: {exc}")


def _transaction_backup_path(statement_id=None):
    os.makedirs("storage", exist_ok=True)
    sid = str(statement_id or st.session_state.get("statement_id") or "").strip()
    if not sid:
        return None
    return os.path.join("storage", f"finora_transactions_{sid}.json")


def _save_transaction_backup(transactions):
    """Persist the current statement only, never a global last-statement backup."""
    payload = [
        transaction.model_dump(mode="json")
        for transaction in (transactions or [])
    ]
    st.session_state.transactions_backup = payload
    backup_path = _transaction_backup_path()
    if not backup_path:
        return
    try:
        with open(backup_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    except Exception:
        # The live session remains usable even if local persistence fails.
        pass


def _restore_transaction_backup():
    """Restore only the backup belonging to the active statement."""
    if st.session_state.get("transactions"):
        return st.session_state.transactions

    # A statement id is required. This deliberately prevents an old/global
    # backup from appearing when a new statement or a different browser tab
    # starts a fresh session.
    if not st.session_state.get("statement_id"):
        return []

    payload = st.session_state.get("transactions_backup") or []

    if not payload:
        try:
            backup_path = _transaction_backup_path()
            if backup_path and os.path.exists(backup_path):
                with open(backup_path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
        except Exception:
            payload = []

    if not payload:
        return []

    restored = []
    for item in payload:
        try:
            restored.append(Transaction.model_validate(item))
        except Exception:
            continue

    if restored:
        st.session_state.transactions = restored
        st.session_state.transactions_backup = payload

    return restored


# Recover the latest successful analysis before routing the page.
# The statement_id lives in the URL so a Streamlit reconnect can restore the
# correct statement without accidentally restoring another browser tab's data.
requested_page = st.query_params.get("page")
requested_statement_id = st.query_params.get("statement_id")
new_statement_request = st.query_params.get("new") == "1"

if requested_statement_id and not new_statement_request:
    st.session_state.statement_id = str(requested_statement_id)
elif (
    not requested_statement_id
    and not new_statement_request
    and st.session_state.get("statement_id")
):
    # Keep the active statement identity in the URL during internal navigation.
    # Native browser links can otherwise open a fresh Streamlit connection with
    # only ?page=Transactions, which would have no way to restore the correct
    # statement-scoped transaction backup.
    requested_statement_id = str(st.session_state.statement_id)
    st.query_params["statement_id"] = requested_statement_id

# Clicking the Finora AI brand always returns to the main upload/home screen.
# The explicit `new=1` flag also prevents the previous statement backup from
# being restored immediately after navigation.
if new_statement_request:
    st.session_state.transactions = []
    st.session_state.transactions_backup = []
    st.session_state.file_name = None
    st.session_state.statement_source_path = None
    st.session_state.statement_id = None
    st.session_state.statement_metadata = {}
    st.session_state.review_cursor = 0
    st.session_state.category_icon_overrides = {}
    st.session_state.category_editor_version = int(st.session_state.get("category_editor_version", 0)) + 1
    st.session_state.chat_history = []
    st.session_state.ai_summary = None
    st.session_state.category_engine = None
    st.session_state.transaction_focus = None
    st.session_state.category_review_skipped = False
    st.session_state.pending_category_campaigns = []
    st.session_state.upload_mode = True
    st.session_state.page = "Home"
    requested_page = "Home"
    st.query_params.clear()
    st.query_params["page"] = "Home"

# Only restore after the explicit new-statement reset has been processed.
if not st.session_state.get("upload_mode", False):
    _restore_transaction_backup()

# Restore statement context (filename + card metadata) after reconnects.
if not st.session_state.get("upload_mode", False):
    _restore_statement_context()
    _ensure_restored_categories()

# Normalize page state after code upgrades. Streamlit keeps session_state
# across hot-reloads, so an older Finora version can leave values such as
# Dashboard / Analytics / Upload Statement behind. If the value is not one
# of the routes used by this version, recover to a real route instead of
# rendering only the navigation bar.
VALID_PAGES = {"Home", "Overview", "Transactions", "AI", "Review", "Upload"}

current_page = st.session_state.get("page")
has_transactions = bool(st.session_state.get("transactions"))

if requested_page == "Upload":
    st.session_state.page = "Home"
elif requested_page in {"Overview", "Transactions", "AI", "Review"}:
    st.session_state.page = requested_page
elif current_page in {"Home", "Upload"} and has_transactions and not st.session_state.get("upload_mode", False):
    # A Streamlit reconnect can restore transactions from disk while the
    # session page is still the initial Home/Upload value. Never leave the
    # application between routes in that state; show the analyzed data.
    st.session_state.page = "Overview"
elif current_page not in VALID_PAGES:
    st.session_state.page = "Overview" if has_transactions else "Home"

# If the active statement exists in this session, keep its identity attached
# to the current browser URL. This makes Overview / Transactions / AI / Review
# navigation reconnect-safe without using a global last-statement pointer.
if (
    not st.session_state.get("upload_mode", False)
    and st.session_state.get("statement_id")
    and st.query_params.get("statement_id") != str(st.session_state.statement_id)
):
    st.query_params["statement_id"] = str(st.session_state.statement_id)


# ============================================================
# GLOBAL STYLE
# ============================================================

st.html("""
<style>
#MainMenu, footer { display:none !important; }
header { background:transparent !important; }

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(99,102,241,.14), transparent 25%),
        radial-gradient(circle at 5% 35%, rgba(14,165,233,.07), transparent 24%),
        #070b14;
    color:#f8fafc;
}

.block-container {
    max-width:1400px;
    padding:18px 34px 70px;
}

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:4px 0 22px;
}

.brand {
    display:flex;
    align-items:center;
    gap:11px;
}

.finora-brand-link {
    display:block;
    color:inherit !important;
    text-decoration:none !important;
    cursor:pointer;
}

.finora-brand-link:hover .brand-name {
    color:#ffffff;
}

.finora-brand-link:hover .logo {
    transform:translateY(-1px);
    box-shadow:0 14px 38px rgba(99,102,241,.36);
}

.finora-brand-link .logo {
    transition:transform .18s ease, box-shadow .18s ease;
}

.logo {
    width:48px; height:48px; display:flex; align-items:center; justify-content:center;
    border-radius:15px; background:linear-gradient(145deg,#7c3aed,#4f46e5);
    box-shadow:0 12px 34px rgba(99,102,241,.28); font-size:27px; font-weight:950;
    color:#fff; font-style:italic;
}
.finora-f-logo { letter-spacing:-3px; text-shadow:0 2px 14px rgba(255,255,255,.18); }

.brand-name {
    color:#fff;
    font-size:1.1rem;
    font-weight:850;
}

.brand-sub {
    color:#64748b;
    font-size:.62rem;
    margin-top:2px;
}

.ai-ready {
    color:#4ade80;
    font-size:.67rem;
    font-weight:750;
    padding:7px 11px;
    border-radius:999px;
    background:rgba(34,197,94,.07);
    border:1px solid rgba(34,197,94,.16);
}

.hero {
    position:relative;
    overflow:hidden;
    min-height:410px;
    border:1px solid #202c40;
    border-radius:28px;
    padding:58px;
    background:
        radial-gradient(circle at 88% 22%, rgba(99,102,241,.27), transparent 26%),
        radial-gradient(circle at 70% 90%, rgba(59,130,246,.11), transparent 28%),
        linear-gradient(135deg,#111827,#0a101c);
    box-shadow:0 30px 90px rgba(0,0,0,.28);
}

.eyebrow {
    color:#60a5fa;
    font-size:.66rem;
    font-weight:850;
    letter-spacing:1.7px;
    margin-bottom:13px;
}

.hero-title {
    max-width:780px;
    color:#fff;
    font-size:clamp(2.5rem,5vw,4.6rem);
    line-height:.98;
    letter-spacing:-3px;
    font-weight:900;
}

.hero-text {
    max-width:660px;
    color:#94a3b8;
    font-size:.97rem;
    line-height:1.7;
    margin-top:20px;
}

.hero-status {
    display:inline-flex;
    margin-top:21px;
    padding:8px 13px;
    border-radius:999px;
    color:#4ade80;
    background:rgba(34,197,94,.08);
    border:1px solid rgba(34,197,94,.18);
    font-size:.68rem;
    font-weight:750;
}

.ai-orbit {
    position:absolute;
    right:95px;
    top:90px;
    width:185px;
    height:185px;
    border-radius:50%;
    border:1px solid rgba(129,140,248,.22);
    display:flex;
    align-items:center;
    justify-content:center;
}

.ai-core {
    width:108px;
    height:108px;
    border-radius:30px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:45px;
    background:linear-gradient(145deg,rgba(99,102,241,.27),rgba(59,130,246,.11));
    border:1px solid rgba(129,140,248,.28);
    box-shadow:0 20px 55px rgba(79,70,229,.23);
}

.feature {
    min-height:150px;
    padding:22px;
    border:1px solid #1d293b;
    border-radius:18px;
    background:linear-gradient(145deg,#101827,#0c131f);
}

.feature-icon {
    font-size:23px;
    margin-bottom:12px;
}

.feature-title {
    color:#fff;
    font-size:.88rem;
    font-weight:800;
}

.feature-text {
    color:#64748b;
    font-size:.72rem;
    line-height:1.55;
    margin-top:7px;
}

.page-title {
    color:#fff;
    font-size:2.1rem;
    font-weight:900;
    letter-spacing:-1px;
    margin-top:25px;
}

.page-subtitle {
    color:#64748b;
    font-size:.82rem;
    margin-top:4px;
    margin-bottom:23px;
}

.metric {
    min-height:125px;
    padding:21px;
    border:1px solid #1d293b;
    border-radius:18px;
    background:linear-gradient(145deg,#101827,#0c131f);
}

.metric-label {
    color:#64748b;
    font-size:.63rem;
    font-weight:850;
    letter-spacing:.9px;
}

.metric-value {
    color:#fff;
    font-size:1.45rem;
    font-weight:850;
    margin-top:9px;
}

.metric-sub {
    color:#64748b;
    font-size:.67rem;
    margin-top:6px;
}

.insight {
    min-height:125px;
    padding:18px;
    border:1px solid #1d293b;
    border-radius:18px;
    background:linear-gradient(145deg,#101827,#0c131f);
}

.insight-icon {
    font-size:20px;
}

.insight-title {
    color:#fff;
    font-size:.78rem;
    font-weight:800;
    margin-top:8px;
}

.insight-value {
    color:#fff;
    font-size:.88rem;
    font-weight:800;
    margin-top:5px;
}

.insight-sub {
    color:#64748b;
    font-size:.69rem;
    margin-top:5px;
}

.upload-card {
    text-align:center;
    padding:38px 25px;
    border:1px dashed #334155;
    border-radius:22px;
    background:#0c131f;
}

.ai-panel {
    padding:24px;
    border:1px solid #26344b;
    border-radius:22px;
    background:
        radial-gradient(circle at 100% 0%,rgba(99,102,241,.15),transparent 33%),
        linear-gradient(135deg,#101827,#0c131f);
}

.ai-head {
    display:flex;
    align-items:center;
    gap:12px;
}

.ai-icon {
    width:41px;
    height:41px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:12px;
    background:rgba(99,102,241,.15);
    border:1px solid rgba(129,140,248,.2);
}

.ai-title {
    color:#fff;
    font-size:1rem;
    font-weight:850;
}

.ai-subtitle {
    color:#64748b;
    font-size:.68rem;
    margin-top:2px;
}

.chat-user {
    max-width:78%;
    margin:14px 0 8px auto;
    padding:12px 15px;
    border-radius:16px;
    background:#172238;
    border:1px solid #25344d;
    color:#dbeafe;
    font-size:.82rem;
}

.chat-ai {
    max-width:88%;
    margin:8px auto 14px 0;
    padding:14px 16px;
    border-radius:16px;
    background:#0f1725;
    border:1px solid #1e2a3d;
    color:#cbd5e1;
    font-size:.82rem;
    line-height:1.65;
}

.section-title, .intel-title { font-size:1.28rem !important; }
.section-subtitle, .intel-sub { font-size:.92rem !important; }
.focus-label, .breakdown-kicker, .action-number { font-size:.72rem !important; }
.focus-title, .breakdown-title, .action-title { font-size:1.15rem !important; }
.focus-amount, .breakdown-value { font-size:1.55rem !important; }
.focus-copy, .breakdown-copy, .action-copy { font-size:.9rem !important; line-height:1.65 !important; }
.rank-name, .rank-amount { font-size:1.03rem !important; }
.rank-sub { font-size:.86rem !important; }
.cockpit-metric-label { font-size:.72rem !important; }
.cockpit-metric-value { font-size:1.35rem !important; }
.cockpit-metric-sub { font-size:.78rem !important; }
div.st-key-finora_ai_popover { position:fixed !important; right:28px !important; bottom:28px !important; z-index:99999 !important; }
div.st-key-finora_ai_popover > div { width:68px !important; }
div.st-key-finora_ai_popover button { width:68px !important; height:68px !important; min-height:68px !important; border-radius:50% !important; border:1px solid rgba(167,139,250,.75) !important; background:linear-gradient(145deg,#7c3aed,#4f46e5) !important; color:#fff !important; font-size:1.7rem !important; box-shadow:0 12px 45px rgba(99,102,241,.42),0 0 0 7px rgba(99,102,241,.08) !important; }
div.st-key-finora_ai_popover button:hover { transform:translateY(-2px) scale(1.03) !important; }
.focus-filter-banner { margin:10px 0 18px; padding:13px 16px; border-radius:14px; background:rgba(99,102,241,.10); border:1px solid rgba(129,140,248,.22); color:#c7d2fe; font-size:.9rem; }
.focus-filter-banner span { color:#94a3b8; }

.stButton > button {
    border-radius:11px !important;
    border:1px solid #26354c !important;
    background:#101827 !important;
    color:#dbeafe !important;
    font-weight:700 !important;
}

.stButton > button:hover {
    border-color:#6366f1 !important;
    color:#fff !important;
}

.stButton > button[kind="primary"] {
    position:relative !important;
    min-height:48px !important;
    background:linear-gradient(110deg,#4f46e5,#6366f1,#7c3aed,#4f46e5) !important;
    background-size:260% 100% !important;
    border:1px solid rgba(129,140,248,.65) !important;
    color:#fff !important;
    font-weight:850 !important;
    box-shadow:0 12px 35px rgba(79,70,229,.20) !important;
    animation:primary-button-flow 4s ease infinite !important;
    transition:transform .2s ease, box-shadow .2s ease !important;
}

.stButton > button[kind="primary"]:hover {
    transform:translateY(-1px) !important;
    box-shadow:0 16px 42px rgba(79,70,229,.30) !important;
}

@keyframes primary-button-flow {
    0% { background-position:0% 50%; }
    50% { background-position:100% 50%; }
    100% { background-position:0% 50%; }
}

[data-testid="stFileUploader"] section {
    background:#0c131f !important;
    border:1px dashed #334155 !important;
    border-radius:18px !important;
}

[data-testid="stDataFrame"] {
    border-radius:15px;
    overflow:hidden;
}

hr {
    border-color:#1b2638 !important;
}


/* ============================================================
   FINORA LANDING PAGE — CENTERED UPLOAD DESIGN
   ============================================================ */

.landing-page {
    text-align:center;
    padding-top:18px;
}

.landing-visual {
    position:relative;
    width:390px;
    height:235px;
    margin:0 auto 2px;
}

.landing-document {
    position:absolute;
    left:50%;
    top:52%;
    transform:translate(-50%,-50%);
    width:104px;
    height:104px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:29px;
    background:linear-gradient(145deg,#6366f1,#4f46e5);
    border:1px solid rgba(165,180,252,.48);
    box-shadow:
        0 0 38px rgba(99,102,241,.36),
        0 0 90px rgba(79,70,229,.22),
        inset 0 1px 0 rgba(255,255,255,.22);
    z-index:4;
}

.document-sheet {
    width:47px;
    height:59px;
    border-radius:5px;
    background:#fff;
    position:relative;
    padding:14px 8px;
    box-shadow:0 8px 20px rgba(0,0,0,.18);
}

.document-sheet::after {
    content:"";
    position:absolute;
    right:0;
    top:0;
    width:14px;
    height:14px;
    background:#dbeafe;
    clip-path:polygon(0 0,100% 100%,0 100%);
}

.document-line {
    height:4px;
    width:27px;
    border-radius:99px;
    background:#6366f1;
    margin-top:7px;
}

.document-line-long { width:31px; margin-top:2px; }

.orbit {
    position:absolute;
    left:50%;
    top:50%;
    transform:translate(-50%,-50%);
    border-radius:50%;
    border:1px solid rgba(99,102,241,.16);
}

.orbit-1 { width:170px; height:170px; }
.orbit-2 { width:245px; height:245px; border-color:rgba(99,102,241,.09); }
.orbit-3 { width:315px; height:315px; border-color:rgba(99,102,241,.045); }

.landing-icon {
    position:absolute;
    width:54px;
    height:54px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:15px;
    font-size:25px;
    font-weight:900;
    z-index:5;
    box-shadow:0 12px 35px rgba(0,0,0,.22);
}

.icon-chart {
    left:48px;
    top:54px;
    color:#c4b5fd;
    background:rgba(30,41,90,.78);
    border:1px solid rgba(99,102,241,.45);
}

.icon-bank {
    right:45px;
    top:35px;
    color:#60a5fa;
    background:rgba(11,38,73,.72);
    border:1px solid rgba(59,130,246,.40);
}

.icon-card {
    left:102px;
    bottom:17px;
    color:#fda4af;
    background:rgba(61,25,45,.72);
    border:1px solid rgba(244,63,94,.35);
}

.icon-ai {
    right:89px;
    bottom:14px;
    color:#c4b5fd;
    background:rgba(45,24,85,.72);
    border:1px solid rgba(139,92,246,.38);
}

.landing-title {
    color:#f8fafc;
    font-size:clamp(2rem,3.3vw,3.05rem);
    line-height:1.1;
    letter-spacing:-1.5px;
    font-weight:900;
    margin-top:4px;
}

.landing-title span {
    color:#8b5cf6;
}

.landing-subtitle {
    color:#94a3b8;
    font-size:.88rem;
    line-height:1.55;
    margin:8px auto 18px;
    max-width:760px;
}

.landing-upload-card {
    width:min(440px,100%);
    margin:0 auto;
    padding:18px 22px 14px;
    text-align:center;
    border:1px dashed #64748b;
    border-radius:18px 18px 0 0;
    border-bottom:0;
    background:
        radial-gradient(circle at 50% 0%,rgba(99,102,241,.12),transparent 50%),
        #0b1422;
}

.landing-upload-icon {
    width:48px;
    height:48px;
    margin:0 auto 8px;
    display:flex;
    align-items:center;
    justify-content:center;
    border-radius:13px;
    color:#fff;
    font-size:25px;
    font-weight:900;
    background:linear-gradient(145deg,#263a68,#1d2c52);
    border:1px solid #344d7b;
}

.landing-upload-title {
    color:#f8fafc;
    font-size:.9rem;
    font-weight:850;
}

.landing-upload-subtitle {
    color:#64748b;
    font-size:.64rem;
    margin-top:4px;
}

.landing-upload-card + div {
    width:min(440px,100%);
    margin:0 auto;
}

[data-testid="stFileUploader"] {
    width:min(440px,100%);
    margin:0 auto !important;
}

[data-testid="stFileUploader"] section {
    min-height:60px !important;
    padding:9px 13px !important;
    border:1px dashed #334155 !important;
    border-top:0 !important;
    border-radius:0 0 18px 18px !important;
    background:#0b1422 !important;
}

[data-testid="stFileUploader"] section > div:first-child {
    min-height:42px !important;
}

[data-testid="stFileUploader"] button {
    border:1px solid rgba(99,102,241,.5) !important;
    background:linear-gradient(110deg,#4f46e5,#6366f1) !important;
    color:#fff !important;
    font-weight:800 !important;
}

.landing-file-selected {
    width:min(440px,100%);
    margin:8px auto 0;
    padding:8px 11px;
    display:flex;
    align-items:center;
    gap:8px;
    border-radius:10px;
    background:rgba(34,197,94,.06);
    border:1px solid rgba(34,197,94,.16);
    color:#cbd5e1;
    font-size:.67rem;
    text-align:left;
}

.file-dot {
    width:7px;
    height:7px;
    flex:0 0 7px;
    border-radius:50%;
    background:#4ade80;
    box-shadow:0 0 10px rgba(74,222,128,.5);
}

.file-ready {
    margin-left:auto;
    color:#4ade80;
    font-size:.56rem;
    font-weight:850;
}

.landing-security {
    width:min(440px,100%);
    margin:6px auto 0;
    color:#526176;
    font-size:.58rem;
    text-align:center;
}

.landing-security span { color:#94a3b8; margin-right:5px; }

.landing-section-kicker {
    margin-top:34px;
    color:#64748b;
    font-size:.58rem;
    font-weight:850;
    letter-spacing:1.6px;
    text-transform:uppercase;
}

.landing-section-title {
    margin-top:7px;
    color:#f8fafc;
    font-size:1.2rem;
    font-weight:900;
    letter-spacing:-.4px;
}

.landing-feature {
    min-height:135px;
    padding:20px 21px;
    text-align:left;
    border:1px solid #1b283b;
    border-radius:18px;
    background:linear-gradient(145deg,#0e1724,#0a111c);
}

.landing-feature-icon {
    color:#c4b5fd;
    font-size:21px;
    margin-bottom:10px;
}

.landing-feature-title {
    color:#f8fafc;
    font-size:.82rem;
    font-weight:850;
}

.landing-feature-text {
    color:#64748b;
    font-size:.66rem;
    line-height:1.55;
    margin-top:6px;
}

/* Make password input and analyze button match the centered uploader. */
div[data-testid="stTextInput"] {
    width:min(440px,100%) !important;
    margin:8px auto 0 !important;
}

div[data-testid="stTextInput"] input {
    min-height:42px !important;
    border-radius:11px !important;
    background:#10131b !important;
    border:1px solid #202b3d !important;
}

button[kind="primary"] {
    border-radius:11px !important;
}

@media (max-width:900px) {
    .landing-visual { transform:scale(.88); margin-bottom:-18px; }
    .landing-title { font-size:2rem; }
    .landing-subtitle { font-size:.8rem; }
    .desktop-only { display:none; }
}


.chart-card {
    position:relative;
    padding:18px 18px 12px;
    border:1px solid #1c293b;
    border-radius:22px;
    background:
        radial-gradient(circle at 90% 0%, rgba(99,102,241,.08), transparent 32%),
        linear-gradient(145deg,#0f1725,#0a111c);
    box-shadow:0 18px 50px rgba(0,0,0,.16);
    overflow:hidden;
}

.chart-card::before {
    content:"";
    position:absolute;
    inset:0;
    pointer-events:none;
    border-radius:22px;
    background:linear-gradient(
        120deg,
        rgba(255,255,255,.025),
        transparent 35%,
        rgba(99,102,241,.025)
    );
}

.chart-heading {
    position:relative;
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:16px;
    margin:2px 3px 6px;
}

.chart-heading-title {
    color:#f8fafc;
    font-size:.86rem;
    font-weight:850;
}

.chart-heading-sub {
    color:#64748b;
    font-size:.67rem;
    line-height:1.5;
    margin-top:4px;
}

.chart-badge {
    flex:0 0 auto;
    padding:6px 9px;
    border-radius:999px;
    color:#93c5fd;
    background:rgba(59,130,246,.08);
    border:1px solid rgba(59,130,246,.16);
    font-size:.59rem;
    font-weight:800;
}

.chart-note {
    color:#475569;
    font-size:.61rem;
    margin:2px 4px 4px;
}


.spending-intro {
    color:#94a3b8;
    font-size:.76rem;
    line-height:1.65;
    margin-top:-8px;
    margin-bottom:18px;
}

.breakdown-card {
    position:relative;
    padding:20px;
    min-height:128px;
    border:1px solid #1d293b;
    border-radius:20px;
    background:
        radial-gradient(circle at 100% 0%, rgba(99,102,241,.08), transparent 38%),
        linear-gradient(145deg,#101827,#0b121e);
}

.breakdown-kicker {
    color:#64748b;
    font-size:.59rem;
    font-weight:850;
    letter-spacing:1px;
    text-transform:uppercase;
}

.breakdown-title {
    color:#f8fafc;
    font-size:.9rem;
    font-weight:850;
    margin-top:8px;
}

.breakdown-value {
    color:#fff;
    font-size:1.18rem;
    font-weight:900;
    margin-top:5px;
}

.breakdown-detail {
    color:#64748b;
    font-size:.66rem;
    line-height:1.5;
    margin-top:6px;
}

.reduction-panel {
    position:relative;
    overflow:hidden;
    padding:24px;
    border:1px solid rgba(99,102,241,.22);
    border-radius:24px;
    background:
        radial-gradient(circle at 90% 10%,rgba(99,102,241,.16),transparent 32%),
        linear-gradient(135deg,#101827,#0b1220);
}

.reduction-panel::before {
    content:"";
    position:absolute;
    width:180px;
    height:180px;
    right:-100px;
    bottom:-110px;
    border-radius:50%;
    border:1px solid rgba(96,165,250,.12);
}

.reduction-title {
    color:#fff;
    font-size:1rem;
    font-weight:900;
}

.reduction-text {
    color:#94a3b8;
    font-size:.72rem;
    line-height:1.65;
    margin-top:6px;
    max-width:720px;
}

.reduction-amount {
    color:#a5b4fc;
    font-size:1.35rem;
    font-weight:900;
    margin-top:15px;
}

.reduction-label {
    color:#64748b;
    font-size:.62rem;
    margin-top:3px;
}

.coverage-pill {
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding:6px 9px;
    border-radius:999px;
    background:rgba(59,130,246,.08);
    border:1px solid rgba(59,130,246,.16);
    color:#93c5fd;
    font-size:.6rem;
    font-weight:800;
}


.story-grid {
    display:grid;
    grid-template-columns:1.05fr 1.55fr;
    gap:20px;
    margin-top:18px;
}

.story-card {
    position:relative;
    overflow:hidden;
    border:1px solid #202d42;
    border-radius:22px;
    background:linear-gradient(145deg,#0f1725,#0a111c);
    padding:23px;
    box-shadow:0 18px 50px rgba(0,0,0,.14);
}

.story-card-title {
    color:#f8fafc;
    font-size:.88rem;
    font-weight:900;
}

.story-card-sub {
    color:#64748b;
    font-size:.66rem;
    line-height:1.5;
    margin-top:4px;
}

.story-number {
    color:#fff;
    font-size:1.65rem;
    font-weight:900;
    letter-spacing:-1px;
    margin-top:18px;
}

.story-muted {
    color:#64748b;
    font-size:.62rem;
    margin-top:3px;
}

.money-row {
    margin-top:17px;
}

.money-row-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:12px;
    margin-bottom:7px;
}

.money-row-name {
    color:#e2e8f0;
    font-size:.68rem;
    font-weight:750;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    max-width:68%;
}

.money-row-value {
    color:#cbd5e1;
    font-size:.66rem;
    font-weight:800;
}

.money-track {
    height:7px;
    overflow:hidden;
    border-radius:999px;
    background:#172236;
}

.money-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#6366f1,#60a5fa);
    box-shadow:0 0 16px rgba(96,165,250,.16);
}

.money-rank {
    color:#475569;
    font-size:.57rem;
    margin-top:5px;
}

.insight-strip {
    display:grid;
    grid-template-columns:1fr 1fr 1fr;
    gap:14px;
    margin-top:18px;
}

.insight-item {
    border:1px solid #1d2a3e;
    border-radius:18px;
    background:#0d1522;
    padding:17px;
}

.insight-kicker {
    color:#64748b;
    font-size:.56rem;
    font-weight:850;
    letter-spacing:1.2px;
}

.insight-title {
    color:#f8fafc;
    font-size:.76rem;
    line-height:1.35;
    font-weight:850;
    margin-top:7px;
}

.insight-copy {
    color:#64748b;
    font-size:.62rem;
    line-height:1.55;
    margin-top:5px;
}

.coverage-line {
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:12px;
    margin-top:15px;
    color:#64748b;
    font-size:.60rem;
}

.coverage-track {
    height:5px;
    border-radius:999px;
    overflow:hidden;
    background:#172236;
    margin-top:7px;
}

.coverage-fill {
    height:100%;
    border-radius:999px;
    background:#60a5fa;
}


/* ============================================================
   FINORA 2.0 — STORY-FIRST OVERVIEW
   ============================================================ */

.story-hero {
    position:relative;
    overflow:hidden;
    padding:28px 30px;
    border:1px solid #1f2b40;
    border-radius:26px;
    background:
        radial-gradient(circle at 88% 10%, rgba(99,102,241,.13), transparent 32%),
        linear-gradient(145deg,#101827,#0a111b);
    box-shadow:0 20px 60px rgba(0,0,0,.16);
}

.story-hero::after {
    content:"";
    position:absolute;
    width:210px;
    height:210px;
    right:-80px;
    top:-105px;
    border-radius:50%;
    border:1px solid rgba(129,140,248,.12);
    box-shadow:
        0 0 0 30px rgba(129,140,248,.025),
        0 0 0 60px rgba(129,140,248,.018);
}

.story-kicker {
    color:#818cf8;
    font-size:.59rem;
    font-weight:850;
    letter-spacing:1.6px;
}

.story-hero-title {
    margin-top:8px;
    color:#f8fafc;
    font-size:1.45rem;
    line-height:1.18;
    font-weight:900;
    letter-spacing:-.7px;
}

.story-hero-copy {
    max-width:760px;
    margin-top:8px;
    color:#7f8da3;
    font-size:.72rem;
    line-height:1.6;
}

.story-hero-insight {
    margin-top:18px;
    color:#cbd5e1;
    font-size:.75rem;
    line-height:1.55;
}

.story-hero-insight strong {
    color:#fff;
}

.summary-strip {
    display:grid;
    grid-template-columns:1.25fr 1.25fr 1.25fr .9fr;
    gap:12px;
    margin-top:14px;
}

.summary-item {
    padding:18px 19px;
    border:1px solid #1b283b;
    border-radius:18px;
    background:#0d1522;
}

.summary-label {
    color:#64748b;
    font-size:.54rem;
    font-weight:850;
    letter-spacing:1.1px;
    text-transform:uppercase;
}

.summary-value {
    margin-top:7px;
    color:#f8fafc;
    font-size:1.04rem;
    font-weight:900;
}

.summary-sub {
    margin-top:4px;
    color:#526176;
    font-size:.58rem;
}

.section-head {
    display:flex;
    align-items:end;
    justify-content:space-between;
    gap:20px;
    margin:34px 0 12px;
}

.section-title {
    color:#f8fafc;
    font-size:1.02rem;
    font-weight:900;
    letter-spacing:-.2px;
}

.section-sub {
    margin-top:4px;
    color:#64748b;
    font-size:.64rem;
    line-height:1.45;
}

.section-meta {
    color:#64748b;
    font-size:.58rem;
    text-align:right;
}

.spending-panel,
.flow-panel,
.ai-read-panel {
    border:1px solid #1c293b;
    border-radius:23px;
    background:
        linear-gradient(145deg,#0e1724,#0a111c);
    box-shadow:0 18px 50px rgba(0,0,0,.14);
}

.spending-panel {
    padding:21px 22px 16px;
}

.spending-row {
    padding:13px 0 14px;
    border-bottom:1px solid rgba(51,65,85,.28);
}

.spending-row:last-child {
    border-bottom:0;
}

.spending-row-head {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
}

.spending-rank {
    width:24px;
    color:#475569;
    font-size:.59rem;
    font-weight:850;
}

.spending-name {
    flex:1;
    min-width:0;
    color:#e2e8f0;
    font-size:.68rem;
    font-weight:800;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}

.spending-name span {
    display:block;
    margin-top:3px;
    color:#4f6075;
    font-size:.53rem;
    font-weight:500;
}

.spending-amount {
    color:#f8fafc;
    font-size:.67rem;
    font-weight:850;
    white-space:nowrap;
}

.spending-track {
    height:5px;
    margin:8px 0 0 24px;
    overflow:hidden;
    border-radius:999px;
    background:#172235;
}

.spending-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#4f46e5,#60a5fa);
}

.spending-foot {
    display:flex;
    justify-content:space-between;
    margin:5px 0 0 24px;
    color:#46566b;
    font-size:.51rem;
}

.flow-panel {
    padding:21px 20px 10px;
}

.ai-read-panel {
    padding:22px 24px;
}

.ai-read-main {
    color:#e2e8f0;
    font-size:.84rem;
    line-height:1.65;
    font-weight:650;
}

.ai-read-main strong {
    color:#fff;
}

.ai-read-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:12px;
    margin-top:17px;
}

.ai-read-item {
    padding:14px;
    border-radius:15px;
    border:1px solid #1c293b;
    background:#0b1320;
}

.ai-read-label {
    color:#64748b;
    font-size:.53rem;
    font-weight:850;
    letter-spacing:.9px;
}

.ai-read-value {
    margin-top:7px;
    color:#e2e8f0;
    font-size:.67rem;
    line-height:1.45;
    font-weight:750;
}

.review-panel {
    padding:21px 23px;
    border:1px solid rgba(129,140,248,.16);
    border-radius:22px;
    background:
        radial-gradient(circle at 90% 0%,rgba(99,102,241,.09),transparent 32%),
        #0d1522;
}

.review-title {
    color:#f8fafc;
    font-size:.84rem;
    font-weight:850;
}

.review-copy {
    margin-top:6px;
    color:#68788e;
    font-size:.63rem;
    line-height:1.55;
}

.review-number {
    margin-top:15px;
    color:#a5b4fc;
    font-size:1.18rem;
    font-weight:900;
}

.review-note {
    margin-top:3px;
    color:#526176;
    font-size:.55rem;
}


/* ============================================================
   FINORA FINANCIAL COCKPIT
   ============================================================ */

.finora-cockpit {
    position:relative;
    overflow:hidden;
    padding:30px 32px 28px;
    border:1px solid #202d42;
    border-radius:28px;
    background:
        radial-gradient(circle at 88% 12%,rgba(99,102,241,.15),transparent 30%),
        radial-gradient(circle at 8% 100%,rgba(14,165,233,.07),transparent 28%),
        linear-gradient(145deg,#101827,#090f19);
    box-shadow:0 24px 70px rgba(0,0,0,.20);
}

.cockpit-kicker {
    color:#818cf8;
    font-size:.58rem;
    font-weight:900;
    letter-spacing:1.7px;
}

.cockpit-title {
    margin-top:8px;
    color:#f8fafc;
    font-size:1.65rem;
    line-height:1.12;
    font-weight:900;
    letter-spacing:-.8px;
}

.cockpit-copy {
    max-width:720px;
    margin-top:9px;
    color:#64748b;
    font-size:.69rem;
    line-height:1.6;
}

.cockpit-story {
    margin-top:20px;
    max-width:880px;
    color:#dbeafe;
    font-size:.82rem;
    line-height:1.65;
}

.cockpit-story strong {
    color:#fff;
}

.cockpit-pill {
    display:inline-flex;
    align-items:center;
    gap:6px;
    margin-top:16px;
    padding:6px 10px;
    border:1px solid rgba(74,222,128,.18);
    border-radius:999px;
    background:rgba(74,222,128,.06);
    color:#86efac;
    font-size:.57rem;
    font-weight:850;
}

.cockpit-pill span {
    width:5px;
    height:5px;
    border-radius:50%;
    background:#4ade80;
    box-shadow:0 0 9px rgba(74,222,128,.55);
}

.cockpit-metrics {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:10px;
    margin-top:18px;
}

.cockpit-metric {
    padding:15px 16px;
    border:1px solid #1c293b;
    border-radius:16px;
    background:rgba(7,13,23,.42);
}

.cockpit-metric-label {
    color:#526176;
    font-size:.52rem;
    font-weight:850;
    letter-spacing:1px;
    text-transform:uppercase;
}

.cockpit-metric-value {
    margin-top:6px;
    color:#f8fafc;
    font-size:.94rem;
    font-weight:900;
}

.cockpit-metric-sub {
    margin-top:3px;
    color:#475569;
    font-size:.55rem;
}

.intel-section {
    margin-top:38px;
}

.intel-head {
    display:flex;
    align-items:end;
    justify-content:space-between;
    gap:20px;
    margin-bottom:13px;
}

.intel-title {
    color:#f8fafc;
    font-size:1.02rem;
    font-weight:900;
}

.intel-sub {
    margin-top:4px;
    color:#64748b;
    font-size:.63rem;
    line-height:1.5;
}

.intel-meta {
    color:#475569;
    font-size:.56rem;
    white-space:nowrap;
}

.focus-grid {
    display:grid;
    grid-template-columns:1.1fr .9fr;
    gap:13px;
}

.focus-card {
    min-height:166px;
    padding:21px 22px;
    border:1px solid #1c293b;
    border-radius:21px;
    background:linear-gradient(145deg,#0f1725,#0a111c);
}

.focus-label {
    color:#64748b;
    font-size:.54rem;
    font-weight:850;
    letter-spacing:1px;
    text-transform:uppercase;
}

.focus-title {
    margin-top:8px;
    color:#f8fafc;
    font-size:.88rem;
    font-weight:850;
}

.focus-amount {
    margin-top:13px;
    color:#a5b4fc;
    font-size:1.18rem;
    font-weight:900;
}

.focus-copy {
    margin-top:5px;
    color:#64748b;
    font-size:.61rem;
    line-height:1.55;
}

.focus-progress {
    height:4px;
    margin-top:15px;
    overflow:hidden;
    border-radius:999px;
    background:#172235;
}

.focus-progress span {
    display:block;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#4f46e5,#60a5fa);
}

.rank-card {
    padding:21px 22px 15px;
    border:1px solid #1c293b;
    border-radius:21px;
    background:linear-gradient(145deg,#0f1725,#0a111c);
}

.rank-row {
    padding:11px 0 12px;
    border-bottom:1px solid rgba(51,65,85,.24);
}

.rank-row:last-child {
    border-bottom:0;
}

.rank-top {
    display:flex;
    align-items:center;
    gap:10px;
}

.rank-number {
    width:19px;
    color:#475569;
    font-size:.53rem;
    font-weight:900;
}

.rank-name {
    flex:1;
    min-width:0;
    overflow:hidden;
    color:#e2e8f0;
    font-size:.65rem;
    font-weight:800;
    white-space:nowrap;
    text-overflow:ellipsis;
}

.rank-amount {
    color:#f8fafc;
    font-size:.63rem;
    font-weight:850;
    white-space:nowrap;
}

.rank-track {
    height:4px;
    margin:7px 0 0 29px;
    overflow:hidden;
    border-radius:999px;
    background:#172235;
}

.rank-track span {
    display:block;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#6366f1,#60a5fa);
}

.rank-sub {
    margin:4px 0 0 29px;
    color:#475569;
    font-size:.51rem;
}

.flow-card {
    padding:21px 22px 10px;
    border:1px solid #1c293b;
    border-radius:21px;
    background:linear-gradient(145deg,#0f1725,#0a111c);
}

.flow-empty {
    display:flex;
    align-items:center;
    justify-content:center;
    min-height:210px;
    color:#64748b;
    font-size:.66rem;
    text-align:center;
}

.read-card {
    padding:22px;
    border:1px solid #202d42;
    border-radius:22px;
    background:
        radial-gradient(circle at 90% 0%,rgba(99,102,241,.10),transparent 35%),
        linear-gradient(145deg,#101827,#0a111b);
}

.read-kicker {
    color:#818cf8;
    font-size:.55rem;
    font-weight:900;
    letter-spacing:1.4px;
}

.read-main {
    margin-top:9px;
    color:#e2e8f0;
    font-size:.79rem;
    line-height:1.65;
}

.read-main strong {
    color:#fff;
}

.action-grid {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:13px;
    margin-top:13px;
}

.action-card {
    padding:19px 20px;
    border:1px solid #1c293b;
    border-radius:19px;
    background:#0d1522;
}

.action-number {
    color:#6366f1;
    font-size:.55rem;
    font-weight:900;
    letter-spacing:1px;
}

.action-title {
    margin-top:6px;
    color:#f8fafc;
    font-size:.73rem;
    font-weight:850;
}

.action-copy {
    margin-top:5px;
    color:#64748b;
    font-size:.59rem;
    line-height:1.55;
}

@media(max-width:900px) {
    .cockpit-metrics {
        grid-template-columns:1fr 1fr;
    }

    .focus-grid,
    .action-grid {
        grid-template-columns:1fr;
    }

    .intel-head {
        display:block;
    }

    .intel-meta {
        margin-top:5px;
    }
}

@media(max-width:600px) {
    .finora-cockpit {
        padding:23px 20px;
    }

    .cockpit-title {
        font-size:1.28rem;
    }

    .cockpit-metrics {
        grid-template-columns:1fr;
    }
}

@media(max-width:900px) {
    .summary-strip {
        grid-template-columns:1fr 1fr;
    }

    .ai-read-grid {
        grid-template-columns:1fr;
    }

    .section-head {
        display:block;
    }

    .section-meta {
        margin-top:5px;
        text-align:left;
    }
}

@media(max-width:600px) {
    .summary-strip {
        grid-template-columns:1fr;
    }

    .story-hero {
        padding:23px 20px;
    }

    .story-hero-title {
        font-size:1.18rem;
    }

    .spending-panel,
    .flow-panel,
    .ai-read-panel {
        padding:17px 16px;
    }
}

@media(max-width:900px) {
    .story-grid {
        grid-template-columns:1fr;
    }

    .insight-strip {
        grid-template-columns:1fr;
    }
}

.home-top-grid {
    display:flex;
    align-items:stretch;
    gap:20px;
    margin-bottom:22px;
}

.home-hero-column {
    flex:1.65;
    min-width:0;
}

.home-upload-column {
    flex:1;
    min-width:330px;
    display:flex;
    flex-direction:column;
}

.home-upload-panel {
    position:relative;
    overflow:hidden;
    flex:1;
    padding:27px 25px 22px;
    border:1px solid #263650;
    border-radius:25px;
    background:
        radial-gradient(circle at 90% 0%,rgba(99,102,241,.20),transparent 34%),
        radial-gradient(circle at 10% 100%,rgba(14,165,233,.08),transparent 32%),
        linear-gradient(145deg,#101827,#0a111c);
    box-shadow:0 20px 60px rgba(0,0,0,.20);
}

.home-upload-panel::before {
    content:"";
    position:absolute;
    left:-30%;
    top:0;
    width:55%;
    height:1px;
    background:linear-gradient(90deg,transparent,#818cf8,transparent);
    animation:upload-scan 3.8s ease-in-out infinite;
}

.home-upload-eyebrow {
    color:#818cf8;
    font-size:.60rem;
    font-weight:850;
    letter-spacing:1.4px;
}

.home-upload-heading {
    color:#fff;
    font-size:1.20rem;
    line-height:1.15;
    font-weight:900;
    margin-top:8px;
}

.home-upload-copy {
    color:#64748b;
    font-size:.68rem;
    line-height:1.55;
    margin-top:7px;
}

.home-upload-icon {
    width:48px;
    height:48px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-bottom:15px;
    border-radius:14px;
    background:rgba(99,102,241,.12);
    border:1px solid rgba(129,140,248,.22);
    font-size:23px;
}

.home-upload-hint {
    display:flex;
    align-items:center;
    gap:7px;
    margin-top:13px;
    color:#64748b;
    font-size:.60rem;
}

.home-upload-hint-dot {
    width:6px;
    height:6px;
    border-radius:50%;
    background:#4ade80;
    box-shadow:0 0 10px rgba(74,222,128,.55);
}

@keyframes upload-scan {
    0% { transform:translateX(-120%); opacity:0; }
    20%,70% { opacity:1; }
    100% { transform:translateX(310%); opacity:0; }
}

@media(max-width:900px) {
    .home-top-grid {
        display:block;
    }

    .home-upload-column {
        min-width:0;
        margin-top:16px;
    }

    .home-upload-panel {
        padding:23px 20px 21px;
    }
}

.home-upload {
    position:relative;
    overflow:hidden;
    margin-top:32px;
    padding:28px;
    border:1px solid #24334a;
    border-radius:25px;
    background:
        radial-gradient(circle at 80% 0%,rgba(99,102,241,.12),transparent 35%),
        linear-gradient(145deg,#0e1726,#0a111c);
}

.home-upload-title {
    color:#fff;
    font-size:1.05rem;
    font-weight:900;
}

.home-upload-sub {
    color:#64748b;
    font-size:.7rem;
    line-height:1.55;
    margin-top:5px;
}

.analyze-shell {
    margin-top:15px;
    padding:2px;
    border-radius:15px;
    background:linear-gradient(90deg,#4f46e5,#7c3aed,#2563eb,#4f46e5);
    background-size:300% 100%;
    animation:analyze-gradient 4s ease infinite;
}

.analyze-shell button {
    width:100%;
    min-height:52px;
    border:0 !important;
    border-radius:12px !important;
    background:#151b2b !important;
    color:#fff !important;
    font-size:.86rem !important;
    font-weight:850 !important;
    letter-spacing:.1px;
}

.analyze-shell button:hover {
    background:#1a2237 !important;
}

@keyframes analyze-gradient {
    0% { background-position:0% 50%; }
    50% { background-position:100% 50%; }
    100% { background-position:0% 50%; }
}

.finora-loader {
    position:relative;
    overflow:hidden;
    margin-top:14px;
    padding:34px 30px 30px;
    border:1px solid #263650;
    border-radius:26px;
    background:
        radial-gradient(circle at 50% 10%, rgba(99,102,241,.20), transparent 34%),
        radial-gradient(circle at 10% 100%, rgba(14,165,233,.10), transparent 30%),
        linear-gradient(145deg,#101827,#0a111c);
    box-shadow:0 25px 75px rgba(0,0,0,.26);
}

.finora-loader::after {
    content:"";
    position:absolute;
    left:-20%;
    top:0;
    width:40%;
    height:1px;
    background:linear-gradient(90deg,transparent,#818cf8,transparent);
    animation:loader-scan 2.7s ease-in-out infinite;
}

.loader-stage {
    position:relative;
    z-index:1;
    display:flex;
    align-items:center;
    gap:22px;
}

.loader-visual {
    position:relative;
    width:82px;
    height:82px;
    flex:0 0 82px;
    display:flex;
    align-items:center;
    justify-content:center;
}

.loader-ring,
.loader-ring::before,
.loader-ring::after {
    position:absolute;
    border-radius:50%;
    border:1px solid rgba(129,140,248,.26);
    content:"";
}

.loader-ring {
    inset:2px;
    animation:loader-spin 4.5s linear infinite;
    border-top-color:#818cf8;
    border-right-color:rgba(96,165,250,.45);
}

.loader-ring::before {
    inset:9px;
    border-color:rgba(96,165,250,.22);
    border-left-color:#60a5fa;
    animation:loader-spin-reverse 2.8s linear infinite;
}

.loader-ring::after {
    inset:18px;
    border-color:rgba(74,222,128,.18);
    border-bottom-color:#4ade80;
    animation:loader-spin 2s linear infinite;
}

.loader-core {
    width:31px;
    height:31px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#fff;
    font-size:15px;
    background:linear-gradient(145deg,#6366f1,#3b82f6);
    box-shadow:0 0 32px rgba(99,102,241,.42);
    animation:loader-pulse 1.8s ease-in-out infinite;
}

.loader-copy {
    min-width:0;
}

.loader-kicker {
    color:#60a5fa;
    font-size:.61rem;
    font-weight:850;
    letter-spacing:1.5px;
}

.loader-title {
    color:#fff;
    font-size:1.08rem;
    font-weight:850;
    margin-top:5px;
}

.loader-message {
    position:relative;
    min-height:21px;
    margin-top:7px;
    color:#94a3b8;
    font-size:.74rem;
    line-height:1.5;
}

.loader-message span {
    position:absolute;
    left:0;
    top:0;
    opacity:0;
    animation:loader-message 15s infinite;
}

.loader-message span:nth-child(1) { animation-delay:0s; }
.loader-message span:nth-child(2) { animation-delay:3s; }
.loader-message span:nth-child(3) { animation-delay:6s; }
.loader-message span:nth-child(4) { animation-delay:9s; }
.loader-message span:nth-child(5) { animation-delay:12s; }

.loader-track {
    position:relative;
    z-index:1;
    height:5px;
    margin-top:25px;
    overflow:hidden;
    border-radius:999px;
    background:#172238;
}

.loader-track span {
    display:block;
    width:32%;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#4f46e5,#60a5fa,#818cf8);
    box-shadow:0 0 18px rgba(96,165,250,.35);
    animation:loader-progress 2.4s ease-in-out infinite;
}

.loader-foot {
    position:relative;
    z-index:1;
    display:flex;
    justify-content:space-between;
    gap:15px;
    margin-top:10px;
    color:#475569;
    font-size:.59rem;
}

@keyframes loader-spin {
    to { transform:rotate(360deg); }
}

@keyframes loader-spin-reverse {
    to { transform:rotate(-360deg); }
}

@keyframes loader-pulse {
    0%,100% { transform:scale(.94); opacity:.85; }
    50% { transform:scale(1.06); opacity:1; }
}

@keyframes loader-progress {
    0% { transform:translateX(-120%); width:24%; }
    50% { width:45%; }
    100% { transform:translateX(330%); width:30%; }
}

@keyframes loader-scan {
    0% { transform:translateX(-120%); opacity:0; }
    15%,70% { opacity:1; }
    100% { transform:translateX(370%); opacity:0; }
}

@keyframes loader-message {
    0%,16% { opacity:0; transform:translateY(5px); }
    20%,34% { opacity:1; transform:translateY(0); }
    38%,100% { opacity:0; transform:translateY(-5px); }
}

@media(max-width:900px) {
    .block-container {
        padding:12px 16px 50px;
    }

    .hero {
        padding:35px 27px;
        min-height:450px;
    }

    .hero-title {
        letter-spacing:-2px;
    }

    .ai-orbit {
        right:-35px;
        top:215px;
        opacity:.35;
    }

    .chart-card {
        padding:14px 10px 9px;
        border-radius:18px;
    }

    .chart-heading {
        margin-left:2px;
        margin-right:2px;
    }

    .chart-badge {
        display:none;
    }

    .finora-loader {
        padding:26px 20px 22px;
        border-radius:21px;
    }

    .loader-stage {
        gap:15px;
    }

    .loader-visual {
        width:66px;
        height:66px;
        flex-basis:66px;
    }

    .loader-title {
        font-size:.94rem;
    }

    .loader-message {
        font-size:.69rem;
    }
}


/* ============================================================
   CATEGORY EDITOR + OVERVIEW ACTIONS
   ============================================================ */
.overview-actions {
    display:flex;
    gap:10px;
    align-items:center;
    justify-content:flex-end;
    margin:16px 0 4px;
}
.category-editor-note {
    margin:10px 0 14px;
    padding:13px 15px;
    border:1px solid rgba(129,140,248,.18);
    border-radius:14px;
    background:rgba(99,102,241,.06);
    color:#94a3b8;
    font-size:.78rem;
    line-height:1.6;
}
.category-editor-note strong { color:#e2e8f0; }

/* ============================================================
   READABILITY + ACCESSIBILITY OVERRIDES
   ============================================================ */
.page-title { font-size:2.65rem !important; line-height:1.12 !important; }
.page-subtitle { font-size:1.02rem !important; line-height:1.6 !important; color:#94a3b8 !important; }
.brand-name { font-size:1.25rem !important; }
.brand-sub { font-size:.78rem !important; }
.ai-ready { font-size:.78rem !important; }
.story-kicker { font-size:.72rem !important; }
.story-hero-title { font-size:1.75rem !important; }
.story-hero-copy { font-size:.92rem !important; }
.story-hero-insight { font-size:.95rem !important; }
.summary-label { font-size:.72rem !important; }
.summary-value { font-size:1.35rem !important; }
.summary-sub { font-size:.78rem !important; }
.section-head { margin-top:42px !important; }
.section-title { font-size:1.45rem !important; }
.section-subtitle { font-size:.92rem !important; }
.insight-title { font-size:1rem !important; }
.insight-value { font-size:1.15rem !important; }
.insight-sub { font-size:.84rem !important; line-height:1.55 !important; }
.metric-label { font-size:.74rem !important; }
.metric-value { font-size:1.65rem !important; }
.metric-sub { font-size:.8rem !important; }
.chart-heading-title { font-size:1.2rem !important; }
.chart-heading-sub { font-size:.82rem !important; }
.feature-title { font-size:1rem !important; }
.feature-text { font-size:.82rem !important; line-height:1.65 !important; }
.action-number { font-size:.72rem !important; }
.action-title { font-size:1.05rem !important; }
.action-copy { font-size:.86rem !important; line-height:1.65 !important; }
.chat-user, .chat-ai { font-size:.98rem !important; line-height:1.7 !important; }
.ai-title { font-size:1.15rem !important; }
.ai-subtitle { font-size:.82rem !important; }
[data-testid="stCaptionContainer"] { font-size:.82rem !important; }
[data-testid="stWidgetLabel"] p { font-size:.9rem !important; }
[data-testid="stRadio"] label p { font-size:1rem !important; font-weight:750 !important; }
[data-testid="stRadio"] > div { gap:8px !important; }
[data-testid="stRadio"] label { padding:10px 16px !important; border:1px solid #26354c !important; border-radius:12px !important; background:#101827 !important; }
[data-testid="stRadio"] label:has(input:checked) { border-color:#6366f1 !important; background:rgba(99,102,241,.16) !important; }
.stButton > button { font-size:.95rem !important; min-height:48px !important; }
[data-testid="stDataFrame"] { font-size:.95rem !important; }
[data-testid="stFileUploaderDropzone"] { min-height:120px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div { font-size:.95rem !important; }


/* FINORA V3 READABILITY + NATIVE NAVIGATION */
html, body, [class*="stApp"] { font-size:17px !important; }
.block-container { max-width:1480px !important; padding:24px 38px 80px !important; }
.brand-name { font-size:1.45rem !important; }
.brand-sub { font-size:.88rem !important; }
.finora-nav { display:flex; align-items:center; justify-content:center; gap:10px; padding-top:7px; }
.finora-nav-link { display:inline-flex; align-items:center; justify-content:center; min-height:52px; padding:0 24px; border-radius:14px; border:1px solid #26354c; background:#101827; color:#dbeafe !important; text-decoration:none !important; font-size:1.02rem; font-weight:800; white-space:nowrap; cursor:pointer; transition:all .18s ease; }
.finora-nav-link:hover { border-color:#818cf8; background:#172238; color:#fff !important; transform:translateY(-1px); }
.finora-nav-link.active { border-color:#6366f1; background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(59,130,246,.14)); color:#fff !important; box-shadow:0 10px 28px rgba(79,70,229,.16); }
.page-title { font-size:3rem !important; line-height:1.12 !important; }
.page-subtitle { font-size:1.12rem !important; line-height:1.65 !important; }
.section-title { font-size:1.7rem !important; }
.section-subtitle { font-size:1rem !important; }
.story-kicker { font-size:.82rem !important; }
.story-hero-title { font-size:2.05rem !important; }
.story-hero-copy { font-size:1.05rem !important; line-height:1.7 !important; }
.story-hero-insight { font-size:1.05rem !important; }
.summary-label { font-size:.82rem !important; }
.summary-value { font-size:1.6rem !important; }
.summary-sub { font-size:.9rem !important; }
.metric-label { font-size:.86rem !important; }
.metric-value { font-size:2rem !important; }
.metric-sub { font-size:.9rem !important; }
.insight-title { font-size:1.12rem !important; }
.insight-value { font-size:1.35rem !important; }
.insight-sub { font-size:.94rem !important; line-height:1.6 !important; }
.feature-title { font-size:1.12rem !important; }
.feature-text { font-size:.94rem !important; line-height:1.65 !important; }
.chart-heading-title { font-size:1.35rem !important; }
.chart-heading-sub { font-size:.94rem !important; }
.action-title { font-size:1.2rem !important; }
.action-copy { font-size:.98rem !important; }
[data-testid="stDataFrame"] { font-size:1rem !important; }
[data-testid="stWidgetLabel"] p { font-size:1rem !important; }
.stButton > button { font-size:1rem !important; min-height:50px !important; }
.ai-panel { padding:28px !important; border-radius:24px !important; }
.ai-title { font-size:1.45rem !important; }
.ai-subtitle { font-size:.98rem !important; line-height:1.5 !important; }
.chat-user, .chat-ai { font-size:1.02rem !important; line-height:1.75 !important; padding:17px 19px !important; max-width:100% !important; }
.ai-shell { border:1px solid #26344b; border-radius:24px; padding:22px; background:linear-gradient(145deg,#101827,#0b111d); min-height:620px; }
.ai-shell-title { color:#fff; font-size:1.35rem; font-weight:850; }
.ai-shell-subtitle { color:#94a3b8; font-size:.92rem; line-height:1.6; margin-top:5px; }
.ai-message-user { margin:16px 0 10px auto; padding:15px 17px; border-radius:17px 17px 5px 17px; background:#24324d; color:#f8fafc; font-size:1rem; line-height:1.65; }
.ai-message-bot { margin:10px auto 16px 0; padding:16px 18px; border-radius:17px 17px 17px 5px; background:#151f32; border:1px solid #273650; color:#e2e8f0; font-size:1rem; line-height:1.7; }

/* Responsive floating Finora AI panel */
.finora-chat-header { display:flex; align-items:center; gap:12px; padding-bottom:10px; }
.finora-chat-intro { color:#94a3b8; font-size:.92rem; line-height:1.6; padding:8px 0 14px; }
div.st-key-finora_ai_popover { position:fixed !important; right:24px !important; bottom:24px !important; z-index:99999 !important; }
div.st-key-finora_ai_popover > div { max-width:min(560px, calc(100vw - 32px)) !important; }
div.st-key-finora_ai_popover [data-testid="stPopoverBody"] { width:min(560px, calc(100vw - 32px)) !important; max-height:78vh !important; overflow-y:auto !important; padding:14px !important; }
div.st-key-finora_ai_popover .stChatMessage { font-size:1rem !important; line-height:1.65 !important; }
div.st-key-finora_ai_popover .stChatMessage p { font-size:1rem !important; line-height:1.65 !important; }
div.st-key-finora_ai_popover .stTextInput input { font-size:1rem !important; min-height:48px !important; }
div.st-key-finora_ai_popover .stButton > button { min-height:46px !important; font-size:.88rem !important; line-height:1.25 !important; white-space:normal !important; }
@media (max-width:900px) {
  div.st-key-finora_ai_popover { right:14px !important; bottom:14px !important; }
  div.st-key-finora_ai_popover > div, div.st-key-finora_ai_popover [data-testid="stPopoverBody"] { width:calc(100vw - 28px) !important; max-width:calc(100vw - 28px) !important; }
}
@media (max-width:900px) { .block-container{padding:18px 18px 70px !important;} .finora-nav{justify-content:flex-start;overflow-x:auto;} .finora-nav-link{min-height:48px;padding:0 17px;font-size:.92rem;} }

/* ============================================================
   FINORA V2 — CATEGORY REVIEW POPUP / REVIEW WORKSPACE
   ============================================================ */
.category-review-shell {
    position:relative;
    margin:8px auto 0;
    max-width:1180px;
    padding:28px;
    border:1px solid #293858;
    border-radius:28px;
    background:
        radial-gradient(circle at 90% 0%,rgba(99,102,241,.18),transparent 30%),
        radial-gradient(circle at 0% 100%,rgba(14,165,233,.07),transparent 28%),
        linear-gradient(145deg,#101827,#090f19);
    box-shadow:0 28px 90px rgba(0,0,0,.34);
}
.category-review-kicker {
    color:#818cf8;
    font-size:.68rem;
    font-weight:900;
    letter-spacing:1.7px;
}
.category-review-title {
    margin-top:7px;
    color:#f8fafc;
    font-size:2rem;
    line-height:1.1;
    font-weight:900;
    letter-spacing:-.8px;
}
.category-review-subtitle {
    max-width:780px;
    margin-top:8px;
    color:#94a3b8;
    font-size:.9rem;
    line-height:1.6;
}
.category-review-summary {
    display:grid;
    grid-template-columns:1fr 1fr 1fr;
    gap:12px;
    margin-top:22px;
}
.category-review-stat {
    padding:16px 18px;
    border:1px solid #1d2a3f;
    border-radius:17px;
    background:rgba(7,13,23,.48);
}
.category-review-stat-label {
    color:#64748b;
    font-size:.64rem;
    font-weight:850;
    letter-spacing:.8px;
    text-transform:uppercase;
}
.category-review-stat-value {
    margin-top:6px;
    color:#f8fafc;
    font-size:1.35rem;
    font-weight:900;
}
.category-review-stat-sub {
    margin-top:3px;
    color:#526176;
    font-size:.65rem;
}
.category-review-progress {
    height:7px;
    margin-top:18px;
    overflow:hidden;
    border-radius:999px;
    background:#172238;
}
.category-review-progress span {
    display:block;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#4f46e5,#818cf8,#60a5fa);
}
.category-review-progress-label {
    display:flex;
    justify-content:space-between;
    gap:12px;
    margin-top:6px;
    color:#64748b;
    font-size:.62rem;
}
.category-review-card {
    margin:0 0 12px;
    padding:19px;
    border:1px solid #1d2a3f;
    border-radius:19px;
    background:linear-gradient(145deg,#0f1725,#0b121e);
}
.category-review-merchant {
    color:#f8fafc;
    font-size:1rem;
    font-weight:850;
}
.category-review-description {
    margin-top:5px;
    color:#64748b;
    font-size:.73rem;
    line-height:1.5;
}
.category-review-amount {
    color:#c7d2fe;
    font-size:1.02rem;
    font-weight:900;
    white-space:nowrap;
}
.category-review-suggestion {
    display:inline-flex;
    margin-top:10px;
    padding:5px 9px;
    border-radius:999px;
    color:#a5b4fc;
    background:rgba(99,102,241,.09);
    border:1px solid rgba(129,140,248,.18);
    font-size:.62rem;
    font-weight:800;
}
.category-review-footer {
    position:sticky;
    bottom:0;
    z-index:10;
    margin-top:15px;
    padding:15px 17px;
    border:1px solid #293858;
    border-radius:17px;
    background:rgba(10,16,28,.94);
    backdrop-filter:blur(16px);
}
.category-review-footer-text {
    color:#94a3b8;
    font-size:.72rem;
    line-height:1.5;
}
.category-review-footer-text strong {
    color:#fff;
}
@media(max-width:900px) {
    .category-review-shell { padding:20px 15px; border-radius:21px; }
    .category-review-title { font-size:1.55rem; }
    .category-review-summary { grid-template-columns:1fr; }
    .category-review-amount { margin-top:8px; }
}


/* ============================================================
   FINORA V2 — CATEGORY REVIEW INTERACTION LAYER
   ============================================================ */
.category-review-workspace {
    margin-top:18px;
    padding:22px;
    border:1px solid #25324a;
    border-radius:24px;
    background:linear-gradient(145deg,#0d1522,#0a101b);
    box-shadow:0 22px 70px rgba(0,0,0,.25);
}
.category-review-item-top {
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:18px;
}
.category-review-item-kicker {
    color:#64748b;
    font-size:.62rem;
    font-weight:900;
    letter-spacing:1.2px;
    text-transform:uppercase;
}
.category-review-merchant-row {
    display:flex;
    align-items:center;
    gap:13px;
    margin-top:7px;
}
.category-review-merchant-icon {
    width:44px;
    height:44px;
    display:flex;
    align-items:center;
    justify-content:center;
    flex:0 0 44px;
    border:1px solid #293858;
    border-radius:14px;
    background:linear-gradient(145deg,#172238,#0e1726);
    font-size:1.25rem;
}
.category-review-suggested-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    margin-top:13px;
    padding:7px 11px;
    border:1px solid rgba(129,140,248,.22);
    border-radius:999px;
    background:rgba(99,102,241,.09);
    color:#c7d2fe;
    font-size:.68rem;
    font-weight:800;
}
.category-review-section-label {
    margin:22px 0 10px;
    color:#f8fafc;
    font-size:.82rem;
    font-weight:850;
}
.category-review-category-help {
    margin:-4px 0 13px;
    color:#64748b;
    font-size:.68rem;
    line-height:1.5;
}
.category-review-custom {
    margin-top:18px;
    padding:18px;
    border:1px dashed #34435f;
    border-radius:18px;
    background:rgba(12,19,31,.72);
}
.category-review-custom-title {
    color:#f8fafc;
    font-size:.88rem;
    font-weight:850;
}
.category-review-custom-copy {
    margin-top:4px;
    color:#64748b;
    font-size:.68rem;
    line-height:1.5;
}
.category-review-match {
    margin:8px 0 2px;
    padding:9px 11px;
    border-radius:11px;
    background:rgba(34,197,94,.06);
    border:1px solid rgba(34,197,94,.12);
    color:#86efac;
    font-size:.68rem;
    font-weight:750;
}
.category-review-progress-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:7px 10px;
    border:1px solid #25324a;
    border-radius:999px;
    background:#0d1625;
    color:#94a3b8;
    font-size:.68rem;
    font-weight:800;
}
.category-review-progress-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#818cf8;
    box-shadow:0 0 12px rgba(129,140,248,.7);
}
@media(max-width:900px) {
    .category-review-workspace { padding:16px; border-radius:19px; }
    .category-review-item-top { flex-direction:column; }
}

</style>
""")


# ============================================================
# HELPERS
# ============================================================

def render(markup: str):
    st.html(markup)


def enum_value(value):
    if value is None:
        return ""
    return getattr(value, "value", str(value))


def number(value):
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def money(value, currency):
    return f"{currency} {number(value):,.2f}"


def transaction_dict(transaction):
    return {
        "Date": (
            transaction.transaction_date.isoformat()
            if getattr(transaction, "transaction_date", None)
            else ""
        ),
        "Merchant": getattr(transaction, "merchant", None) or "",
        "Description": getattr(transaction, "description_raw", "") or "",
        "Amount": number(getattr(transaction, "original_amount", None)),
        "Currency": getattr(transaction, "original_currency", "") or "",
        "Direction": enum_value(getattr(transaction, "direction", None)),
        "Type": enum_value(getattr(transaction, "transaction_type", None)),
        "Category": getattr(transaction, "category", None) or "Uncategorized",
        "Confidence": round(
            number(getattr(transaction, "extraction_confidence", 0)) * 100,
            1,
        ),
        "Review": bool(
            getattr(transaction, "requires_review", False)
        ),
    }


def make_dataframe(transactions):
    columns = [
        "Date",
        "Merchant",
        "Description",
        "Amount",
        "Currency",
        "Direction",
        "Type",
        "Category",
        "Confidence",
        "Review",
    ]

    rows = [transaction_dict(t) for t in transactions]
    return pd.DataFrame(rows, columns=columns)


def build_overview_download_dataframe(transactions):
    """Build a compact CSV-friendly overview without changing dashboard calculations."""
    income, expenses, net = calculate_financials(transactions)
    currency = get_currency(transactions)
    review_count = sum(
        bool(getattr(transaction, "requires_review", False))
        for transaction in transactions
    )

    rows = [
        {"Section": "Summary", "Item": "Money received", "Amount": number(income), "Percentage": ""},
        {"Section": "Summary", "Item": "Money spent", "Amount": number(expenses), "Percentage": ""},
        {"Section": "Summary", "Item": "Net movement", "Amount": number(net), "Percentage": ""},
        {"Section": "Summary", "Item": "Transactions", "Amount": len(transactions), "Percentage": ""},
        {"Section": "Summary", "Item": "Needs review", "Amount": review_count, "Percentage": ""},
    ]

    spending_df = get_categories(transactions)
    total_spend = float(expenses) if expenses else 0.0

    if not spending_df.empty:
        for _, row in spending_df.iterrows():
            amount = float(row["Amount"])
            rows.append({
                "Section": "Spending by category",
                "Item": str(row["Category"]),
                "Amount": amount,
                "Percentage": f"{(amount / total_spend * 100) if total_spend else 0:.1f}%",
            })

    result = pd.DataFrame(
        rows,
        columns=["Section", "Item", "Amount", "Percentage"],
    )
    result.insert(4, "Currency", currency)
    return result



# ============================================================
# CATEGORY CAMPAIGN HELPERS
# ============================================================

_KNOWN_MERCHANT_FAMILIES = {
    "mcdonalds": ("mcdonald", "mcdonalds", "mcdonalds-"),
    "adnoc": ("adnoc",),
    "kfc": ("kfc",),
    "star cinemas": ("star cinemas", "starcinemas"),
    "carrefour": ("carrefour", "carrefoure"),
    "lulu": ("lulu", "luluhypermarket"),
    "nesto": ("nesto",),
    "safeer": ("safeer",),
    "paris cafe": ("paris cafe", "pariscafe"),
    "grandiose": ("grandiose",),
    "new parco": ("new parco", "newparco"),
    "golden family baqala": ("golden family baqala",),
    "millennium hospital": ("millennium hospital",),
    "e&": ("e&", "e and", "e digital", "e& digital"),
}


def _campaign_normalize(value):
    """Normalize merchant names for conservative same-brand grouping."""
    text = str(value or "").casefold().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[’'`´]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # Remove company suffixes and common location fragments.
    text = re.sub(
        r"\b(llc|ltd|limited|inc|opc|spc|l l c|br|ph|auh|are|dubai|sharjah|abudhabi|abu dhabi)\b",
        " ",
        text,
    )
    text = re.sub(r"\b\d+[a-z]*\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _merchant_family(value):
    normalized = _campaign_normalize(value)
    if not normalized:
        return ""

    for family, aliases in _KNOWN_MERCHANT_FAMILIES.items():
        for alias in aliases:
            alias_norm = _campaign_normalize(alias)
            if not alias_norm:
                continue
            if normalized == alias_norm or normalized.startswith(alias_norm + " "):
                return family

    # Generic merchants stay conservative: remove numeric/store suffixes,
    # but do not collapse unrelated merchants to a single first word.
    return normalized


def _merchant_campaign_match(anchor, candidate):
    """Return True only for a reasonably strong same-merchant-family match."""
    anchor_name = getattr(anchor, "merchant", None) or getattr(anchor, "description_raw", "")
    candidate_name = getattr(candidate, "merchant", None) or getattr(candidate, "description_raw", "")

    anchor_family = _merchant_family(anchor_name)
    candidate_family = _merchant_family(candidate_name)

    if not anchor_family or not candidate_family:
        return False

    if anchor_family == candidate_family:
        return True

    # Handle close spelling variants only when both normalized names are
    # already very similar. This catches small OCR/store-name variations
    # without broadly grouping unrelated merchants.
    ratio = SequenceMatcher(None, anchor_family, candidate_family).ratio()
    if ratio >= 0.92:
        return True

    return False


def _campaign_candidates(transactions, anchor_index, excluded_indices=None):
    """Find same-direction transactions that look like the same merchant family."""
    excluded = set(excluded_indices or set())
    if anchor_index < 0 or anchor_index >= len(transactions):
        return []

    anchor = transactions[anchor_index]
    anchor_direction = enum_value(getattr(anchor, "direction", None)).casefold()
    matches = []

    for idx, candidate in enumerate(transactions):
        if idx == anchor_index or idx in excluded:
            continue
        direction = enum_value(getattr(candidate, "direction", None)).casefold()
        if direction != anchor_direction:
            continue
        if _merchant_campaign_match(anchor, candidate):
            matches.append(idx)

    return matches


def _apply_campaign(engine, transactions, indices, category):
    """Apply a confirmed category campaign and remember each merchant variant."""
    changed = 0
    for idx in indices:
        if idx < 0 or idx >= len(transactions):
            continue
        tx = transactions[idx]
        engine.learn_from_user(
            tx,
            category,
            "General",
            apply_all=False,
            transactions=transactions,
        )
        changed += 1
    return changed


def _campaign_label(transactions, indices):
    names = []
    for idx in indices:
        if idx < 0 or idx >= len(transactions):
            continue
        tx = transactions[idx]
        name = str(
            getattr(tx, "merchant", None)
            or getattr(tx, "description_raw", None)
            or "Unknown merchant"
        ).strip()
        if name and name not in names:
            names.append(name)
    return names


def _redirect_to_overview():
    st.session_state.transaction_focus = None
    st.session_state.page = "Overview"
    st.query_params["page"] = "Overview"
    st.session_state.pending_category_campaigns = []


def get_currency(transactions):
    currencies = [
        str(getattr(t, "original_currency", "")).upper()
        for t in transactions
        if getattr(t, "original_currency", None)
    ]

    if not currencies:
        return "UNKNOWN"

    return Counter(currencies).most_common(1)[0][0]


def get_statement_metadata():
    value = st.session_state.get("statement_metadata")
    return value if isinstance(value, dict) else {}


def is_credit_card_statement(transactions=None):
    metadata = get_statement_metadata()
    statement_type = str(
        metadata.get("statement_type")
        or ""
    ).lower()
    if statement_type == "credit_card":
        return True

    for transaction in (transactions or st.session_state.get("transactions") or []):
        value = enum_value(getattr(transaction, "statement_type", None)).lower()
        if value == "credit_card":
            return True

    return False


def card_snapshot(transactions):
    metadata = get_statement_metadata()
    if not is_credit_card_statement(transactions):
        return {}

    snapshot = {}
    for key in (
        "card_limit",
        "available_limit",
        "minimum_payment_due",
        "payment_due_date",
        "total_payment_due",
        "profit_other_charges",
        "current_balance",
        "opening_balance",
    ):
        value = metadata.get(key)
        if value is not None:
            snapshot[key] = value

    # If the statement summary is unavailable, use the transaction flow
    # only as a fallback for the due/current balance when possible.
    income, expenses, _ = calculate_financials(transactions)
    if snapshot.get("current_balance") is None and snapshot.get("opening_balance") is not None:
        snapshot["current_balance"] = float(snapshot["opening_balance"]) + float(expenses) - float(income)
    if snapshot.get("available_limit") is None and snapshot.get("card_limit") is not None and snapshot.get("current_balance") is not None:
        snapshot["available_limit"] = float(snapshot["card_limit"]) - float(snapshot["current_balance"])
    if snapshot.get("total_payment_due") is None and snapshot.get("current_balance") is not None:
        snapshot["total_payment_due"] = snapshot["current_balance"]

    return snapshot


def calculate_financials(transactions):
    income = Decimal("0")
    expenses = Decimal("0")

    for transaction in transactions:
        amount = abs(
            getattr(
                transaction,
                "original_amount",
                Decimal("0"),
            )
            or Decimal("0")
        )

        if (
            getattr(transaction, "direction", None)
            == TransactionDirection.CREDIT
        ):
            income += amount
        else:
            expenses += amount

    return income, expenses, income - expenses


def get_bank_closing_balance(transactions):
    """
    Return the actual closing/account balance for bank statements.

    Priority:
      1. Explicit closing_balance supplied by statement metadata.
      2. The last transaction's running_balance extracted from the statement.
      3. Opening balance + net movement, when an opening balance is available.

    This is intentionally separate from net movement:
        net movement = credits - debits
        closing balance = opening balance + net movement
    """
    metadata = get_statement_metadata()

    value = metadata.get("closing_balance")
    if value is not None:
        parsed = _parse_decimal_text(value)
        if parsed is not None:
            return parsed

    ordered_transactions = list(transactions or [])
    for transaction in reversed(ordered_transactions):
        running_balance = getattr(transaction, "running_balance", None)
        if running_balance is not None:
            parsed = _parse_decimal_text(running_balance)
            if parsed is not None:
                return parsed

    opening = metadata.get("opening_balance")
    opening_value = _parse_decimal_text(opening)
    if opening_value is not None:
        _, _, net = calculate_financials(ordered_transactions)
        return float(opening_value) + number(net)

    return None


def get_categories(transactions):
    rows = []

    for transaction in transactions:
        if (
            getattr(transaction, "direction", None)
            != TransactionDirection.DEBIT
        ):
            continue

        rows.append(
            {
                "Category": (
                    getattr(transaction, "category", None)
                    or "Uncategorized"
                ),
                "Amount": number(
                    getattr(
                        transaction,
                        "original_amount",
                        None,
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .groupby("Category", as_index=False)
        .sum()
        .sort_values("Amount", ascending=False)
    )



def spending_insights(transactions):
    """Return explainable spending insights without inventing categories."""
    df = make_dataframe(transactions)

    if df.empty:
        return {
            "category_coverage": 0.0,
            "category_df": pd.DataFrame(),
            "merchant_df": pd.DataFrame(),
            "top_category": None,
            "least_category": None,
            "top_merchant": None,
            "review_category": None,
            "review_scenario": 0.0,
        }

    debit_df = df[df["Direction"] == "debit"].copy()

    if debit_df.empty:
        return {
            "category_coverage": 0.0,
            "category_df": pd.DataFrame(),
            "merchant_df": pd.DataFrame(),
            "top_category": None,
            "least_category": None,
            "top_merchant": None,
            "review_category": None,
            "review_scenario": 0.0,
        }

    category_series = debit_df["Category"].fillna("").astype(str).str.strip()
    valid_category_mask = ~category_series.str.lower().isin(
        {"", "uncategorized", "unknown", "other"}
    )

    coverage = float(valid_category_mask.mean() * 100)

    categorized = debit_df[valid_category_mask].copy()

    if not categorized.empty:
        category_breakdown = (
            categorized.groupby("Category", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
        )
    else:
        category_breakdown = pd.DataFrame(
            columns=["Category", "Amount"]
        )

    merchant_series = (
        debit_df["Merchant"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    merchant_series = merchant_series.where(
        merchant_series != "",
        debit_df["Description"].fillna("Unknown merchant").astype(str),
    )

    merchant_breakdown = (
        pd.DataFrame(
            {
                "Merchant": merchant_series,
                "Amount": debit_df["Amount"].astype(float),
            }
        )
        .groupby("Merchant", as_index=False)["Amount"]
        .sum()
        .sort_values("Amount", ascending=False)
    )

    top_category = (
        category_breakdown.iloc[0].to_dict()
        if not category_breakdown.empty
        else None
    )

    least_category = (
        category_breakdown.iloc[-1].to_dict()
        if len(category_breakdown) >= 2
        else None
    )

    top_merchant = (
        merchant_breakdown.iloc[0].to_dict()
        if not merchant_breakdown.empty
        else None
    )

    discretionary = {
        "Food & Dining",
        "Restaurants",
        "Fast Food",
        "Food Delivery",
        "Shopping",
        "Entertainment",
        "Travel",
        "Cafes",
        "Subscriptions",
        "Gaming",
        "Events",
    }

    review_category = None
    review_scenario = 0.0

    if not category_breakdown.empty:
        candidates = category_breakdown[
            category_breakdown["Category"].astype(str).isin(discretionary)
        ]

        if not candidates.empty:
            review_category = candidates.iloc[0].to_dict()
            review_scenario = float(review_category["Amount"]) * 0.10

    return {
        "category_coverage": coverage,
        "category_df": category_breakdown,
        "merchant_df": merchant_breakdown,
        "top_category": top_category,
        "least_category": least_category,
        "top_merchant": top_merchant,
        "review_category": review_category,
        "review_scenario": review_scenario,
    }


def _analyze_uploaded_statement(uploaded, password, loader_placeholder=None):
    """Run the existing universal pipeline from any UI entry point."""
    os.makedirs("storage", exist_ok=True)

    # Every uploaded statement gets its own identity. This is critical when
    # multiple Streamlit tabs/sessions are open: one statement must never
    # restore another statement's transactions.
    statement_id = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", uploaded.name)
    file_path = os.path.join(
        "storage",
        f"statement_{statement_id}_{safe_name}",
    )
    # Start with a completely clean statement context. This is important
    # even when the user uploads from an existing Streamlit session: the
    # previous statement's metadata must never survive into the new one.
    st.session_state.statement_id = statement_id
    st.session_state.statement_source_path = file_path
    st.session_state.file_name = None
    st.session_state.statement_metadata = {}
    st.session_state.transactions = []
    st.session_state.transactions_backup = []
    st.session_state.category_engine = None
    st.session_state.chat_history = []
    st.session_state.ai_summary = None
    st.session_state.transaction_focus = None
    st.session_state.review_cursor = 0
    st.session_state.pending_category_campaigns = []
    st.session_state.category_review_skipped = False
    st.session_state.category_icon_overrides = {}
    st.session_state.category_editor_version = int(st.session_state.get("category_editor_version", 0)) + 1
    st.query_params["statement_id"] = statement_id

    with open(file_path, "wb") as file:
        file.write(uploaded.getbuffer())

    if loader_placeholder is not None:
        loader_placeholder.html("""
        <div class="finora-loader" role="status" aria-live="polite">
            <div class="loader-stage">
                <div class="loader-visual">
                    <div class="loader-ring"></div>
                    <div class="loader-core">✦</div>
                </div>

                <div class="loader-copy">
                    <div class="loader-kicker">
                        FINORA AI · ANALYSIS IN PROGRESS
                    </div>

                    <div class="loader-title">
                        Hang on — your financial intelligence is loading.
                    </div>

                    <div class="loader-message">
                        <span>Reading your financial statement...</span>
                        <span>Detecting the statement structure...</span>
                        <span>Extracting and normalizing transactions...</span>
                        <span>Validating your financial data...</span>
                        <span>Building your financial intelligence...</span>
                    </div>
                </div>
            </div>

            <div class="loader-track">
                <span></span>
            </div>

            <div class="loader-foot">
                <span>Secure local processing</span>
                <span>Almost there</span>
            </div>
        </div>
        """)

    time.sleep(0.15)

    pipeline = FinancialStatementPipeline(
        file_path=file_path,
        password=password or None,
    )

    result = list(pipeline.run() or [])

    metadata = getattr(pipeline, "statement_metadata", None)
    if metadata is not None:
        try:
            st.session_state.statement_metadata = metadata.model_dump(mode="json")
        except Exception:
            st.session_state.statement_metadata = dict(
                getattr(metadata, "__dict__", {}) or {}
            )

    # Use the pipeline metadata when available; otherwise recover card summary
    # directly from the source PDF. This does not alter extraction results.
    current_metadata = dict(st.session_state.get("statement_metadata") or {})
    parsed_metadata = _extract_credit_card_summary(
        file_path,
        password=password or None,
    )
    if parsed_metadata:
        # Merge the PDF-derived fields only when the pipeline did not already provide them.
        current_metadata = {**current_metadata, **parsed_metadata}
        st.session_state.statement_metadata = current_metadata

    st.session_state.file_name = uploaded.name

    # Save the successful extraction BEFORE running any optional V2 logic.
    # If the browser disconnects or categorization encounters an error, the
    # extracted transactions can still be restored on the next Streamlit run.
    st.session_state.transactions = result
    _save_transaction_backup(result)

    # V2: classify transactions immediately and apply previously learned
    # merchant/category mappings before the dashboard is rendered.
    try:
        engine = HybridCategoryEngine(CategoryMemory())
        engine.classify_transactions(result)
        st.session_state.category_engine = engine
        # Save the categorized version as well.
        _save_transaction_backup(result)
    except Exception:
        # Categorization must never destroy a successful extraction.
        st.session_state.category_engine = None

    # Bank statements need a real closing-balance calculation. Do NOT trust
    # an arbitrary closing_balance emitted by the generic transaction parser.
    # Reconcile the PDF's opening balance, transaction flow and final row.
    if not is_credit_card_statement(result):
        pipeline_metadata = dict(st.session_state.get("statement_metadata") or {})
        bank_balance_metadata = _extract_bank_balance_metadata(
            file_path,
            password=password or None,
        )

        # PDF-derived opening/closing evidence takes precedence over generic
        # pipeline metadata. The pipeline's value can be wrong when a PDF
        # extractor maps a nearby numeric field to "closing_balance".
        metadata_now = dict(pipeline_metadata)
        metadata_now.pop("closing_balance", None)
        metadata_now.update(bank_balance_metadata)

        income_now, expenses_now, net_now = calculate_financials(result)
        opening_value = _parse_decimal_text(metadata_now.get("opening_balance"))

        # The mathematically correct statement closing balance when an opening
        # balance is known is: opening + credits - debits.
        calculated_closing = None
        if opening_value is not None:
            calculated_closing = float(opening_value) + float(net_now)
            metadata_now["calculated_closing_balance"] = calculated_closing

        # Also inspect the final transaction's extracted running balance.
        final_row_balance = _parse_decimal_text(
            bank_balance_metadata.get("final_transaction_balance")
        )
        if final_row_balance is None:
            for transaction in reversed(result):
                running_balance = getattr(transaction, "running_balance", None)
                if running_balance is not None:
                    final_row_balance = _parse_decimal_text(running_balance)
                    if final_row_balance is not None:
                        break

        # Selection rules:
        #   A) PDF final-row balance + calculated balance agree -> use it.
        #   B) They disagree -> prefer the PDF final-row balance because it is
        #      the bank's reported ending balance, but keep the difference.
        #   C) No final-row balance -> use opening + net movement.
        #   D) No opening/final balance -> use explicit PDF closing metadata.
        if final_row_balance is not None:
            metadata_now["closing_balance"] = final_row_balance
            metadata_now["closing_balance_source"] = "final_transaction_balance"
            if calculated_closing is not None:
                difference = final_row_balance - calculated_closing
                metadata_now["closing_balance_difference"] = difference
                metadata_now["balance_reconciled"] = abs(difference) < 0.01
        elif calculated_closing is not None:
            metadata_now["closing_balance"] = calculated_closing
            metadata_now["closing_balance_source"] = "opening_plus_net_movement"
            metadata_now["closing_balance_difference"] = 0.0
            metadata_now["balance_reconciled"] = True
        elif metadata_now.get("closing_balance") is not None:
            metadata_now["closing_balance_source"] = "pdf_closing_balance"

        # Keep the transaction-flow totals explicit for reports/RAG.
        metadata_now["net_movement"] = float(net_now)
        metadata_now["total_received"] = float(income_now)
        metadata_now["total_spent"] = float(expenses_now)

        st.session_state.statement_metadata = metadata_now

    st.session_state.upload_mode = False
    _save_statement_context()
    st.session_state.chat_history = []
    st.session_state.ai_summary = None
    st.session_state.category_review_skipped = False

    review_count = sum(
        bool(getattr(transaction, "requires_review", False))
        for transaction in result
    )

    if review_count > 0:
        st.session_state.page = "Review"
        st.query_params["page"] = "Review"
    else:
        st.session_state.page = "Overview"
        st.query_params["page"] = "Overview"


def build_ai_context(transactions):
    """Build a stable, statement-grounded context for the AI layer."""
    df = make_dataframe(transactions)
    income, expenses, net = calculate_financials(transactions)
    category_df = get_categories(transactions)
    insights = spending_insights(transactions)

    largest_transactions = []
    if not df.empty and "Amount" in df.columns:
        largest_transactions = (
            df.sort_values("Amount", ascending=False)
            .head(10)
            .to_dict("records")
        )

    rag_summary = {}
    try:
        rag_summary = StatementRAG(transactions).summary()
    except Exception:
        rag_summary = {}

    return {
        "currency": get_currency(transactions),
        "transaction_count": len(transactions),
        "statement_metadata": get_statement_metadata(),
        "total_income": float(income),
        "total_expenses": float(expenses),
        "net_cash_flow": float(net),
        "categories": (
            category_df.to_dict("records")
            if not category_df.empty
            else []
        ),
        "largest_transactions": largest_transactions,
        "spending_insights": {
            "category_coverage_percent": insights["category_coverage"],
            "top_category": insights["top_category"],
            "least_category": insights["least_category"],
            "top_merchant": insights["top_merchant"],
            "review_category": insights["review_category"],
            "review_scenario_10_percent": insights["review_scenario"],
        },
        # RAG-derived intelligence is kept separate from canonical extracted fields.
        # This lets Finora answer category questions even when the PDF extractor
        # could not populate a category column.
        "rag_intelligence": rag_summary,
    }



def _safe_focus_value(value):
    return str(value or "").strip()


def set_transaction_focus(kind, value):
    st.session_state.transaction_focus = {"kind": kind, "value": _safe_focus_value(value)}
    st.session_state.page = "Transactions"
    st.query_params["page"] = "Transactions"


def _append_chat_turn(question, answer):
    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _deterministic_finora_answer(rag_context):
    """Return an exact answer for common financial questions without LLM arithmetic."""
    intent = rag_context.get("intent", {}).get("intent", "general")
    evidence = rag_context.get("computed_evidence", {})
    answer = evidence.get("answer")
    analytics = rag_context.get("statement_analytics", {})
    currency = analytics.get("currency", "UNKNOWN")

    def amount(value):
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    if intent == "top_merchant":
        rows = answer or []
        if not rows:
            return "I couldn't identify an outgoing merchant from the analyzed transactions."
        top = rows[0]
        return (
            f"Your highest-spending merchant is **{top['merchant']}**, with **{currency} {amount(top['amount'])}** "
            f"across **{top['count']} transaction(s)**."
        )

    if intent == "category_spend":
        category = evidence.get("category")
        if category and isinstance(answer, dict):
            total = float(answer.get("total", 0))
            count = int(answer.get("matched_transactions", 0))
            if count == 0:
                return f"I couldn't identify any transactions that can be attributed to **{category}** in this statement."
            confidence = answer.get("confidence", "unknown")
            qualifier = "This is inferred from merchant/description text." if confidence == "inferred" else "This category is present in the extracted statement data."
            return f"You spent **{currency} {amount(total)}** on **{category}** across **{count} transaction(s)**. {qualifier}"

        rows = answer or []
        if not rows:
            return "There isn't enough category information in this statement to summarize spending areas."
        lines = [f"- **{row['category']}** — {currency} {amount(row['amount'])} ({row['count']} transactions)" for row in rows[:5]]
        return "The largest inferred spending areas are:\n\n" + "\n".join(lines)

    if intent == "largest_transaction":
        rows = answer or []
        if not rows:
            return "I couldn't find outgoing transactions to rank."
        lines = []
        for row in rows[:5]:
            merchant = row.get("merchant") or row.get("description") or "Unknown"
            lines.append(f"- **{merchant}** — {currency} {row.get('amount', '0.00')} on {row.get('transaction_date') or 'date unavailable'}")
        return "Your largest outgoing transactions are:\n\n" + "\n".join(lines)

    if intent == "smallest_transaction":
        rows = answer or []
        if not rows:
            return "I couldn't find outgoing transactions to rank."
        lines = []
        for row in rows[:5]:
            merchant = row.get("merchant") or row.get("description") or "Unknown"
            lines.append(f"- **{merchant}** — {currency} {row.get('amount', '0.00')} on {row.get('transaction_date') or 'date unavailable'}")
        return "Your smallest outgoing transactions are:\n\n" + "\n".join(lines)

    if intent == "income":
        return f"Total money received in this statement: **{currency} {amount(answer.get('total_income', 0))}**."

    if intent == "expenses":
        return f"Total money spent in this statement: **{currency} {amount(answer.get('total_expenses', 0))}**."

    if intent == "net":
        net = float(answer.get("net_cash_flow", 0))
        direction = "positive" if net >= 0 else "negative"
        return f"Your net cash flow is **{currency} {amount(abs(net))}** ({direction})."

    if intent == "review":
        review_count = int(answer.get("review_count", 0))
        if review_count:
            return f"Finora currently has **{review_count} transaction(s)** marked for review, including **{int(answer.get('duplicate_review_count', 0))} possible duplicate(s)**."
        concentration = answer.get("largest_spending_concentration") or []
        if concentration:
            top = concentration[0]
            return (
                "No transactions are currently marked for review. "
                f"The largest spending concentration is **{top['merchant']} — {currency} {amount(top['amount'])}**. "
                "You can inspect those transactions if you want to understand the spending."
            )
        return "No transactions are currently marked for review."

    if intent == "summary":
        top = answer.get("top_merchant") if isinstance(answer, dict) else None
        top_text = f" Your largest merchant concentration is **{top['merchant']} — {currency} {amount(top['amount'])}**." if top else ""
        return (
            f"This statement contains **{answer.get('transactions', 0)} transactions**, "
            f"with **{currency} {amount(answer.get('income', 0))} received** and "
            f"**{currency} {amount(answer.get('expenses', 0))} spent**. "
            f"Net cash flow is **{currency} {amount(answer.get('net_cash_flow', 0))}**.{top_text}"
        )

    if intent == "merchant_spend" and isinstance(answer, dict):
        merchant = answer.get("merchant")
        if merchant:
            return f"You spent **{currency} {amount(answer.get('total', 0))}** at **{merchant}** across **{answer.get('count', 0)} transaction(s)**."
        return "I couldn't confidently identify the merchant you asked about in this statement."

    if intent == "identity" and isinstance(answer, dict):
        holders = answer.get("account_holders") or []
        if holders:
            return "The statement data identifies the account holder as **" + ", ".join(holders) + "**."
        return "The extracted transaction data does not contain enough account-holder information to identify whose statement this is."

    return None


def render_chat_history(history, limit=8):
    """Use native Streamlit chat messages so markdown, wrapping and mobile layout stay responsive."""
    for message in (history or [])[-limit:]:
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            # Do not pass a Unicode symbol as avatar. Streamlit may interpret
            # arbitrary strings as image paths, which causes a MediaFileStorageError.
            # Use the native assistant avatar and render the Finora identity inside the message.
            with st.chat_message("assistant"):
                st.markdown("**✦ Finora AI**")
                st.markdown(content)


def render_floating_finora_chat(transactions):
    if not transactions:
        return

    with st.popover("✦", key="finora_ai_popover", help="Ask Finora AI"):
        render("""
        <div class="finora-chat-header">
            <div class="ai-icon">✦</div>
            <div>
                <div class="ai-title">Finora AI</div>
                <div class="ai-subtitle">● Ready · Grounded in this statement</div>
            </div>
        </div>
        <div class="finora-chat-intro">
            Ask about merchants, spending, categories, dates, income, transactions or anything visible in your statement.
        </div>
        """)

        prompts = [
            "Where did I spend the most?",
            "What were my largest transactions?",
            "How much did I spend on Healthcare?",
            "Which spending areas should I review?",
        ]
        prompt_cols = st.columns(2)
        for idx, prompt in enumerate(prompts):
            with prompt_cols[idx % 2]:
                if st.button(prompt, key=f"finora_float_prompt_{idx}", width="stretch"):
                    with st.spinner("Finora is analyzing your statement..."):
                        answer = ask_finora(prompt, build_ai_context(transactions), transactions=transactions, history=st.session_state.chat_history)
                    _append_chat_turn(prompt, answer)
                    st.rerun()

        if st.session_state.chat_history:
            render_chat_history(st.session_state.chat_history, limit=8)

        with st.form("finora_floating_chat_form", clear_on_submit=True):
            question = st.text_input(
                "Ask Finora",
                placeholder="Ask about your finances...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send", type="primary", width="stretch")

        if send and question.strip():
            clean_question = question.strip()
            with st.spinner("Finora is analyzing your statement..."):
                answer = ask_finora(clean_question, build_ai_context(transactions), transactions=transactions, history=st.session_state.chat_history)
            _append_chat_turn(clean_question, answer)
            st.rerun()


def ask_finora(question, context, transactions=None, history=None):
    """Hybrid RAG assistant: exact deterministic answers first, LLM for explanation/open questions."""
    if not transactions:
        return "I don't have an analyzed statement yet. Upload a financial statement first, then I can answer from its transaction data."

    try:
        rag_context = StatementRAG(transactions).context(question, top_k=12)
    except Exception as exc:
        return f"I couldn't prepare the transaction evidence for this question. Please try again. ({exc})"

    deterministic = _deterministic_finora_answer(rag_context)
    intent = rag_context.get("intent", {}).get("intent", "general")

    # Common numerical/data questions should not depend on an LLM to calculate money.
    if deterministic and intent != "general":
        return deterministic

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            deterministic
            or "Finora can answer the common statement questions locally, but GROQ_API_KEY is not configured for open-ended AI questions."
        )

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    recent_history = (history or [])[-10:]

    system_prompt = """
You are Finora AI, a concise financial statement copilot.

Your source of truth is the transaction-grounded RAG evidence supplied below.
The user may have uploaded any bank, card, wallet, loan, investment, or other financial statement from any country.

STRICT RULES:
1. Never invent a merchant, transaction, amount, currency, date, account holder, category, or balance.
2. Never use the sample statement as a template for the user's current data.
3. Prefer computed_evidence and statement_analytics for totals and rankings. Do not recalculate them from prose.
4. If a category is inferred from merchant/description text, say that it is inferred when relevant.
5. If evidence is missing, say exactly what is unavailable instead of guessing.
6. For follow-up questions, use the recent conversation only to resolve the user's reference; always ground the final fact in the current statement evidence.
7. Keep answers short, readable and useful. Use bullets when listing transactions.
8. For questions asking what to review, identify concrete evidence such as review flags, duplicates, large concentrations, or inferred categories; do not give regulated financial advice.
9. Do not expose internal RAG implementation details unless the user asks.
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "STATEMENT CONTEXT:\n" + json.dumps(context, indent=2, default=str) +
                    "\n\nRAG EVIDENCE:\n" + json.dumps(rag_context, indent=2, default=str) +
                    "\n\nRECENT CONVERSATION:\n" + json.dumps(recent_history, indent=2, default=str) +
                    "\n\nCURRENT QUESTION:\n" + question
                ),
            },
        ],
        "temperature": 0.1,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer or deterministic or "I couldn't produce an answer from the available statement evidence."
    except requests.exceptions.Timeout:
        return deterministic or "Finora AI timed out while contacting the AI service. Please try again."
    except requests.exceptions.RequestException:
        return deterministic or "Finora AI could not reach the AI service right now. Please try again."
    except (KeyError, IndexError, TypeError):
        return deterministic or "Finora AI received an unexpected response from the AI service."
    except Exception:
        return deterministic or "Finora AI encountered an unexpected error."


# ============================================================
# TOP NAVIGATION — NATIVE BROWSER LINKS
# ============================================================

nav_brand, nav_links, nav_actions = st.columns([3.0, 5.0, 1.8])

with nav_brand:
    render("""
    <a class="finora-brand-link" href="?page=Home&new=1" title="Go to Finora AI home">
        <div class="topbar">
            <div class="brand">
                <div class="logo finora-f-logo">F</div>
                <div>
                    <div class="brand-name">Finora AI</div>
                    <div class="brand-sub">Intelligent Financial Intelligence</div>
                </div>
            </div>
        </div>
    </a>
    """)

with nav_links:
    active = st.session_state.page
    active_label = "AI Intelligence" if active == "AI" else active
    if active_label not in {"Overview", "Transactions", "AI Intelligence"}:
        active_label = "Overview"

    def nav_link(label, page):
        active_class = " active" if active_label == label else ""
        statement_id = str(st.session_state.get("statement_id") or "").strip()
        statement_query = (
            f"&statement_id={statement_id}"
            if statement_id
            else ""
        )
        return f'<a class="finora-nav-link{active_class}" href="?page={page}{statement_query}">{label}</a>'

    if active == "Review":
        render("""
        <div class="finora-nav">
            <div class="finora-nav-link active">Category Review</div>
        </div>
        """)
    else:
        render(f"""
        <div class="finora-nav">
            {nav_link("Overview", "Overview")}
            {nav_link("Transactions", "Transactions")}
            {nav_link("AI Intelligence", "AI")}
        </div>
        """)

with nav_actions:
    new_statement_clicked = st.button(
        "＋ New Statement",
        key="global_new_statement",
        width="stretch",
    )
    if new_statement_clicked:
        # Explicitly enter upload mode. We do not restore the previous
        # statement from disk while the user is choosing a new one.
        st.session_state.upload_mode = True
        st.session_state.transactions = []
        st.session_state.transactions_backup = []
        st.session_state.file_name = None
        st.session_state.statement_source_path = None
        st.session_state.statement_id = None
        st.session_state.statement_metadata = {}
        st.session_state.review_cursor = 0
        st.session_state.category_editor_version = int(st.session_state.get("category_editor_version", 0)) + 1
        st.session_state.chat_history = []
        st.session_state.ai_summary = None
        st.session_state.category_engine = None
        st.session_state.transaction_focus = None
        st.session_state.category_review_skipped = False
        st.session_state.pending_category_campaigns = []
        st.session_state.page = "Home"
        st.query_params.clear()
        st.query_params["page"] = "Upload"
        st.rerun()
    render("""<div class="ai-ready" style="margin-top:7px;text-align:center;">● AI READY</div>""")


transactions = st.session_state.transactions


# ============================================================
# HOME
# ============================================================

if not transactions and st.session_state.page in {
    "Home",
    "Overview",
    "Upload",
}:

    # ========================================================
    # FINORA LANDING / UPLOAD PAGE
    # ========================================================

    render("""
    <div class="landing-page">

        <div class="landing-visual" aria-hidden="true">
            <div class="orbit orbit-1"></div>
            <div class="orbit orbit-2"></div>
            <div class="orbit orbit-3"></div>

            <div class="landing-icon icon-chart">▥</div>
            <div class="landing-icon icon-bank">⌂</div>
            <div class="landing-icon icon-card">▭</div>
            <div class="landing-icon icon-ai">✦</div>

            <div class="landing-document">
                <div class="document-sheet">
                    <div class="document-line document-line-long"></div>
                    <div class="document-line"></div>
                    <div class="document-line"></div>
                </div>
            </div>
        </div>

        <div class="landing-title">
            Understand your money <span>in minutes.</span>
        </div>

        <div class="landing-subtitle">
            Upload your bank or card statement and Finora will detect its structure,
            extract transactions<br class="desktop-only">
            and prepare your financial intelligence.
        </div>

    </div>
    """)

    # --------------------------------------------------------
    # CENTERED UPLOAD CONTROL
    # --------------------------------------------------------

    upload_wrap = st.container()

    with upload_wrap:

        render("""
        <div class="landing-upload-card">
            <div class="landing-upload-icon">↥</div>
            <div class="landing-upload-title">Upload your statement</div>
            <div class="landing-upload-subtitle">PDF or CSV · Up to 200MB</div>
        </div>
        """)

        uploaded = st.file_uploader(
            "Choose financial statement",
            type=["pdf", "csv"],
            label_visibility="collapsed",
            key="home_statement_uploader",
        )

        if uploaded is not None:
            render(f"""
            <div class="landing-file-selected">
                <span class="file-dot"></span>
                <span>{escape(uploaded.name)}</span>
                <span class="file-ready">READY</span>
            </div>
            """)

        password = st.text_input(
            "PDF password",
            type="password",
            placeholder="Password only if your PDF is protected",
            label_visibility="collapsed",
            key="home_pdf_password",
        )

        analyze_clicked = st.button(
            "✦  Analyze my finances with Finora AI  →",
            type="primary",
            width="stretch",
            key="home_analyze_button",
        )

        if analyze_clicked:

            if uploaded is None:
                st.warning("Please choose a financial statement first.")

            else:

                loader = st.empty()

                try:
                    _analyze_uploaded_statement(
                        uploaded,
                        password,
                        loader,
                    )

                    loader.success(
                        f"Analysis complete — "
                        f"{len(st.session_state.transactions):,} "
                        f"transactions analyzed."
                    )

                    st.rerun()

                except Exception as exc:
                    loader.empty()

                    error_text = str(exc).lower()
                    password_related = any(
                        term in error_text
                        for term in (
                            "password",
                            "encrypted",
                            "decrypt",
                            "incorrect password",
                            "wrong password",
                        )
                    )

                    if password_related:
                        st.warning(
                            "Incorrect PDF password. Please enter the correct "
                            "password and try again."
                        )
                    else:
                        st.error(
                            "Finora could not read this statement. "
                            "Please check the file and try again."
                        )

                    # Keep the technical exception out of the customer-facing UI.
                    # It can still be inspected in the terminal during development.
                    print(f"Finora analysis error: {exc}")

    # --------------------------------------------------------
    # SECURITY NOTE
    # --------------------------------------------------------

    render("""
    <div class="landing-security">
        <span>▣</span>
        Your data is secure and private
    </div>
    """)

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    render("""
    <div class="landing-section-kicker">
        WHAT FINORA UNDERSTANDS
    </div>

    <div class="landing-section-title">
        From raw statements to real financial clarity.
    </div>
    """)

    feature_cols = st.columns(3, gap="large")

    features = [
        (
            "⌕",
            "Understand every transaction",
            "Extract and organize all transactions with accurate details, dates and financial fields."
        ),
        (
            "✿",
            "Find spending patterns",
            "See where your money goes, identify trends and surface unusual or large expenses."
        ),
        (
            "✦",
            "Ask Finora anything",
            "Get answers about your transactions, spending, income and statement activity."
        ),
    ]

    for col, (icon, title, description) in zip(feature_cols, features):
        with col:
            render(f"""
            <div class="landing-feature">
                <div class="landing-feature-icon">{icon}</div>
                <div class="landing-feature-title">{title}</div>
                <div class="landing-feature-text">{description}</div>
            </div>
            """)

    render("""
    <div style="height:55px;"></div>
    """)

    st.stop()


# DATA
# ============================================================

df = make_dataframe(transactions)

currency = get_currency(transactions)

income, expenses, net_cash_flow = (
    calculate_financials(transactions)
)

category_df = get_categories(transactions)

spending = spending_insights(transactions)
category_breakdown = spending["category_df"]
merchant_breakdown = spending["merchant_df"]
category_coverage = spending["category_coverage"]
top_category = spending["top_category"]
least_category = spending["least_category"]
top_merchant = spending["top_merchant"]
review_category = spending["review_category"]
review_scenario = spending["review_scenario"]

review_count = sum(
    bool(
        getattr(
            transaction,
            "requires_review",
            False,
        )
    )
    for transaction in transactions
)

credit_card_mode = is_credit_card_statement(transactions)
credit_card = card_snapshot(transactions)


def render_ai_assistant_page(transactions):
    """Render the Finora AI page in the requested dashboard + assistant layout."""
    render("""
    <div class="page-title">Finora Intelligence</div>
    <div class="page-subtitle">
        Your statement, your spending, and a conversational financial assistant in one place.
    </div>
    """)

    if not transactions:
        left, right = st.columns([1.55, 0.95], gap="large")
        with left:
            render("""
            <div class="ai-shell">
                <div class="ai-shell-title">Financial intelligence starts with your statement</div>
                <div class="ai-shell-subtitle">
                    Upload a bank, credit-card, wallet, loan, or other financial statement.
                    Finora will extract transactions and build the dashboard automatically.
                </div>
                <div style="height:22px"></div>
                <div class="feature">
                    <div class="feature-title">Universal statement analysis</div>
                    <div class="feature-text">Finora adapts to different financial statement structures.</div>
                </div>
                <div style="height:14px"></div>
                <div class="feature">
                    <div class="feature-title">RAG-powered questions</div>
                    <div class="feature-text">After analysis, ask about merchants, amounts, dates, categories and spending patterns.</div>
                </div>
            </div>
            """)
        with right:
            render("""
            <div class="ai-shell">
                <div class="ai-head">
                    <div class="ai-icon">✨</div>
                    <div>
                        <div class="ai-title">Spending Assistant</div>
                        <div class="ai-subtitle">Upload a statement to start chatting</div>
                    </div>
                </div>
                <div style="height:24px"></div>
                <div class="ai-message-bot">
                    Hi. I can explain your statement and answer questions using your actual transactions.
                </div>
            </div>
            """)

        st.markdown("### Upload and analyze your statement")
        uploaded = st.file_uploader("Financial statement", type=["pdf", "csv"], key="ai_statement_uploader_v3")
        password = st.text_input("PDF password", type="password", placeholder="Enter password only if the PDF is protected", key="ai_pdf_password_v3")
        if st.button("✦ Analyze statement with Finora AI", type="primary", width="stretch", key="ai_analyze_statement_v3"):
            if uploaded is None:
                st.warning("Please upload a financial statement first.")
            else:
                try:
                    loader = st.empty()
                    _analyze_uploaded_statement(uploaded, password, loader)
                    st.success(f"Analysis complete — {len(st.session_state.transactions):,} transactions analyzed.")
                    st.query_params["page"] = "AI"
                    st.session_state.page = "AI"
                    st.rerun()
                except Exception as exc:
                    st.error("Finora could not analyze this statement.")
                    st.exception(exc)
        return

    income, expenses, net = calculate_financials(transactions)
    currency = get_currency(transactions)
    bank_closing_balance = (
        get_bank_closing_balance(transactions)
        if not credit_card_mode
        else None
    )
    spending = spending_insights(transactions)
    category_df = spending["category_df"]
    merchant_df = spending["merchant_df"]

    left_col, right_col = st.columns([1.48, 0.92], gap="large")

    with left_col:
        render("""
        <div class="ai-shell" style="min-height:auto;">
            <div class="ai-shell-title">Financial overview</div>
            <div class="ai-shell-subtitle">Live numbers from the analyzed statement.</div>
        </div>
        """)
        if credit_card_mode:
            m1, m2 = st.columns(2)
            with m1:
                render(f"<div class='metric'><div class='metric-label'>CARD LIMIT</div><div class='metric-value'>{money(credit_card.get('card_limit', 0), currency)}</div><div class='metric-sub'>Credit limit on the statement</div></div>")
            with m2:
                render(f"<div class='metric'><div class='metric-label'>CURRENT BALANCE</div><div class='metric-value'>{money(credit_card.get('current_balance', 0), currency)}</div><div class='metric-sub'>Outstanding card balance</div></div>")
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                render(f"<div class='metric'><div class='metric-label'>AVAILABLE CREDIT</div><div class='metric-value'>{money(credit_card.get('available_limit', 0), currency)}</div><div class='metric-sub'>Credit still available</div></div>")
            with m4:
                render(f"<div class='metric'><div class='metric-label'>TOTAL PAYMENT DUE</div><div class='metric-value'>{money(credit_card.get('total_payment_due', 0), currency)}</div><div class='metric-sub'>Due on {escape(str(credit_card.get('payment_due_date') or 'the statement due date'))}</div></div>")
        else:
            m1, m2 = st.columns(2)
            with m1:
                render(f"<div class='metric'><div class='metric-label'>MONEY RECEIVED</div><div class='metric-value'>{money(income, currency)}</div><div class='metric-sub'>Total credited transactions</div></div>")
            with m2:
                render(f"<div class='metric'><div class='metric-label'>MONEY SPENT</div><div class='metric-value'>{money(expenses, currency)}</div><div class='metric-sub'>Total debited transactions</div></div>")
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                render(f"<div class='metric'><div class='metric-label'>NET MOVEMENT</div><div class='metric-value'>{money(net, currency)}</div><div class='metric-sub'>Received minus spent</div></div>")
            with m4:
                closing_text = money(bank_closing_balance, currency) if bank_closing_balance is not None else "Not available"
                closing_sub = "Ending account balance" if bank_closing_balance is not None else "Statement balance not detected"
                render(f"<div class='metric'><div class='metric-label'>CLOSING BALANCE</div><div class='metric-value'>{closing_text}</div><div class='metric-sub'>{closing_sub}</div></div>")

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        render("""<div class="section-head"><div class="section-title">Where your money went</div><div class="section-subtitle">Ask Finora about any of these merchants.</div></div>""")
        if not merchant_df.empty:
            for _, row in merchant_df.head(8).iterrows():
                name = escape(str(row["Merchant"]))
                amount = number(row["Amount"])
                pct = (amount / float(expenses) * 100) if float(expenses) else 0
                render(f"""
                <div style="padding:15px 0;border-bottom:1px solid #1d293b;">
                    <div style="display:flex;justify-content:space-between;gap:16px;">
                        <span style="font-size:1.05rem;font-weight:800;color:#f8fafc;">{name}</span>
                        <span style="font-size:1.05rem;font-weight:800;color:#c7d2fe;">{money(amount, currency)}</span>
                    </div>
                    <div style="margin-top:8px;height:8px;background:#172238;border-radius:999px;overflow:hidden;"><div style="width:{min(pct,100):.1f}%;height:100%;background:#818cf8;border-radius:999px;"></div></div>
                    <div style="margin-top:6px;font-size:.86rem;color:#94a3b8;">{pct:.1f}% of outgoing money</div>
                </div>
                """)
        else:
            st.info("Merchant spending data is not available yet.")
        if not category_df.empty:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            render("<div class='section-title'>Spending by category</div>")
            st.dataframe(category_df, use_container_width=True, hide_index=True)

    with right_col:
        render("""
        <div class="ai-shell" style="min-height:700px;">
            <div class="ai-head">
                <div class="ai-icon">✨</div>
                <div>
                    <div class="ai-title">Spending Assistant</div>
                    <div class="ai-subtitle">● Active · Analyzing your statement</div>
                </div>
            </div>
        """)
        if not st.session_state.chat_history:
            render("<div class='ai-message-bot'>I've analyzed your statement. Ask me about spending, merchants, dates, categories, or unusual transactions.</div>")
        for message in st.session_state.chat_history[-8:]:
            content = escape(str(message.get("content", ""))).replace("\n", "<br>")
            if message.get("role") == "user":
                render(f"<div class='ai-message-user'><strong>You</strong><br>{content}</div>")
            else:
                render(f"<div class='ai-message-bot'><strong>✨ Finora AI</strong><br><br>{content}</div>")
        prompts = [
            "Where did I spend the most?",
            "How much did I spend at my biggest merchant?",
            "Show my largest transactions",
            "What spending should I review?",
        ]
        for idx, prompt in enumerate(prompts):
            if st.button(prompt, width="stretch", key=f"ai_side_prompt_{idx}"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.spinner("Finora is analyzing..."):
                    answer = ask_finora(prompt, build_ai_context(transactions), transactions=transactions, history=st.session_state.chat_history)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
        with st.form("finora_side_chat_form", clear_on_submit=True):
            question = st.text_input("Ask about a transaction", placeholder="Ask about a transaction...", label_visibility="collapsed")
            send = st.form_submit_button("➤ Send", type="primary", width="stretch")
        if send and question.strip():
            clean_question = question.strip()
            st.session_state.chat_history.append({"role": "user", "content": clean_question})
            with st.spinner("Finora is analyzing..."):
                answer = ask_finora(clean_question, build_ai_context(transactions), transactions=transactions, history=st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
        render("</div>")



# ============================================================
# V2 CATEGORY REVIEW — BETWEEN EXTRACTION AND ANALYTICS
# ============================================================

if st.session_state.page == "Review":
    review_transactions = list(st.session_state.get("transactions") or [])

    if not review_transactions:
        st.session_state.page = "Home"
        st.query_params["page"] = "Overview"
        st.rerun()

    review_engine = st.session_state.get("category_engine")

    if review_engine is None:
        review_engine = HybridCategoryEngine(CategoryMemory())
        review_engine.classify_transactions(review_transactions)
        st.session_state.category_engine = review_engine
        _save_transaction_backup(review_transactions)

    review_items = review_engine.review_items(review_transactions)
    total_transactions = len(review_transactions)
    needs_review = len(review_items)
    categorized_count = total_transactions - needs_review
    progress = categorized_count / total_transactions if total_transactions else 1.0

    if needs_review == 0:
        st.session_state.category_review_skipped = False
        st.session_state.page = "Overview"
        st.query_params["page"] = "Overview"
        st.rerun()

    # The review screen deliberately handles one transaction at a time.
    # This keeps the UI focused and avoids a giant nested scroll area.
    cursor = int(st.session_state.get("review_cursor", 0) or 0)
    cursor = max(0, min(cursor, needs_review - 1))
    st.session_state.review_cursor = cursor

    current_item = review_items[cursor]
    index = current_item["index"]
    tx = current_item["transaction"]

    merchant = str(
        getattr(tx, "merchant", None)
        or getattr(tx, "description_raw", None)
        or "Unknown merchant"
    ).strip()
    description = str(getattr(tx, "description_raw", "") or "").strip()
    amount = number(getattr(tx, "original_amount", 0))
    tx_currency = str(
        getattr(tx, "original_currency", currency) or currency
    )
    date_value = getattr(tx, "transaction_date", None)
    date_text = (
        date_value.strftime("%d %b %Y")
        if date_value
        else "Date unavailable"
    )

    suggested = current_item.get("suggested_category")
    confidence = float(current_item.get("confidence", 0.0) or 0.0)

    category_icons = {
        "Food & Dining": "🍔",
        "Groceries": "🛒",
        "Transportation": "🚗",
        "Housing": "🏠",
        "Bills & Utilities": "💡",
        "Healthcare": "🏥",
        "Shopping": "🛍️",
        "Entertainment": "🎬",
        "Travel": "✈️",
        "Education": "📚",
        "Financial": "💳",
        "Income": "💰",
        "Transfers": "🔄",
        "Investments": "📈",
        "Taxes & Government": "🧾",
        "Cash": "💵",
        "Other": "📦",
        "Uncategorized": "📦",
        "Unknown": "❔",
    }

    def category_icon(category_name):
        return category_icons.get(str(category_name), "•")

    category_choices = list(review_engine.available_categories())

    # Preserve the model suggestion as the initial selection, but let the
    # reviewer override it immediately with a visual category grid.
    selected_key = f"review_selected_category_{index}"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = (
            suggested if suggested in category_choices else None
        )

    selected_category = st.session_state.get(selected_key)

    render(f"""
    <div class="category-review-shell">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:20px;">
            <div>
                <div class="category-review-kicker">FINORA AI · CATEGORY REVIEW</div>
                <div class="category-review-title">Let's organize your transactions.</div>
                <div class="category-review-subtitle">
                    Finora has already categorized the transactions it can identify confidently.
                    You only need to review the remaining items.
                </div>
            </div>
            <div class="category-review-progress-pill">
                <span class="category-review-progress-dot"></span>
                Reviewing {cursor + 1} of {needs_review}
            </div>
        </div>

        <div class="category-review-summary">
            <div class="category-review-stat">
                <div class="category-review-stat-label">Transactions analyzed</div>
                <div class="category-review-stat-value">{total_transactions:,}</div>
                <div class="category-review-stat-sub">Universal extraction completed</div>
            </div>
            <div class="category-review-stat">
                <div class="category-review-stat-label">Automatically categorized</div>
                <div class="category-review-stat-value">{categorized_count:,}</div>
                <div class="category-review-stat-sub">Ready for financial analysis</div>
            </div>
            <div class="category-review-stat">
                <div class="category-review-stat-label">Need your attention</div>
                <div class="category-review-stat-value">{needs_review:,}</div>
                <div class="category-review-stat-sub">Your choices improve future statements</div>
            </div>
        </div>

        <div class="category-review-progress">
            <span style="width:{progress * 100:.1f}%"></span>
        </div>
        <div class="category-review-progress-label">
            <span>{categorized_count:,} categorized</span>
            <span>{needs_review:,} remaining</span>
        </div>
    </div>
    """)

    head_left, head_right = st.columns([5.2, 1.0])
    with head_left:
        st.markdown(f"### Needs your attention · {needs_review:,}")
    with head_right:
        if st.button("✕ Continue", key="category_review_close", width="stretch"):
            st.session_state.category_review_skipped = True
            st.session_state.page = "Overview"
            st.query_params["page"] = "Overview"
            st.rerun()

    # --------------------------------------------------------
    # CURRENT TRANSACTION
    # --------------------------------------------------------
    render(f"""
    <div class="category-review-workspace">
        <div class="category-review-item-top">
            <div style="min-width:0;">
                <div class="category-review-item-kicker">Transaction {cursor + 1} of {needs_review}</div>
                <div class="category-review-merchant-row">
                    <div class="category-review-merchant-icon">{category_icon(suggested or "Other")}</div>
                    <div style="min-width:0;">
                        <div class="category-review-merchant">{escape(merchant)}</div>
                        <div class="category-review-description">
                            {escape(description)} · {escape(date_text)}
                        </div>
                    </div>
                </div>
            </div>
            <div class="category-review-amount">
                {escape(tx_currency)} {amount:,.2f}
            </div>
        </div>
        {
            "<div class='category-review-suggested-pill'>✦ Finora suggests "
            + escape(str(suggested))
            + f" · {confidence:.0%} confidence</div>"
            if suggested else
            "<div class='category-review-suggested-pill'>✦ Finora needs your help choosing a category</div>"
        }
    </div>
    """)

    st.markdown("<div class='category-review-section-label'>Choose a category</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='category-review-category-help'>Select the category that best describes this transaction. "
        "The selected category will be used in your spending analysis.</div>",
        unsafe_allow_html=True,
    )

    # Visual category grid — standard categories are now the primary action.
    grid_columns = 4
    for row_start in range(0, len(category_choices), grid_columns):
        row_categories = category_choices[row_start:row_start + grid_columns]
        cols = st.columns(grid_columns)
        for col, category in zip(cols, row_categories):
            with col:
                is_selected = selected_category == category
                if st.button(
                    f"{category_icon(category)}  {category}",
                    key=f"review_cat_button_{index}_{row_start}_{category}",
                    type="primary" if is_selected else "secondary",
                    width="stretch",
                ):
                    st.session_state[selected_key] = category
                    st.rerun()

    # --------------------------------------------------------
    # CUSTOM CATEGORY
    # --------------------------------------------------------
    with st.expander("✦  Create custom category", expanded=False):
        st.markdown(
            "<div class='category-review-custom'>"
            "<div class='category-review-custom-title'>Create a category that fits your life</div>"
            "<div class='category-review-custom-copy'>"
            "Use this only when none of the standard categories accurately describe the transaction. "
            "Custom categories can be remembered for future statements."
            "</div></div>",
            unsafe_allow_html=True,
        )

        new_category = st.text_input(
            "Category name",
            placeholder="e.g. Business Expenses",
            key=f"review_new_category_{index}",
        )
        new_subcategory = st.text_input(
            "Subcategory",
            placeholder="e.g. Client Meetings",
            key=f"review_new_subcategory_{index}",
        )
        apply_custom_all = st.checkbox(
            "Apply to all matching transactions",
            value=True,
            key=f"review_new_all_{index}",
        )

        if st.button(
            "Create & Apply →",
            key=f"review_create_apply_{index}",
            type="primary",
            width="stretch",
        ):
            clean_name = new_category.strip()
            clean_subcategory = new_subcategory.strip() or "General"

            if not clean_name:
                st.warning("Enter a category name first.")
            else:
                review_engine.create_custom_category(
                    clean_name,
                    clean_subcategory,
                )
                review_engine.learn_from_user(
                    tx,
                    clean_name,
                    clean_subcategory,
                    apply_all=apply_custom_all,
                    transactions=review_transactions,
                )
                st.session_state.transactions = review_transactions
                st.session_state.category_engine = review_engine
                _save_transaction_backup(review_transactions)
                st.session_state.review_cursor = 0
                st.rerun()

    # --------------------------------------------------------
    # SUBCATEGORY / SAVE
    # --------------------------------------------------------
    selected_category = st.session_state.get(selected_key)

    if selected_category:
        st.markdown(
            f"<div class='category-review-section-label'>Selected: {escape(category_icon(selected_category) + ' ' + selected_category)}</div>",
            unsafe_allow_html=True,
        )

        subcategories = review_engine.subcategories_for(selected_category)
        selected_subcategory = st.selectbox(
            "Subcategory",
            ["General"] + [
                value for value in subcategories
                if value != "General"
            ],
            key=f"review_subcategory_{index}",
        )

        apply_all = st.checkbox(
            "Apply to all matching transactions",
            value=False,
            key=f"review_apply_all_{index}",
        )

        if apply_all:
            target_key = merchant_key(tx)
            matching_count = sum(
                1
                for other in review_transactions
                if merchant_key(other) == target_key
            )
            st.markdown(
                f"<div class='category-review-match'>✓ {matching_count} matching transaction(s) will be updated.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([2.2, 1.0])

        with b1:
            if st.button(
                "✓ Save & Next →",
                key=f"review_save_{index}",
                type="primary",
                width="stretch",
            ):
                review_engine.learn_from_user(
                    tx,
                    selected_category,
                    (
                        None
                        if selected_subcategory == "General"
                        else selected_subcategory
                    ),
                    apply_all=apply_all,
                    transactions=review_transactions,
                )
                st.session_state.transactions = review_transactions
                st.session_state.category_engine = review_engine
                _save_transaction_backup(review_transactions)
                st.session_state.review_cursor = 0
                st.rerun()

        with b2:
            if st.button(
                "Leave uncategorized",
                key=f"review_leave_{index}",
                width="stretch",
            ):
                tx.category = "Uncategorized"
                tx.subcategory = "Uncategorized"
                tx.category_confidence = 0.0
                tx.requires_review = False
                st.session_state.transactions = review_transactions
                st.session_state.category_engine = review_engine
                _save_transaction_backup(review_transactions)
                st.session_state.review_cursor = 0
                st.rerun()
    else:
        st.info("Select a category above to continue.")

    # --------------------------------------------------------
    # REVIEW NAVIGATION
    # --------------------------------------------------------
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns([1.0, 1.5, 1.0])

    with nav1:
        if st.button(
            "← Previous",
            key=f"review_previous_{index}",
            disabled=cursor <= 0,
            width="stretch",
        ):
            st.session_state.review_cursor = max(0, cursor - 1)
            st.rerun()

    with nav2:
        st.markdown(
            f"<div style='text-align:center;color:#64748b;font-size:.72rem;padding-top:10px;'>"
            f"{cursor + 1} / {needs_review} · {needs_review - cursor - 1} remaining after this"
            f"</div>",
            unsafe_allow_html=True,
        )

    with nav3:
        if st.button(
            "Next →",
            key=f"review_next_{index}",
            disabled=cursor >= needs_review - 1,
            width="stretch",
        ):
            st.session_state.review_cursor = min(needs_review - 1, cursor + 1)
            st.rerun()

    remaining = len(review_engine.review_items(review_transactions))

    render(f"""
    <div class="category-review-footer">
        <div class="category-review-footer-text">
            <strong>{remaining:,}</strong> transaction(s) still need attention.
            Saved choices are written directly into Finora's transaction data
            and will feed the spending dashboard, Transactions page and AI.
        </div>
    </div>
    """)

    st.stop()


# ============================================================

# ============================================================

def overview_category_icon(category_name):
    overrides = st.session_state.get("category_icon_overrides") or {}
    icons = {
        "Food & Dining": "🍔",
        "Groceries": "🛒",
        "Transportation": "🚗",
        "Housing": "🏠",
        "Bills & Utilities": "💡",
        "Healthcare": "🏥",
        "Shopping": "🛍️",
        "Entertainment": "🎬",
        "Travel": "✈️",
        "Education": "📚",
        "Financial": "💳",
        "Income": "💰",
        "Transfers": "🔄",
        "Investments": "📈",
        "Taxes & Government": "🧾",
        "Cash": "💵",
        "Other": "📦",
        "Uncategorized": "📦",
        "Unknown": "❔",
    }
    name = str(category_name or "").strip()
    return overrides.get(name, icons.get(name, "📦"))


if st.session_state.page == "Overview":

    # ========================================================
    # PDF EXPORT — ALWAYS BUILT FROM CURRENT TRANSACTION DATA
    # ========================================================

    pdf_left, pdf_right = st.columns([4.9, 1.1])
    with pdf_right:
        try:
            overview_pdf = build_finora_report(
                st.session_state.get("transactions") or [],
                file_name=st.session_state.get("file_name"),
                statement_metadata=get_statement_metadata(),
            )
            st.download_button(
                "📄 Download Overview PDF",
                data=overview_pdf,
                file_name="finora_financial_intelligence_report.pdf",
                mime="application/pdf",
                key="download_overview_pdf_live_v4",
                width="stretch",
                help="Generated from the latest saved transaction categories and totals.",
            )
        except Exception as exc:
            print(f"Finora PDF report error: {exc}")

    render("""
    <div class="category-editor-note" style="margin-top:8px;margin-bottom:18px;">
        📄 <strong>Live report:</strong> the Overview PDF is generated from the current transaction data.
        Any category correction you save is reflected automatically in the dashboard and the next PDF download.
    </div>
    """)

    # ========================================================
    # FINORA FINANCIAL COCKPIT
    # ========================================================

    debit_df = df[
        df["Direction"].astype(str).str.lower() == "debit"
    ].copy()

    if not debit_df.empty:
        merchant_rows = (
            debit_df.groupby(
                debit_df["Merchant"].fillna(
                    debit_df["Description"]
                ).astype(str)
            )["Amount"]
            .sum()
            .sort_values(ascending=False)
        )
    else:
        merchant_rows = pd.Series(dtype=float)

    def _display_spend_name(value):
        raw = str(value).strip()

        if not raw:
            return "Unknown payment"

        upper = raw.upper()

        if (
            upper.startswith("UPIOUT")
            or upper.startswith("UPI OUT")
            or upper.startswith("UPI/")
        ):
            return "UPI transfer"

        if "PAYTM" in upper:
            return "Paytm"

        return raw

    bank_closing_balance = (
        get_bank_closing_balance(transactions)
        if not credit_card_mode
        else None
    )

    has_categories = (
        category_coverage > 0
        and not category_breakdown.empty
    )

    if has_categories:
        spend_source = category_breakdown.copy()
        spend_label = "categories"
    else:
        spend_source = merchant_breakdown.copy()
        spend_label = "merchants"

    # --------------------------------------------------------
    # MAIN STORY
    # --------------------------------------------------------

    if credit_card_mode:
        story = (
            f"Your current card balance is <strong>{money(credit_card.get('current_balance', 0), currency)}</strong> "
            f"against a <strong>{money(credit_card.get('card_limit', 0), currency)}</strong> credit limit, "
            f"leaving <strong>{money(credit_card.get('available_limit', 0), currency)}</strong> available."
        )
        pill = "Credit card statement"
    elif bank_closing_balance is not None:
        story = (
            f"You received {money(income, currency)} and spent "
            f"{money(expenses, currency)} during this statement period. "
            f"Your closing balance is "
            f"<strong>{money(bank_closing_balance, currency)}</strong>."
        )
        pill = "Closing balance"
    elif net_cash_flow > 0:
        story = (
            f"You received {money(income, currency)} and spent "
            f"{money(expenses, currency)}. "
            f"That leaves a positive net position of "
            f"<strong>{money(net_cash_flow, currency)}</strong>."
        )
        pill = "Positive cash position"
    elif net_cash_flow < 0:
        story = (
            f"You received {money(income, currency)} and spent "
            f"{money(expenses, currency)}. "
            f"Spending was higher than incoming money by "
            f"<strong>{money(abs(net_cash_flow), currency)}</strong>."
        )
        pill = "Spending exceeded incoming money"
    else:
        story = (
            "Incoming and outgoing money were approximately balanced "
            "across this statement."
        )
        pill = "Balanced cash movement"

    render(f"""
    <div class="finora-cockpit">

        <div class="cockpit-kicker">
            FINORA · FINANCIAL COCKPIT
        </div>

        <div class="cockpit-title">
            Here's the financial story.
        </div>

        <div class="cockpit-copy">
            {st.session_state.file_name} · {currency} ·
            {len(transactions):,} transactions analyzed
        </div>

        <div class="cockpit-story">
            {story}
        </div>

        <div class="cockpit-pill">
            <span></span>
            {pill}
        </div>

        <div class="cockpit-metrics">

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">{"Card limit" if credit_card_mode else "Received"}</div>
                <div class="cockpit-metric-value">
                    {money(credit_card.get('card_limit', 0), currency) if credit_card_mode else money(income, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    {"maximum card limit" if credit_card_mode else "money coming in"}
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">{"Current balance" if credit_card_mode else "Spent"}</div>
                <div class="cockpit-metric-value">
                    {money(credit_card.get('current_balance', 0), currency) if credit_card_mode else money(expenses, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    {"outstanding card balance" if credit_card_mode else "money going out"}
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">{"Available credit" if credit_card_mode else "Net movement"}</div>
                <div class="cockpit-metric-value">
                    {money(credit_card.get('available_limit', 0), currency) if credit_card_mode else money(net_cash_flow, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    {"credit still available" if credit_card_mode else "received minus spent"}
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">{"Total payment due" if credit_card_mode else "Closing balance"}</div>
                <div class="cockpit-metric-value">
                    {money(credit_card.get('total_payment_due', 0), currency) if credit_card_mode else (money(bank_closing_balance, currency) if bank_closing_balance is not None else "Not available")}
                </div>
                <div class="cockpit-metric-sub">
                    {f"Due {credit_card.get('payment_due_date')}" if credit_card_mode and credit_card.get('payment_due_date') else ("ending account balance" if bank_closing_balance is not None else "statement balance not detected")}
                </div>
            </div>

        </div>

    </div>
    """)

    # --------------------------------------------------------
    # OVERVIEW ACTIONS
    # --------------------------------------------------------
    # The primary Overview export is the PDF button above.
    # CSV export remains available only on the Transactions page.
    with st.container():
        if st.button(
            "✏️ Review & Edit Categories",
            key="overview_review_categories_v3",
            width="stretch",
        ):
            st.session_state.open_category_editor = True
            st.session_state.page = "Transactions"
            st.query_params["page"] = "Transactions"
            st.rerun()

    # --------------------------------------------------------
    # QUICK INTELLIGENCE
    # --------------------------------------------------------

    render("""
    <div class="intel-section">

        <div class="intel-head">
            <div>
                <div class="intel-title">
                    What stands out
                </div>
                <div class="intel-sub">
                    Finora surfaces the strongest signals first.
                </div>
            </div>
            <div class="intel-meta">
                Based on extracted statement data
            </div>
        </div>

    </div>
    """)

    focus_left, focus_right = st.columns(
        [1.05, .95],
        gap="large",
    )

    # Find largest merchant and smallest meaningful merchant.
    largest_name = "No outgoing activity"
    largest_amount = 0.0

    smallest_name = "Not available"
    smallest_amount = 0.0

    if not merchant_rows.empty:

        largest_raw = merchant_rows.index[0]
        largest_name = _display_spend_name(largest_raw)
        largest_amount = float(merchant_rows.iloc[0])

        meaningful = merchant_rows[
            merchant_rows > 0
        ].sort_values()

        if not meaningful.empty:
            smallest_raw = meaningful.index[0]
            smallest_name = _display_spend_name(
                smallest_raw
            )
            smallest_amount = float(
                meaningful.iloc[0]
            )

    largest_share = (
        largest_amount / float(expenses) * 100
        if float(expenses) > 0
        else 0
    )

    with focus_left:

        render(f"""
        <div class="focus-card">

            <div class="focus-label">
                Biggest outgoing destination
            </div>

            <div class="focus-title">
                {largest_name}
            </div>

            <div class="focus-amount">
                {money(largest_amount, currency)}
            </div>

            <div class="focus-copy">
                This is the largest identifiable outgoing destination
                in the statement.
            </div>

            <div class="focus-progress">
                <span style="width:{min(100, largest_share):.1f}%"></span>
            </div>

            <div class="focus-copy">
                {largest_share:.1f}% of outgoing money
            </div>

        </div>
        """)
        if largest_amount > 0 and largest_name != "No outgoing activity":
            if st.button(f"View {largest_name} transactions", key="view_largest_spending", width="stretch"):
                set_transaction_focus("category" if has_categories else "merchant", largest_name)
                st.rerun()

    with focus_right:

        render(f"""
        <div class="focus-card">

            <div class="focus-label">
                Smallest outgoing destination
            </div>

            <div class="focus-title">
                {smallest_name}
            </div>

            <div class="focus-amount">
                {money(smallest_amount, currency)}
            </div>

            <div class="focus-copy">
                Useful for understanding the long tail of small
                payments. A small payment is not automatically
                unnecessary.
            </div>

        </div>
        """)

    # --------------------------------------------------------
    # SPENDING RANKING + MONEY FLOW
    # --------------------------------------------------------

    render("""
    <div class="intel-section">

        <div class="intel-head">
            <div>
                <div class="intel-title">
                    Where the money went
                </div>
                <div class="intel-sub">
                    The largest outgoing areas appear first.
                </div>
            </div>
            <div class="intel-meta">
                """ + (
                    "Category intelligence"
                    if has_categories
                    else "Merchant intelligence"
                ) + """
            </div>
        </div>

    </div>
    """)

    rank_col, flow_col = st.columns(
        [1.0, 1.0],
        gap="large",
    )

    with rank_col:

        render("""
        <div class="rank-card">
        """)

        if not spend_source.empty:

            top_items = spend_source.head(6).copy()
            max_amount = float(top_items["Amount"].max())
            total_spend = float(expenses)

            for position, (_, row) in enumerate(
                top_items.iterrows(),
                start=1,
            ):

                raw_name = (
                    row["Category"]
                    if has_categories
                    else row["Merchant"]
                )
                display_name = (
                    str(raw_name)
                    if has_categories
                    else _display_spend_name(raw_name)
                )
                amount = float(row["Amount"])
                share = amount / total_spend * 100 if total_spend > 0 else 0
                width = amount / max_amount * 100 if max_amount > 0 else 0

                row_left, row_right = st.columns([0.76, 0.24], gap="small")
                with row_left:
                    icon = (
                        overview_category_icon(display_name)
                        if has_categories
                        else "🏪"
                    )
                    safe_key = re.sub(
                        r"[^a-z0-9]+",
                        "_",
                        display_name.casefold(),
                    ).strip("_") or "item"
                    if st.button(
                        f"{position:02d}  {icon}  {display_name}",
                        key=f"overview_spend_{position}_{safe_key}",
                        width="stretch",
                        help="Open the transactions behind this spending total.",
                    ):
                        set_transaction_focus(
                            "category" if has_categories else "merchant",
                            display_name,
                        )
                        st.rerun()

                with row_right:
                    st.markdown(
                        f"<div class='rank-amount' style='text-align:right;padding-top:9px;'>{money(amount, currency)}</div>",
                        unsafe_allow_html=True,
                    )

                render(f"""
                <div class="rank-track">
                    <span style="width:{min(100, width):.1f}%"></span>
                </div>
                <div class="rank-sub">
                    {share:.1f}% of outgoing money · click the category to inspect transactions
                </div>
                """)

        else:

            render("""
            <div style="
                padding:25px 0;
                color:#64748b;
                font-size:.64rem;
            ">
                No outgoing transactions were available to rank.
            </div>
            """)

        render("</div>")

    with flow_col:

        render("""
        <div class="flow-card">

            <div class="intel-title">
                How money moved
            </div>

            <div class="intel-sub">
                A simple period view of money received, money spent
                and the resulting movement.
            </div>
        """)

        chart_df = df.copy()

        if not chart_df.empty:

            chart_df["Date"] = pd.to_datetime(
                chart_df["Date"],
                errors="coerce",
            )

            chart_df = chart_df.dropna(
                subset=["Date"]
            )

        if not chart_df.empty:

            chart_df["Period"] = (
                chart_df["Date"]
                .dt.to_period("W")
                .apply(lambda x: x.start_time)
            )

            income_by_period = (
                chart_df[
                    chart_df["Direction"] == "credit"
                ]
                .groupby("Period")["Amount"]
                .sum()
            )

            expense_by_period = (
                chart_df[
                    chart_df["Direction"] == "debit"
                ]
                .groupby("Period")["Amount"]
                .sum()
            )

            periods = sorted(
                set(income_by_period.index)
                | set(expense_by_period.index)
            )

            # Avoid an empty-looking chart for statements that only
            # contain one time bucket.
            if len(periods) >= 2:

                labels = [
                    pd.Timestamp(period).strftime("%d %b")
                    for period in periods
                ]

                received_values = [
                    float(
                        income_by_period.get(period, 0)
                    )
                    for period in periods
                ]

                spent_values = [
                    float(
                        expense_by_period.get(period, 0)
                    )
                    for period in periods
                ]

                net_values = [
                    received - spent
                    for received, spent in zip(
                        received_values,
                        spent_values,
                    )
                ]

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=received_values,
                        name="Received",
                        mode="lines+markers",
                        line={
                            "width":2.8,
                            "color":"#7dd3fc",
                        },
                        marker={
                            "size":7,
                            "line":{"width":0},
                        },
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Received: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=spent_values,
                        name="Spent",
                        mode="lines+markers",
                        line={
                            "width":2.8,
                            "color":"#6366f1",
                        },
                        marker={
                            "size":7,
                            "line":{"width":0},
                        },
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Spent: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=labels,
                        y=net_values,
                        name="Net",
                        mode="lines+markers",
                        line={
                            "width":1.8,
                            "dash":"dot",
                            "color":"#f0a0a8",
                        },
                        marker={
                            "size":5,
                            "line":{"width":0},
                        },
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Net: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={
                        "color":"#94a3b8",
                        "family":"Inter, system-ui, sans-serif",
                    },
                    margin={
                        "l":0,
                        "r":0,
                        "t":18,
                        "b":4,
                    },
                    hovermode="x unified",
                    hoverlabel={
                        "bgcolor":"#111827",
                        "bordercolor":"#334155",
                        "font":{
                            "color":"#f8fafc",
                            "family":"Inter, system-ui, sans-serif",
                            "size":12,
                        },
                    },
                    xaxis={
                        "showgrid":False,
                        "zeroline":False,
                        "fixedrange":True,
                        "tickfont":{"size":12},
                        "automargin":True,
                    },
                    yaxis={
                        "showgrid":True,
                        "gridcolor":"rgba(148,163,184,0.10)",
                        "zeroline":True,
                        "zerolinecolor":"rgba(148,163,184,0.28)",
                        "zerolinewidth":1,
                        "fixedrange":True,
                        "tickfont":{"size":12},
                        "tickformat":"~s",
                        "automargin":True,
                    },
                    legend={
                        "orientation":"h",
                        "yanchor":"top",
                        "y":-0.04,
                        "x":0,
                        "font":{"size":12},
                    },
                    showlegend=True,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config={
                        "displayModeBar":False,
                        "responsive":True,
                        "scrollZoom":False,
                    },
                )

            else:

                render("""
                <div class="flow-empty">
                    This statement has one main time period,
                    so Finora is showing the totals above instead
                    of stretching a misleading chart.
                </div>
                """)

        else:

            render("""
            <div class="flow-empty">
                Reliable transaction dates were not available
                for a movement chart.
            </div>
            """)

        render("</div>")

    # --------------------------------------------------------
    # FINORA'S INTERPRETATION
    # --------------------------------------------------------

    render("""
    <div class="intel-section">

        <div class="intel-head">
            <div>
                <div class="intel-title">
                    Finora's read
                </div>
                <div class="intel-sub">
                    The statement explained in plain language.
                </div>
            </div>
        </div>

    </div>
    """)

    if largest_amount > 0:

        if has_categories:
            focus_text = (
                f"{largest_name} is the largest spending category "
                f"at {money(largest_amount, currency)}, representing "
                f"{largest_share:.1f}% of outgoing money."
            )
        else:
            focus_text = (
                f"{largest_name} is the largest identifiable spending "
                f"destination at {money(largest_amount, currency)}, "
                f"representing {largest_share:.1f}% of outgoing money."
            )

    else:
        focus_text = "Finora could not identify a meaningful outgoing destination."

    category_text = (
        f"Finora has {category_coverage:.0f}% reliable category coverage."
        if category_coverage > 0
        else
        "Category coverage is currently 0%, so Finora is using "
        "merchant-level intelligence rather than inventing categories."
    )

    render(f"""
    <div class="read-card">

        <div class="read-kicker">
            FINORA'S INTERPRETATION
        </div>

        <div class="read-main">
            {focus_text}
            {category_text}
            Review items currently marked for attention:
            <strong>{review_count}</strong>.
        </div>

    </div>
    """)

    # --------------------------------------------------------
    # WHERE TO LOOK
    # --------------------------------------------------------

    render("""
    <div class="intel-section">

        <div class="intel-head">
            <div>
                <div class="intel-title">
                    Where should you look first?
                </div>
                <div class="intel-sub">
                    Finora highlights areas worth investigating;
                    it does not decide what you should cut.
                </div>
            </div>
        </div>

    </div>
    """)

    action_left, action_right = st.columns(
        2,
        gap="large",
    )

    with action_left:

        render(f"""
        <div class="action-card">

            <div class="action-number">
                01 · INVESTIGATE
            </div>

            <div class="action-title">
                Start with {largest_name}
            </div>

            <div class="action-copy">
                {money(largest_amount, currency)} is the largest
                identifiable outgoing amount. Check whether it is
                essential, recurring, business-related or discretionary.
            </div>

        </div>
        """)

    with action_right:

        if has_categories:

            review_title = "Look at the highest discretionary category"

            review_copy = (
                "Finora can compare categories and estimate where "
                "a reduction would have the largest effect."
            )

        else:

            review_title = "Unlock category intelligence"

            review_copy = (
                "Reliable categories are not available for this "
                "statement yet. Merchant intelligence is being shown "
                "instead of making unsupported spending claims."
            )

        render(f"""
        <div class="action-card">

            <div class="action-number">
                02 · NEXT SIGNAL
            </div>

            <div class="action-title">
                {review_title}
            </div>

            <div class="action-copy">
                {review_copy}
            </div>

        </div>
        """)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # Recent activity belongs in Transactions; keep Overview focused.
    render("""
    <div style="color:#64748b;font-size:.78rem;text-align:center;padding:10px 0 80px;">
        Full transaction-level inspection is available in <strong style="color:#94a3b8;">Transactions</strong>.
        Ask Finora AI anytime using the floating button.
    </div>
    """)
    render_floating_finora_chat(transactions)



if st.session_state.page == "Transactions":

    render("""
    <div class="page-title">
        Transactions
    </div>

    <div class="page-subtitle">
        Search, filter, inspect and correct transaction categories.
    </div>
    """)

    transaction_source = list(
        st.session_state.get("transactions") or []
    )

    transaction_df = make_dataframe(transaction_source)

    if transaction_df.empty and transaction_source:
        transaction_df = pd.DataFrame(
            [transaction_dict(t) for t in transaction_source]
        )

    # --------------------------------------------------------
    # CATEGORY REVIEW / EDITOR
    # --------------------------------------------------------
    open_editor = bool(
        st.session_state.pop("open_category_editor", False)
    )

    edit_categories = st.toggle(
        "✏️ Review & edit transaction categories",
        value=open_editor,
        key="transaction_category_editor_toggle_v2",
        help=(
            "Review every transaction, change incorrect categories, "
            "and save the corrections to Finora's category memory."
        ),
    )

    if edit_categories and transaction_source:
        editor_engine = st.session_state.get("category_engine")

        if editor_engine is None:
            editor_engine = HybridCategoryEngine(CategoryMemory())
            editor_engine.classify_transactions(transaction_source)
            st.session_state.category_engine = editor_engine
            _save_transaction_backup(transaction_source)

        standard_category_icons = {
            "Food & Dining": "🍔",
            "Groceries": "🛒",
            "Transportation": "🚗",
            "Housing": "🏠",
            "Bills & Utilities": "💡",
            "Healthcare": "🏥",
            "Shopping": "🛍️",
            "Entertainment": "🎬",
            "Travel": "✈️",
            "Education": "📚",
            "Financial": "💳",
            "Income": "💰",
            "Transfers": "🔄",
            "Investments": "📈",
            "Taxes & Government": "🧾",
            "Cash": "💵",
            "Other": "📦",
            "Uncategorized": "📦",
            "Unknown": "❔",
        }
        overrides = st.session_state.get("category_icon_overrides") or {}

        def editor_category_icon(name):
            return overrides.get(name, standard_category_icons.get(name, "📦"))

        category_names = list(dict.fromkeys(editor_engine.available_categories()))
        if "Uncategorized" not in category_names:
            category_names.append("Uncategorized")

        # Review mode deliberately hides already-set categories. It shows only
        # transactions where Finora has a review flag or a generic/empty label.
        review_candidate_indices = []
        for source_index, tx in enumerate(transaction_source):
            category_value = str(getattr(tx, "category", "") or "").strip()
            is_generic = category_value.casefold() in {
                "",
                "other",
                "uncategorized",
                "unknown",
            }
            if bool(getattr(tx, "requires_review", False)) or is_generic:
                review_candidate_indices.append(source_index)

        st.markdown(
            """
            <div class="category-editor-note">
                <strong>Category review mode:</strong> by default, Finora shows only
                transactions that need a category decision. Already-categorized
                transactions stay hidden so you can focus on unclear items without
                seeing their existing category.
            </div>
            """,
            unsafe_allow_html=True,
        )

        show_all_editor = st.checkbox(
            "Show already-categorized transactions",
            value=False,
            key="review_show_all_categories_v4",
            help="Enable this only when you want to change a category that is already set.",
        )

        editor_indices = (
            list(range(len(transaction_source)))
            if show_all_editor
            else review_candidate_indices
        )

        # Search + category filtering inside the category-review workspace.
        # Search covers the full statement; the category filter narrows the
        # current result set without changing the underlying transactions.
        filter_col1, filter_col2 = st.columns([2.2, 1.0], gap="small")

        with filter_col1:
            review_search = st.text_input(
                "Search transactions",
                placeholder="Search merchant, description, or transaction ID...",
                key="category_review_search_v2",
                help=(
                    "Search the full statement to quickly find any transaction. "
                    "Clear the search to return to the normal category-review queue."
                ),
            )

        # Use the categories actually present in the statement, while also
        # keeping the standard/custom category list available to the filter.
        present_categories = []
        for tx in transaction_source:
            raw_category = str(getattr(tx, "category", "") or "").strip()
            normalized_category = raw_category or "Uncategorized"
            if normalized_category not in present_categories:
                present_categories.append(normalized_category)

        category_filter_options = ["All"] + [
            name for name in category_names
            if name in present_categories
        ]
        for name in present_categories:
            if name not in category_filter_options:
                category_filter_options.append(name)

        with filter_col2:
            review_category_filter = st.selectbox(
                "Category",
                category_filter_options,
                key="category_review_filter_v2",
                help="Filter the category-review list by its current category.",
            )

        if review_search.strip():
            query = review_search.strip().casefold()
            editor_indices = [
                source_index
                for source_index in editor_indices
                if query in str(
                    getattr(transaction_source[source_index], "merchant", None)
                    or getattr(transaction_source[source_index], "description_raw", None)
                    or ""
                ).casefold()
                or query in str(
                    getattr(transaction_source[source_index], "description_raw", None)
                    or ""
                ).casefold()
                or query in str(
                    getattr(transaction_source[source_index], "transaction_id", None)
                    or ""
                ).casefold()
            ]

        if review_category_filter != "All":
            target_category = review_category_filter.casefold()
            editor_indices = [
                source_index
                for source_index in editor_indices
                if (
                    str(
                        getattr(transaction_source[source_index], "category", "")
                        or "Uncategorized"
                    ).strip().casefold()
                    == target_category
                )
            ]

        if review_search.strip() or review_category_filter != "All":
            st.caption(
                f"Showing {len(editor_indices):,} matching transaction(s). "
                "Search and category filters apply to the review list."
            )

        # ----------------------------------------------------
        # ADD CUSTOM CATEGORY
        # ----------------------------------------------------
        with st.expander("➕ Add a new category", expanded=False):
            st.markdown(
                "Use your own category when Finora's standard list does not fit.",
            )
            custom_col1, custom_col2 = st.columns([1.9, 1.0], gap="small")
            with custom_col1:
                custom_name = st.text_input(
                    "Category name",
                    placeholder="e.g. Business Expenses",
                    key="review_custom_category_name_v4",
                )
            with custom_col2:
                custom_icon = st.selectbox(
                    "Icon",
                    [
                        "📦", "💼", "🧾", "🎓", "🏢", "❤️", "🛒", "🍔",
                        "🚗", "✈️", "🎬", "💡", "💳", "💰", "📈", "🔄",
                    ],
                    key="review_custom_category_icon_v4",
                )
            custom_subcategory = st.text_input(
                "Subcategory (optional)",
                placeholder="e.g. Client Meetings",
                key="review_custom_category_sub_v4",
            )
            if st.button(
                "Create category",
                key="review_create_category_v4",
                type="primary",
                width="stretch",
            ):
                clean_name = custom_name.strip()
                clean_sub = custom_subcategory.strip() or "General"
                existing = {name.casefold() for name in category_names}
                if not clean_name:
                    st.warning("Enter a category name first.")
                elif clean_name.casefold() in existing:
                    st.warning(f"The category **{clean_name}** already exists.")
                else:
                    editor_engine.create_custom_category(clean_name, clean_sub)
                    overrides[clean_name] = custom_icon
                    st.session_state.category_icon_overrides = overrides
                    st.session_state.category_engine = editor_engine
                    st.session_state.category_editor_version = int(
                        st.session_state.get("category_editor_version", 0)
                    ) + 1
                    _save_statement_context()
                    st.success(f"Created **{custom_icon} {clean_name}**. It is now available in the category picker.")
                    st.rerun()

        if editor_indices:
            editor_rows = []
            for source_index in editor_indices:
                tx = transaction_source[source_index]
                row = transaction_dict(tx)
                row["Transaction ID"] = getattr(tx, "transaction_id", "")
                # In focused review mode, hide the existing category. In full-edit mode,
                # display the current value using the same emoji labels used by the picker.
                if not show_all_editor:
                    row["Category"] = ""
                else:
                    current_category = str(row.get("Category") or "").strip()
                    row["Category"] = (
                        f"{editor_category_icon(current_category)}  {current_category}"
                        if current_category in category_names
                        else ""
                    )
                editor_rows.append(row)

            editor_df = pd.DataFrame(editor_rows)
            editor_columns = [
                "Transaction ID",
                "Date",
                "Merchant",
                "Amount",
                "Currency",
                "Direction",
                "Category",
                "Review",
            ]
            editor_df = editor_df[editor_columns]

            category_labels = [
                f"{editor_category_icon(name)}  {name}"
                for name in category_names
            ]
            category_label_to_name = dict(
                zip(category_labels, category_names)
            )

            with st.form(
                f"transaction_category_review_form_v{st.session_state.category_editor_version}",
                clear_on_submit=False,
            ):
                edited_df = st.data_editor(
                    editor_df,
                    width="stretch",
                    hide_index=True,
                    height=600,
                    key=f"transaction_category_review_editor_v{st.session_state.category_editor_version}",
                    column_config={
                        "Transaction ID": st.column_config.TextColumn(
                            "Transaction ID",
                            disabled=True,
                            width="small",
                        ),
                        "Date": st.column_config.TextColumn(
                            "Date",
                            disabled=True,
                        ),
                        "Merchant": st.column_config.TextColumn(
                            "Merchant",
                            disabled=True,
                        ),
                        "Amount": st.column_config.NumberColumn(
                            "Amount",
                            disabled=True,
                            format="%.2f",
                        ),
                        "Currency": st.column_config.TextColumn(
                            "Currency",
                            disabled=True,
                        ),
                        "Direction": st.column_config.TextColumn(
                            "Direction",
                            disabled=True,
                        ),
                        "Category": st.column_config.SelectboxColumn(
                            "Choose category ✏️",
                            options=category_labels,
                            required=False,
                        ),
                        "Review": st.column_config.CheckboxColumn(
                            "Review",
                            disabled=True,
                        ),
                    },
                    disabled=[
                        "Transaction ID",
                        "Date",
                        "Merchant",
                        "Amount",
                        "Currency",
                        "Direction",
                        "Review",
                    ],
                )

                apply_matching = st.checkbox(
                    "Apply each saved correction to the exact matching merchant",
                    value=True,
                    key="transaction_editor_apply_matching_v4",
                )

                save_changes = st.form_submit_button(
                    "💾 Save category corrections",
                    type="primary",
                    width="stretch",
                )

            if save_changes:
                changed_count = 0
                changed_indices = set()
                pending_campaigns = []

                # Only explicitly chosen categories are saved. Blank cells stay untouched.
                for row_position, (_, row) in enumerate(edited_df.iterrows()):
                    if row_position >= len(editor_indices):
                        continue

                    chosen_label = str(row.get("Category") or "").strip()
                    if not chosen_label:
                        continue

                    new_category = category_label_to_name.get(
                        chosen_label,
                        chosen_label,
                    ).strip()
                    if not new_category:
                        continue

                    source_index = editor_indices[row_position]
                    if source_index >= len(transaction_source):
                        continue
                    tx = transaction_source[source_index]
                    old_category = str(getattr(tx, "category", "") or "").strip()

                    if new_category.casefold() == old_category.casefold():
                        continue

                    try:
                        editor_engine.learn_from_user(
                            tx,
                            new_category,
                            "General",
                            apply_all=False,
                            transactions=transaction_source,
                        )
                    except Exception as exc:
                        print(f"Finora category correction warning: {exc}")
                        tx.category = new_category
                        tx.subcategory = "General"
                        tx.category_confidence = 1.0
                        tx.requires_review = False
                        tx.user_corrected = True

                    changed_count += 1
                    changed_indices.add(source_index)

                if apply_matching and changed_indices:
                    seen_family_categories = set()
                    for anchor_index in sorted(changed_indices):
                        anchor_tx = transaction_source[anchor_index]
                        new_category = str(
                            getattr(anchor_tx, "category", None)
                            or "Uncategorized"
                        )
                        family = _merchant_family(
                            getattr(anchor_tx, "merchant", None)
                            or getattr(anchor_tx, "description_raw", "")
                        )
                        if not family:
                            continue
                        campaign_key = (family, new_category)
                        if campaign_key in seen_family_categories:
                            continue
                        seen_family_categories.add(campaign_key)
                        matches = _campaign_candidates(
                            transaction_source,
                            anchor_index,
                            excluded_indices=changed_indices,
                        )
                        if matches:
                            pending_campaigns.append({
                                "anchor_index": anchor_index,
                                "category": new_category,
                                "match_indices": matches,
                            })

                st.session_state.transactions = transaction_source
                st.session_state.category_engine = editor_engine
                _save_transaction_backup(transaction_source)
                _save_statement_context()
                st.session_state.pending_category_campaigns = pending_campaigns

                if changed_count:
                    if pending_campaigns:
                        st.toast(
                            f"✅ {changed_count} correction(s) saved. Review the related merchant groups below.",
                            icon="✨",
                        )
                        st.rerun()

                    st.toast(
                        f"✅ {changed_count} category correction(s) saved.",
                        icon="✅",
                    )
                    _redirect_to_overview()
                    st.rerun()
                else:
                    st.info("No category selections were saved. Choose at least one category before saving.")
        else:
            st.success(
                "✅ No uncategorized or review-flagged transactions remain. "
                "Already-categorized transactions are intentionally hidden in review mode."
            )
            if st.button(
                "Open all transactions",
                key="open_all_transactions_v4",
                width="stretch",
            ):
                st.session_state.open_category_editor = False
                st.rerun()

    elif edit_categories and not transaction_source:
        st.info("Analyze a financial statement first.")


    # --------------------------------------------------------
    # MERCHANT CATEGORY CAMPAIGN CONFIRMATION
    # --------------------------------------------------------
    pending_campaigns = list(
        st.session_state.get("pending_category_campaigns") or []
    )

    if pending_campaigns:
        campaign = pending_campaigns[0]
        anchor_index = int(campaign.get("anchor_index", -1))
        campaign_category = str(
            campaign.get("category") or "Uncategorized"
        )
        match_indices = [
            int(idx)
            for idx in campaign.get("match_indices", [])
        ]

        if 0 <= anchor_index < len(transaction_source):
            anchor_tx = transaction_source[anchor_index]
            anchor_name = str(
                getattr(anchor_tx, "merchant", None)
                or getattr(anchor_tx, "description_raw", None)
                or "this merchant"
            ).strip()
            matched_names = _campaign_label(
                transaction_source,
                match_indices,
            )

            @st.dialog("✨ Related merchant transactions found")
            def _show_category_campaign():
                st.markdown(
                    f"### Apply **{campaign_category}** to the related transactions?"
                )
                st.write(
                    f"You changed **{anchor_name}** to **{campaign_category}**. "
                    f"Finora found **{len(match_indices)} other transaction(s)** "
                    "that look like the same merchant family."
                )

                if matched_names:
                    shown = matched_names[:8]
                    for name in shown:
                        st.markdown(f"• `{name}`")
                    if len(matched_names) > 8:
                        st.caption(
                            f"+ {len(matched_names) - 8} other matching merchant name(s)"
                        )

                st.info(
                    "Finora will only campaign this group after you confirm. "
                    "This is useful for names such as McDonald's store/reference variants, "
                    "while still letting you review groups such as different ADNOC descriptions."
                )

                c_apply, c_only = st.columns(2)

                with c_apply:
                    if st.button(
                        f"✅ Apply to all {len(match_indices)}",
                        type="primary",
                        width="stretch",
                        key="confirm_category_campaign_v3",
                    ):
                        engine = st.session_state.get("category_engine")
                        if engine is None:
                            engine = HybridCategoryEngine(CategoryMemory())
                            st.session_state.category_engine = engine

                        applied = _apply_campaign(
                            engine,
                            transaction_source,
                            match_indices,
                            campaign_category,
                        )
                        st.session_state.transactions = transaction_source
                        st.session_state.category_engine = engine
                        _save_transaction_backup(transaction_source)
                        _save_statement_context()
                        st.session_state.pending_category_campaigns = pending_campaigns[1:]

                        if st.session_state.pending_category_campaigns:
                            st.rerun()
                        st.toast(
                            f"✅ {applied} related transaction(s) updated to {campaign_category}.",
                            icon="✨",
                        )
                        _redirect_to_overview()
                        st.rerun()

                with c_only:
                    if st.button(
                        "Only this transaction",
                        width="stretch",
                        key="reject_category_campaign_v3",
                    ):
                        st.session_state.pending_category_campaigns = pending_campaigns[1:]
                        if st.session_state.pending_category_campaigns:
                            st.rerun()
                        st.toast(
                            "Kept the change only for the selected transaction.",
                            icon="ℹ️",
                        )
                        _redirect_to_overview()
                        st.rerun()

            _show_category_campaign()

    # --------------------------------------------------------
    # SEARCH / FILTERS
    # --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        search = st.text_input(
            "Search",
            placeholder="Merchant or description...",
            key="transaction_search_v2",
        )

    with c2:
        direction = st.selectbox(
            "Direction",
            ["All", "credit", "debit"],
            key="transaction_direction_v2",
        )

    with c3:
        status = st.selectbox(
            "Status",
            ["All", "Needs Review", "Validated"],
            key="transaction_status_v2",
        )

    with c4:
        category_values = sorted({
            str(value).strip()
            for value in transaction_df.get(
                "Category",
                pd.Series(dtype=str),
            ).dropna().tolist()
            if str(value).strip()
        })
        category_filter = st.selectbox(
            "Category",
            ["All"] + category_values,
            key="transaction_category_v2",
        )

    filtered = transaction_df.copy()

    focus = st.session_state.get("transaction_focus")
    if focus and focus.get("value") and not filtered.empty:
        focus_value = _safe_focus_value(focus.get("value"))
        focus_kind = str(focus.get("kind") or "").casefold()

        if focus_kind == "category" and "Category" in filtered.columns:
            filtered = filtered[
                filtered["Category"]
                .fillna("")
                .astype(str)
                .str.casefold()
                == focus_value.casefold()
            ]
        elif focus_kind == "merchant" and "Merchant" in filtered.columns:
            filtered = filtered[
                filtered["Merchant"]
                .fillna("")
                .astype(str)
                .str.casefold()
                == focus_value.casefold()
            ]

        render(
            f"<div class='focus-filter-banner'>"
            f"Showing transactions for <strong>{escape(focus_value)}</strong>. "
            f"Use the controls below to narrow the view further."
            f"</div>"
        )

    if search and not filtered.empty:
        merchant_match = (
            filtered["Merchant"]
            .fillna("")
            .astype(str)
            .str.contains(search, case=False, na=False)
        )
        description_match = (
            filtered["Description"]
            .fillna("")
            .astype(str)
            .str.contains(search, case=False, na=False)
        )
        filtered = filtered[merchant_match | description_match]

    if direction != "All" and not filtered.empty:
        filtered = filtered[
            filtered["Direction"].astype(str).str.lower()
            == direction.lower()
        ]

    if status == "Needs Review" and not filtered.empty:
        filtered = filtered[filtered["Review"] == True]
    elif status == "Validated" and not filtered.empty:
        filtered = filtered[filtered["Review"] == False]

    if category_filter != "All" and not filtered.empty:
        filtered = filtered[
            filtered["Category"].fillna("").astype(str)
            == category_filter
        ]

    st.caption(
        f"{len(filtered):,} of {len(transaction_source):,} transactions"
    )

    if not transaction_source:
        st.warning(
            "No analyzed transactions are currently stored in this session. "
            "Return to Overview and analyze the statement again."
        )
    elif filtered.empty:
        st.info(
            "No transactions match the selected filters. "
            "Set Direction, Status and Category to All and clear Search."
        )
    else:
        display_columns = [
            "Date",
            "Merchant",
            "Description",
            "Amount",
            "Currency",
            "Direction",
            "Type",
            "Category",
            "Confidence",
            "Review",
        ]
        display_columns = [
            column
            for column in display_columns
            if column in filtered.columns
        ]

        st.dataframe(
            filtered[display_columns],
            width="stretch",
            hide_index=True,
            height=600,
        )

    if focus and focus.get("value"):
        if st.button(
            "Clear focused transaction view",
            key="clear_transaction_focus_v2",
        ):
            st.session_state.transaction_focus = None
            st.query_params["page"] = "Transactions"
            st.rerun()

    st.download_button(
        "⬇️ Download Transactions CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "finora_transactions.csv",
        "text/csv",
        key="download_transactions_v2",
    )


# ============================================================
# AI INTELLIGENCE
# ============================================================

elif st.session_state.page == "AI":
    render_ai_assistant_page(transactions)


# FOOTER
# ============================================================

st.html("""
<div style="
    margin-top:60px;
    padding-top:20px;
    border-top:1px solid #182235;
    text-align:center;
    color:#334155;
    font-size:.66rem;
">
    FINORA AI · Turn financial statements into financial intelligence.
</div>
""")

st.html('\n<style>\n/* ============================================================\n   FINORA READABILITY PASS - SPENDING RANKING\n   ============================================================ */\n\n.rank-card {\n    padding: 24px 24px 18px !important;\n}\n\n.rank-row {\n    padding: 15px 0 17px !important;\n}\n\n.rank-top {\n    gap: 14px !important;\n}\n\n.rank-number {\n    width: 28px !important;\n    font-size: .72rem !important;\n}\n\n.rank-name {\n    font-size: 1.28rem !important;\n    line-height: 1.35 !important;\n    font-weight: 850 !important;\n    color: #f1f5f9 !important;\n}\n\n.rank-amount {\n    font-size: 1.28rem !important;\n    line-height: 1.35 !important;\n    font-weight: 850 !important;\n    color: #f8fafc !important;\n}\n\n.rank-track {\n    height: 6px !important;\n    margin: 10px 0 0 42px !important;\n}\n\n.rank-sub {\n    margin: 7px 0 0 42px !important;\n    color: #64748b !important;\n    font-size: .94rem !important;\n    line-height: 1.45 !important;\n}\n\n/* Covers the newer clickable category/merchant button implementation. */\ndiv[class*="st-key-category_spend_btn"] button {\n    min-height: 46px !important;\n    padding: 6px 10px !important;\n    font-size: 1.15rem !important;\n    line-height: 1.35 !important;\n    font-weight: 850 !important;\n}\n\ndiv[class*="st-key-category_spend_btn"] button p,\ndiv[class*="st-key-category_spend_btn"] button span {\n    font-size: 1.15rem !important;\n    line-height: 1.35 !important;\n    font-weight: 850 !important;\n}\n\n/* Also enlarge the explanatory line below each category. */\n.rank-card .rank-sub {\n    font-size: .94rem !important;\n}\n</style>\n')
