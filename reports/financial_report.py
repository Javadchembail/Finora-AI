from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from html import escape
from io import BytesIO
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PAGE_SIZE = landscape(A4)
PAGE_W, PAGE_H = PAGE_SIZE

# Existing Finora palette - preserved.
NAVY = colors.HexColor("#0B1730")
INDIGO = colors.HexColor("#4F46E5")
INDIGO_LIGHT = colors.HexColor("#EEF2FF")
TEXT = colors.HexColor("#182235")
MUTED = colors.HexColor("#667085")
GRID = colors.HexColor("#D9DFEA")
SOFT = colors.HexColor("#F7F9FC")
WHITE = colors.white
GREEN = colors.HexColor("#16834A")
RED = colors.HexColor("#C43A3A")
AMBER = colors.HexColor("#A85C00")


class FinoraBarChart(Flowable):
    """Small horizontal bar chart using the existing report palette."""

    def __init__(self, rows, width=240 * mm, bar_height=6, row_gap=7):
        super().__init__()
        self.rows = rows[:8]
        self.width = width
        self.bar_height = bar_height
        self.row_gap = row_gap
        self.height = max(1, len(self.rows)) * (bar_height + row_gap) + 8

    def wrap(self, availWidth, availHeight):
        self.width = min(self.width, availWidth)
        return self.width, self.height

    def draw(self):
        if not self.rows:
            return

        canvas = self.canv
        left_label = 58 * mm
        amount_width = 35 * mm
        track_width = max(40 * mm, self.width - left_label - amount_width)
        max_amount = max((float(amount) for _, amount in self.rows), default=0.0)

        y = self.height - self.bar_height - 3
        for category, amount in self.rows:
            label = str(category)
            if len(label) > 28:
                label = label[:27] + "…"

            canvas.setFillColor(TEXT)
            canvas.setFont("Helvetica-Bold", 7.6)
            canvas.drawString(0, y + 1, label)

            track_x = left_label
            canvas.setFillColor(colors.HexColor("#E7EBF2"))
            canvas.roundRect(
                track_x,
                y,
                track_width,
                self.bar_height,
                2.5,
                fill=1,
                stroke=0,
            )

            ratio = (float(amount) / max_amount) if max_amount else 0
            fill_width = track_width * max(0.0, min(1.0, ratio))
            canvas.setFillColor(INDIGO)
            canvas.roundRect(
                track_x,
                y,
                fill_width,
                self.bar_height,
                2.5,
                fill=1,
                stroke=0,
            )

            canvas.setFillColor(TEXT)
            canvas.setFont("Helvetica-Bold", 7.5)
            canvas.drawRightString(
                self.width,
                y + 1,
                f"{float(amount):,.2f}",
            )

            y -= self.bar_height + self.row_gap


class FinoraMiniBars(Flowable):
    """Compact weekly spending/payment bars for page 2."""

    def __init__(self, series, width=240 * mm, height=58 * mm):
        super().__init__()
        self.series = series
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        self.width = min(self.width, availWidth)
        return self.width, self.height

    def draw(self):
        if not self.series:
            return

        canvas = self.canv
        left = 12 * mm
        bottom = 11 * mm
        top = self.height - 7 * mm
        chart_w = self.width - left - 5 * mm
        chart_h = top - bottom

        max_value = max(
            [max(v[1], v[2], 0.0) for v in self.series] or [1.0]
        )
        max_value = max(max_value, 1.0)

        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        for fraction in (0, 0.5, 1):
            gy = bottom + chart_h * fraction
            canvas.line(left, gy, left + chart_w, gy)

        bar_slot = chart_w / max(1, len(self.series))
        bar_w = min(13 * mm, bar_slot * 0.24)

        for idx, (label, spent, payments) in enumerate(self.series):
            x = left + idx * bar_slot + bar_slot * 0.33
            spent_h = chart_h * (spent / max_value)
            pay_h = chart_h * (payments / max_value)

            canvas.setFillColor(RED)
            canvas.roundRect(x, bottom, bar_w, spent_h, 2, fill=1, stroke=0)

            canvas.setFillColor(GREEN)
            canvas.roundRect(
                x + bar_w + 2,
                bottom,
                bar_w,
                pay_h,
                2,
                fill=1,
                stroke=0,
            )

            canvas.setFillColor(MUTED)
            canvas.setFont("Helvetica", 6.2)
            short = str(label)
            if len(short) > 8:
                short = short[:8]
            canvas.drawCentredString(
                x + bar_w,
                3.2 * mm,
                short,
            )

        canvas.setFillColor(RED)
        canvas.roundRect(2 * mm, self.height - 5 * mm, 7 * mm, 2.8 * mm, 1, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(11 * mm, self.height - 5.6 * mm, "Spending")

        canvas.setFillColor(GREEN)
        canvas.roundRect(36 * mm, self.height - 5 * mm, 7 * mm, 2.8 * mm, 1, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.drawString(45 * mm, self.height - 5.6 * mm, "Payments / credits")


class FinoraReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=PAGE_SIZE, **kwargs)
        frame = Frame(
            14 * mm,
            16 * mm,
            PAGE_W - 28 * mm,
            PAGE_H - 31 * mm,
            id="normal",
        )
        self.addPageTemplates([
            PageTemplate(
                id="Finora",
                frames=frame,
                onPage=_draw_page_chrome,
            )
        ])


def _draw_page_chrome(canvas, doc):
    canvas.saveState()

    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.6)
    canvas.line(
        14 * mm,
        PAGE_H - 13 * mm,
        PAGE_W - 14 * mm,
        PAGE_H - 13 * mm,
    )

    x = 15 * mm
    y = PAGE_H - 9.6 * mm
    canvas.setFillColor(INDIGO)
    canvas.roundRect(
        x,
        y - 7 * mm,
        9 * mm,
        7 * mm,
        1.8 * mm,
        fill=1,
        stroke=0,
    )
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(x + 4.5 * mm, y - 5.1 * mm, "F")

    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(x + 12 * mm, y - 4.7 * mm, "Finora AI")

    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.2)
    canvas.drawRightString(
        PAGE_W - 14 * mm,
        y - 4.7 * mm,
        "Financial Intelligence Report",
    )

    canvas.setStrokeColor(GRID)
    canvas.line(14 * mm, 12 * mm, PAGE_W - 14 * mm, 12 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 6.6)
    canvas.drawString(
        14 * mm,
        7.8 * mm,
        "FINORA AI - Turn financial statements into financial intelligence.",
    )
    canvas.drawRightString(
        PAGE_W - 14 * mm,
        7.8 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def _n(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def _enum(value) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _money(value, currency: str) -> str:
    return f"{currency} {_n(value):,.2f}"


def _colored_money(value, currency: str, color) -> str:
    """Return a ReportLab-safe colored monetary value."""
    return f'<font color="{color.hexval()}">{escape(_money(value, currency))}</font>'


def _colored_text(value, color) -> str:
    return f'<font color="{color.hexval()}">{escape(str(value))}</font>'


def _safe_text(value: object, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _cat_name(tx) -> str:
    return _safe_text(getattr(tx, "category", None), "Uncategorized")


def _merchant_name(tx) -> str:
    return _safe_text(
        getattr(tx, "merchant", None)
        or getattr(tx, "description_normalized", None)
        or getattr(tx, "description_raw", None),
        "Unknown merchant",
    )


def _sort_transactions(transactions: Iterable) -> list:
    return sorted(
        list(transactions or []),
        key=lambda tx: (
            getattr(tx, "transaction_date", None) is None,
            getattr(tx, "transaction_date", None) or "",
            _merchant_name(tx).casefold(),
        ),
    )


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "FinoraTitleV2",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "subtitle": ParagraphStyle(
            "FinoraSubtitleV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=11.2,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "FinoraSectionV2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=TEXT,
            spaceBefore=7,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "FinoraBodyV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.1,
            leading=11.0,
            textColor=TEXT,
        ),
        "small": ParagraphStyle(
            "FinoraSmallV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=9.0,
            textColor=MUTED,
        ),
        "small_dark": ParagraphStyle(
            "FinoraSmallDarkV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=9.0,
            textColor=TEXT,
        ),
        "metric_label": ParagraphStyle(
            "FinoraMetricLabelV2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.3,
            leading=7.7,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "metric_value": ParagraphStyle(
            "FinoraMetricValueV2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11.0,
            leading=12.7,
            textColor=TEXT,
            alignment=TA_LEFT,
        ),
        "metric_value_small": ParagraphStyle(
            "FinoraMetricValueSmallV2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.6,
            leading=11.2,
            textColor=TEXT,
            alignment=TA_LEFT,
        ),
        "table_head": ParagraphStyle(
            "FinoraTableHeadV2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.4,
            leading=7.5,
            textColor=WHITE,
        ),
        "table_cell": ParagraphStyle(
            "FinoraTableCellV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=6.55,
            leading=8.0,
            textColor=TEXT,
        ),
        "table_cell_bold": ParagraphStyle(
            "FinoraTableCellBoldV2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.55,
            leading=8.0,
            textColor=TEXT,
        ),
        "table_cell_right": ParagraphStyle(
            "FinoraTableCellRightV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=6.55,
            leading=8.0,
            textColor=TEXT,
            alignment=TA_RIGHT,
        ),
        "insight": ParagraphStyle(
            "FinoraInsightV2",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.05,
            leading=11.0,
            textColor=TEXT,
        ),
    }


def _box_table(content, widths, padding=7, background=SOFT):
    table = Table(content, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
    ]))
    return table


def _is_credit_card(metadata: dict | None, txs: list) -> bool:
    statement_type = str((metadata or {}).get("statement_type") or "").lower()
    if statement_type == "credit_card":
        return True
    return any(
        _enum(getattr(tx, "statement_type", None)).lower() == "credit_card"
        for tx in txs
    )


def _metadata_number(metadata: dict, key: str):
    value = metadata.get(key)
    return None if value is None else _n(value)


def _weekly_series(txs: list):
    buckets = defaultdict(lambda: {"spent": 0.0, "payments": 0.0})
    for tx in txs:
        d = getattr(tx, "transaction_date", None)
        if d is None:
            continue
        try:
            week_start = d.fromordinal(d.toordinal() - d.weekday())
        except Exception:
            continue
        amount = abs(_n(getattr(tx, "original_amount", 0)))
        direction = _enum(getattr(tx, "direction", None)).lower()
        if direction == "credit":
            buckets[week_start]["payments"] += amount
        else:
            buckets[week_start]["spent"] += amount

    rows = []
    for week, values in sorted(buckets.items()):
        rows.append((week.strftime("%d %b"), values["spent"], values["payments"]))
    return rows[-8:]


def _largest_transactions(txs: list):
    return sorted(
        txs,
        key=lambda tx: abs(_n(getattr(tx, "original_amount", 0))),
        reverse=True,
    )[:8]


def _repeated_merchants(txs: list):
    counts = Counter(
        _merchant_name(tx).strip()
        for tx in txs
        if _enum(getattr(tx, "direction", None)).lower() == "debit"
    )
    totals = defaultdict(float)
    for tx in txs:
        if _enum(getattr(tx, "direction", None)).lower() == "debit":
            totals[_merchant_name(tx).strip()] += abs(_n(getattr(tx, "original_amount", 0)))
    return [
        (name, counts[name], totals[name])
        for name in counts
        if counts[name] >= 2
    ][:8]


def _build_report(
    transactions,
    file_name: str | None = None,
    statement_metadata: dict | None = None,
):
    txs = _sort_transactions(transactions)
    metadata = dict(statement_metadata or {})
    credit_card = _is_credit_card(metadata, txs)
    styles = _styles()

    currencies = [
        str(getattr(tx, "original_currency", "") or "").upper()
        for tx in txs
        if getattr(tx, "original_currency", None)
    ]
    currency = max(set(currencies), key=currencies.count) if currencies else "UNKNOWN"

    income = sum(
        abs(_n(getattr(tx, "original_amount", 0)))
        for tx in txs
        if _enum(getattr(tx, "direction", None)).lower() == "credit"
    )
    expenses = sum(
        abs(_n(getattr(tx, "original_amount", 0)))
        for tx in txs
        if _enum(getattr(tx, "direction", None)).lower() != "credit"
    )
    net = income - expenses
    review_count = sum(bool(getattr(tx, "requires_review", False)) for tx in txs)

    outgoing_count = sum(
        1 for tx in txs
        if _enum(getattr(tx, "direction", None)).lower() != "credit"
    )
    meaningful_categories = 0
    category_totals = defaultdict(float)
    merchant_totals = defaultdict(float)

    for tx in txs:
        if _enum(getattr(tx, "direction", None)).lower() != "debit":
            continue
        amount = abs(_n(getattr(tx, "original_amount", 0)))
        category = _cat_name(tx)
        merchant = _merchant_name(tx)
        category_totals[category] += amount
        merchant_totals[merchant] += amount
        if category.casefold() not in {"uncategorized", "unknown", "other", ""}:
            meaningful_categories += 1

    category_rows = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    merchant_rows = sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)
    category_coverage = (meaningful_categories / outgoing_count * 100) if outgoing_count else 0.0

    # Prefer exact statement metadata for the credit-card headline metrics.
    card_limit = _metadata_number(metadata, "card_limit")
    current_balance = _metadata_number(metadata, "current_balance")
    available_limit = _metadata_number(metadata, "available_limit")
    minimum_due = _metadata_number(metadata, "minimum_payment_due")
    total_due = _metadata_number(metadata, "total_payment_due")
    charges = _metadata_number(metadata, "profit_other_charges")
    opening_balance = _metadata_number(metadata, "opening_balance")
    due_date = metadata.get("payment_due_date")

    if credit_card:
        if total_due is None and current_balance is not None:
            total_due = current_balance
        if available_limit is None and card_limit is not None and current_balance is not None:
            available_limit = card_limit - current_balance
        if current_balance is None and opening_balance is not None:
            current_balance = opening_balance + expenses - income
        if card_limit is not None and current_balance is not None and available_limit is None:
            available_limit = max(0.0, card_limit - current_balance)

    utilization = None
    if credit_card and card_limit and current_balance is not None:
        utilization = (current_balance / card_limit) * 100

    top_category = category_rows[0] if category_rows else ("Not available", 0.0)
    top_merchant = merchant_rows[0] if merchant_rows else ("Not available", 0.0)
    largest = _largest_transactions(txs)
    repeated = _repeated_merchants(txs)
    weekly = _weekly_series(txs)

    if credit_card:
        story = (
            f"Current card balance is <b>{_money(current_balance or 0, currency)}</b> "
            f"against a <b>{_money(card_limit or 0, currency)}</b> credit limit, "
            f"leaving <b>{_money(available_limit or 0, currency)}</b> available. "
            f"Total payment due is <b>{_money(total_due or 0, currency)}</b>."
        )
    else:
        story = (
            f"You received <b>{_money(income, currency)}</b> and spent "
            f"<b>{_money(expenses, currency)}</b>. "
            + (
                f"That leaves a positive net movement of <b>{_money(net, currency)}</b>."
                if net >= 0
                else f"Spending was higher than incoming money by <b>{_money(abs(net), currency)}</b>."
            )
        )

    buffer = BytesIO()
    doc = FinoraReportDocTemplate(
        buffer,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title="Finora AI Financial Intelligence Report",
        author="Finora AI",
    )

    story_flow = []
    report_name = "Credit card statement report" if credit_card else "Financial statement report"

    # ========================================================
    # PAGE 1 - EXECUTIVE SUMMARY
    # ========================================================
    story_flow.append(Spacer(1, 4 * mm))
    story_flow.append(Paragraph("Finora AI Financial Intelligence Report", styles["title"]))
    story_flow.append(Paragraph(
        f"{escape(report_name)} · {escape(_safe_text(file_name, 'Financial statement'))} · {escape(currency)} · {len(txs):,} transactions",
        styles["subtitle"],
    ))
    story_flow.append(Spacer(1, 3.5 * mm))

    if credit_card:
        metrics = [[
            Paragraph("CARD LIMIT", styles["metric_label"]),
            Paragraph("CURRENT BALANCE", styles["metric_label"]),
            Paragraph("AVAILABLE CREDIT", styles["metric_label"]),
            Paragraph("TOTAL PAYMENT DUE", styles["metric_label"]),
            Paragraph("MINIMUM PAYMENT DUE", styles["metric_label"]),
        ], [
            Paragraph(_colored_money(card_limit or 0, currency, INDIGO), styles["metric_value"]),
            Paragraph(_colored_money(current_balance or 0, currency, RED), styles["metric_value"]),
            Paragraph(_colored_money(available_limit or 0, currency, GREEN), styles["metric_value"]),
            Paragraph(_colored_money(total_due or 0, currency, INDIGO), styles["metric_value"]),
            Paragraph(_colored_money(minimum_due or 0, currency, AMBER), styles["metric_value"]),
        ]]
        table = _box_table(metrics, [48 * mm] * 5, padding=5.5, background=colors.HexColor("#FAFBFD"))
        story_flow.append(table)
        story_flow.append(Spacer(1, 2.5 * mm))

        facts = [[
            Paragraph("PAYMENT DUE DATE", styles["metric_label"]),
            Paragraph("CREDIT UTILIZATION", styles["metric_label"]),
            Paragraph("PROFIT / OTHER CHARGES", styles["metric_label"]),
            Paragraph("TRANSACTIONS", styles["metric_label"]),
            Paragraph("REVIEW ITEMS", styles["metric_label"]),
        ], [
            Paragraph(escape(_safe_text(due_date, "Not stated")), styles["metric_value_small"]),
            Paragraph(f"{utilization:.1f}%" if utilization is not None else "Not available", styles["metric_value_small"]),
            Paragraph(_money(charges or 0, currency), styles["metric_value_small"]),
            Paragraph(f"{len(txs):,}", styles["metric_value_small"]),
            Paragraph(f"{review_count:,}", styles["metric_value_small"]),
        ]]
        story_flow.append(_box_table(facts, [48 * mm] * 5, padding=5, background=INDIGO_LIGHT))
        story_flow.append(Spacer(1, 3.5 * mm))

        if utilization is not None:
            bar_data = [[
                Paragraph("CREDIT UTILIZATION", styles["metric_label"]),
                Paragraph(f"{utilization:.1f}% of limit used", styles["metric_value_small"]),
            ]]
            util_table = Table(bar_data, colWidths=[50 * mm, 40 * mm,], hAlign="LEFT")
            util_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story_flow.append(util_table)
            # Visual utilization bar using a tiny one-row table.
            used = max(0.0, min(100.0, utilization))
            remaining = 100.0 - used
            util_bar = Table([["", ""]], colWidths=[240 * mm * used / 100, 240 * mm * remaining / 100], rowHeights=[4.5 * mm])
            util_bar.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), INDIGO),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E8EBF2")),
                ("BOX", (0, 0), (-1, -1), 0, WHITE),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            story_flow.append(util_bar)
            story_flow.append(Spacer(1, 3.2 * mm))
    else:
        metric_data = [[
            Paragraph("RECEIVED", styles["metric_label"]),
            Paragraph("SPENT", styles["metric_label"]),
            Paragraph("NET MOVEMENT", styles["metric_label"]),
            Paragraph("ACTIVITY", styles["metric_label"]),
            Paragraph("NEEDS REVIEW", styles["metric_label"]),
        ], [
            Paragraph(_colored_money(income, currency, GREEN), styles["metric_value"]),
            Paragraph(_colored_money(expenses, currency, RED), styles["metric_value"]),
            Paragraph(_colored_money(net, currency, INDIGO), styles["metric_value"]),
            Paragraph(f"{len(txs):,}", styles["metric_value"]),
            Paragraph(_colored_text(f"{review_count:,}", AMBER), styles["metric_value"]),
        ]]
        metrics = _box_table(metric_data, [48 * mm] * 5, padding=5.5, background=colors.HexColor("#FAFBFD"))
        story_flow.append(metrics)
        story_flow.append(Spacer(1, 3.5 * mm))

    story_flow.append(Paragraph("Statement facts", styles["section"]))
    fact_left = [
        [Paragraph("STATEMENT TYPE", styles["metric_label"]), Paragraph("CREDIT CARD" if credit_card else "BANK / FINANCIAL STATEMENT", styles["metric_value_small"])],
        [Paragraph("CURRENCY", styles["metric_label"]), Paragraph(escape(currency), styles["metric_value_small"])],
        [Paragraph("TRANSACTIONS ANALYZED", styles["metric_label"]), Paragraph(f"{len(txs):,}", styles["metric_value_small"])],
    ]
    if credit_card:
        fact_right = [
            [Paragraph("OPENING BALANCE", styles["metric_label"]), Paragraph(_money(opening_balance, currency) if opening_balance is not None else "Not stated", styles["metric_value_small"])],
            [Paragraph("CURRENT BALANCE", styles["metric_label"]), Paragraph(_money(current_balance, currency) if current_balance is not None else "Not stated", styles["metric_value_small"])],
            [Paragraph("DUE DATE", styles["metric_label"]), Paragraph(escape(_safe_text(due_date, "Not stated")), styles["metric_value_small"])],
        ]
    else:
        fact_right = [
            [Paragraph("OPENING BALANCE", styles["metric_label"]), Paragraph(_money(metadata.get("opening_balance"), currency) if metadata.get("opening_balance") is not None else "Not stated", styles["metric_value_small"])],
            [Paragraph("CLOSING BALANCE", styles["metric_label"]), Paragraph(_money(metadata.get("closing_balance"), currency) if metadata.get("closing_balance") is not None else "Not stated", styles["metric_value_small"])],
            [Paragraph("STATEMENT PERIOD", styles["metric_label"]), Paragraph(escape(_safe_text(metadata.get("statement_period") or (str(metadata.get("statement_start_date", "")) + " - " + str(metadata.get("statement_end_date", ""))), "Not stated")), styles["metric_value_small"])],
        ]

    fact_table = Table([
        [_box_table(fact_left, [48 * mm, 65 * mm], padding=5, background=SOFT), _box_table(fact_right, [48 * mm, 65 * mm], padding=5, background=SOFT)],
    ], colWidths=[120 * mm, 120 * mm], hAlign="LEFT")
    fact_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story_flow.append(fact_table)
    story_flow.append(Spacer(1, 2.8 * mm))

    story_flow.append(Paragraph("Finora's executive read", styles["section"]))
    story_flow.append(_box_table(
        [[Paragraph(story, styles["insight"])]],
        [240 * mm],
        padding=8,
        background=INDIGO_LIGHT,
    ))

    # Keep page 1 focused on the account/card snapshot and executive read.
    # The category visualization starts on page 2 so its heading and bars never split.
    story_flow.append(PageBreak())

    # ========================================================
    # PAGE 2 - SPENDING + MERCHANT INTELLIGENCE
    # ========================================================
    story_flow.append(Spacer(1, 4 * mm))
    story_flow.append(Paragraph("Spending and merchant intelligence", styles["title"]))
    story_flow.append(Paragraph(
        "Category concentration and merchant activity derived from the finalized transaction set.",
        styles["subtitle"],
    ))
    story_flow.append(Spacer(1, 2 * mm))

    story_flow.append(Paragraph("Spending by category", styles["section"]))
    if category_rows:
        story_flow.append(FinoraBarChart(category_rows))
    else:
        story_flow.append(Paragraph("No outgoing transactions were available for category analysis.", styles["small"]))

    story_flow.append(Paragraph("Top outgoing merchants", styles["section"]))

    merchant_head = [
        Paragraph("MERCHANT", styles["table_head"]),
        Paragraph("SPENDING", styles["table_head"]),
        Paragraph("SHARE OF OUTGOING", styles["table_head"]),
    ]
    merchant_data = [merchant_head]
    for merchant, amount in merchant_rows[:10]:
        share = amount / expenses * 100 if expenses else 0
        merchant_data.append([
            Paragraph(escape(merchant), styles["table_cell"]),
            Paragraph(_money(amount, currency), styles["table_cell_right"]),
            Paragraph(f"{share:.1f}%", styles["table_cell_right"]),
        ])
    merchant_table = Table(merchant_data, colWidths=[128 * mm, 58 * mm, 54 * mm], repeatRows=1)
    merchant_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.55, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story_flow.append(merchant_table)
    story_flow.append(Spacer(1, 3 * mm))

    top_metrics = [[
        Paragraph("TOP CATEGORY", styles["metric_label"]),
        Paragraph("TOP MERCHANT", styles["metric_label"]),
        Paragraph("CATEGORY COVERAGE", styles["metric_label"]),
        Paragraph("REPEATED MERCHANTS", styles["metric_label"]),
    ], [
        Paragraph(escape(top_category[0]), styles["metric_value_small"]),
        Paragraph(escape(top_merchant[0]), styles["metric_value_small"]),
        Paragraph(f"{category_coverage:.0f}%", styles["metric_value_small"]),
        Paragraph(f"{len(repeated):,}", styles["metric_value_small"]),
    ], [
        Paragraph(_money(top_category[1], currency), styles["small"]),
        Paragraph(_money(top_merchant[1], currency), styles["small"]),
        Paragraph(f"{meaningful_categories:,} meaningful outgoing transactions", styles["small"]),
        Paragraph("Merchant families appearing 2+ times", styles["small"]),
    ]]
    story_flow.append(_box_table(top_metrics, [60 * mm] * 4, padding=6, background=SOFT))
    story_flow.append(Spacer(1, 3.5 * mm))

    # Keep the trend chart together on its own page so its heading never
    # gets stranded at the bottom of the previous page.

    story_flow.append(PageBreak())
    story_flow.append(Spacer(1, 4 * mm))
    story_flow.append(Paragraph("Spending trend and repeated activity", styles["title"]))
    story_flow.append(Paragraph(
        "Weekly spending and payment activity, followed by merchants that appear repeatedly in the statement.",
        styles["subtitle"],
    ))
    story_flow.append(Spacer(1, 2 * mm))

    story_flow.append(Paragraph("Spending trend", styles["section"]))
    if weekly:
        story_flow.append(FinoraMiniBars(weekly, height=45 * mm))
    else:
        story_flow.append(Paragraph("Not enough dated transactions for a trend view.", styles["small"]))

    story_flow.append(Paragraph("Repeated merchants", styles["section"]))
    if repeated:
        repeat_data = [[
            Paragraph("MERCHANT", styles["table_head"]),
            Paragraph("TRANSACTIONS", styles["table_head"]),
            Paragraph("TOTAL", styles["table_head"]),
        ]]
        for merchant, count, amount in repeated[:8]:
            repeat_data.append([
                Paragraph(escape(merchant), styles["table_cell"]),
                Paragraph(str(count), styles["table_cell_right"]),
                Paragraph(_money(amount, currency), styles["table_cell_right"]),
            ])
        repeat_table = Table(repeat_data, colWidths=[160 * mm, 35 * mm, 45 * mm], repeatRows=1)
        repeat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
            ("BOX", (0, 0), (-1, -1), 0.5, GRID),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story_flow.append(repeat_table)
    else:
        story_flow.append(Paragraph("No merchants appeared more than once in the finalized debit set.", styles["small"]))

    # ========================================================
    # PAGE 4 - RECONCILIATION + REVIEW + LARGEST TRANSACTIONS
    # ========================================================
    story_flow.append(PageBreak())
    story_flow.append(Spacer(1, 4 * mm))
    story_flow.append(Paragraph("Reconciliation and review", styles["title"]))
    story_flow.append(Paragraph(
        "Finora uses the supplied statement metadata where available and compares it with the extracted transaction flow.",
        styles["subtitle"],
    ))
    story_flow.append(Spacer(1, 2 * mm))

    if credit_card and opening_balance is not None and current_balance is not None:
        calc_balance = opening_balance + expenses - income
        difference = calc_balance - current_balance
        recon_title = "CREDIT-CARD RECONCILIATION"
        recon_text = (
            f"Opening balance {_money(opening_balance, currency)} + purchases {_money(expenses, currency)} "
            f"- payments/credits {_money(income, currency)} = calculated current balance {_money(calc_balance, currency)}."
        )
    else:
        closing_balance = _metadata_number(metadata, "closing_balance")
        opening_bank = _metadata_number(metadata, "opening_balance")
        calc_balance = (opening_bank + income - expenses) if opening_bank is not None else None
        difference = (calc_balance - closing_balance) if calc_balance is not None and closing_balance is not None else None
        recon_title = "STATEMENT RECONCILIATION"
        recon_text = (
            f"Opening balance {_money(opening_bank, currency)} + received {_money(income, currency)} "
            f"- spent {_money(expenses, currency)} = calculated closing balance {_money(calc_balance, currency)}."
            if calc_balance is not None
            else "No statement opening/closing balance pair was supplied for a full reconciliation."
        )

    if difference is not None:
        recon_status = "Reconciled" if abs(difference) < 0.01 else f"Difference {_money(abs(difference), currency)}"
        recon_color = GREEN if abs(difference) < 0.01 else AMBER
    else:
        recon_status = "Not available"
        recon_color = MUTED

    recon_box = Table([
        [
            Paragraph(recon_title, styles["metric_label"]),
            Paragraph(recon_status, styles["metric_value_small"]),
        ],
        [Paragraph(recon_text, styles["insight"]), ""],
    ], colWidths=[70 * mm, 170 * mm], hAlign="LEFT")
    recon_box.setStyle(TableStyle([
        ("SPAN", (0, 1), (1, 1)),
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
        ("TEXTCOLOR", (1, 0), (1, 0), recon_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story_flow.append(recon_box)
    story_flow.append(Spacer(1, 3.5 * mm))

    review_text = (
        f"{review_count:,} transaction(s) are currently flagged for review. "
        f"Category coverage is {category_coverage:.0f}% based on meaningful categories among {outgoing_count:,} outgoing transactions. "
        "Category assignments reflect the current finalized Finora transaction data."
    )
    story_flow.append(Paragraph("Review signals", styles["section"]))
    story_flow.append(_box_table(
        [[Paragraph(review_text, styles["insight"])]],
        [240 * mm],
        padding=8,
        background=INDIGO_LIGHT,
    ))

    story_flow.append(Paragraph("Largest transactions", styles["section"]))
    largest_data = [[
        Paragraph("DATE", styles["table_head"]),
        Paragraph("MERCHANT", styles["table_head"]),
        Paragraph("DIRECTION", styles["table_head"]),
        Paragraph("AMOUNT", styles["table_head"]),
        Paragraph("CATEGORY", styles["table_head"]),
    ]]
    for tx in largest:
        d = getattr(tx, "transaction_date", None)
        date_text = d.isoformat() if d else "-"
        direction = _enum(getattr(tx, "direction", None))
        largest_data.append([
            Paragraph(escape(date_text), styles["table_cell"]),
            Paragraph(escape(_merchant_name(tx)), styles["table_cell"]),
            Paragraph(escape(direction), styles["table_cell"]),
            Paragraph(_money(getattr(tx, "original_amount", 0), currency), styles["table_cell_right"]),
            Paragraph(escape(_cat_name(tx)), styles["table_cell"]),
        ])
    largest_table = Table(largest_data, colWidths=[30 * mm, 95 * mm, 30 * mm, 40 * mm, 45 * mm], repeatRows=1)
    largest_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story_flow.append(largest_table)

    # ========================================================
    # PAGE 5+ - COMPLETE LEDGER
    # ========================================================
    story_flow.append(PageBreak())
    story_flow.append(Spacer(1, 3 * mm))
    story_flow.append(Paragraph("Complete transaction ledger", styles["title"]))
    story_flow.append(Paragraph(
        "Every extracted transaction is included below. Categories and review status reflect the current Finora transaction data.",
        styles["subtitle"],
    ))
    story_flow.append(Spacer(1, 2.5 * mm))

    ledger_head = [
        Paragraph("DATE", styles["table_head"]),
        Paragraph("MERCHANT", styles["table_head"]),
        Paragraph("DESCRIPTION", styles["table_head"]),
        Paragraph("AMOUNT", styles["table_head"]),
        Paragraph("CURRENCY", styles["table_head"]),
        Paragraph("DIRECTION", styles["table_head"]),
        Paragraph("TYPE", styles["table_head"]),
        Paragraph("CATEGORY", styles["table_head"]),
        Paragraph("CONF.", styles["table_head"]),
        Paragraph("REVIEW", styles["table_head"]),
    ]
    ledger_rows = [ledger_head]
    for tx in txs:
        date_value = getattr(tx, "transaction_date", None)
        date_text = date_value.isoformat() if date_value else "-"
        confidence = _n(getattr(tx, "extraction_confidence", 0)) * 100
        review = "Yes" if getattr(tx, "requires_review", False) else "No"
        direction = _enum(getattr(tx, "direction", None))
        tx_type = _enum(getattr(tx, "transaction_type", None))
        ledger_rows.append([
            Paragraph(escape(date_text), styles["table_cell"]),
            Paragraph(escape(_merchant_name(tx)), styles["table_cell"]),
            Paragraph(escape(_safe_text(getattr(tx, "description_raw", None))), styles["table_cell"]),
            Paragraph(f"{_n(getattr(tx, 'original_amount', 0)):,.2f}", styles["table_cell_right"]),
            Paragraph(escape(_safe_text(getattr(tx, "original_currency", None), currency)), styles["table_cell"]),
            Paragraph(escape(direction), styles["table_cell"]),
            Paragraph(escape(tx_type), styles["table_cell"]),
            Paragraph(escape(_cat_name(tx)), styles["table_cell"]),
            Paragraph(f"{confidence:.1f}%", styles["table_cell_right"]),
            Paragraph(review, styles["table_cell"]),
        ])

    ledger = Table(
        ledger_rows,
        colWidths=[22 * mm, 34 * mm, 64 * mm, 23 * mm, 20 * mm, 20 * mm, 21 * mm, 31 * mm, 17 * mm, 13 * mm],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    ledger_style = [
        ("BACKGROUND", (0, 0), (-1, 0), INDIGO),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
    ]
    for row_index in range(1, len(ledger_rows)):
        if row_index % 2 == 0:
            ledger_style.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#FBFCFE"))
            )
    ledger.setStyle(TableStyle(ledger_style))
    story_flow.append(ledger)

    doc.build(story_flow)
    return buffer.getvalue()


def build_finora_report(
    transactions,
    file_name: str | None = None,
    statement_metadata: dict | None = None,
) -> bytes:
    """Return a polished Finora financial intelligence PDF in memory.

    Backward compatible with the original two-argument call. The optional
    statement_metadata argument enables richer bank/credit-card report details
    without requiring any app.py changes.
    """
    return _build_report(
        transactions,
        file_name=file_name,
        statement_metadata=statement_metadata,
    )
