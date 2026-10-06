from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from datetime import timedelta
from html import escape
from io import BytesIO
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PAGE_SIZE = A4
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

# Emoji font used by the category-spending visual. TwemojiMozilla is embedded
# when available so category pictograms survive PDF export.
try:
    _TWEMOJI_FONT_PATH = "/usr/share/texlive/texmf-dist/fonts/truetype/public/twemoji-colr/TwemojiMozilla.ttf"
    pdfmetrics.registerFont(TTFont("Twemoji", _TWEMOJI_FONT_PATH))
    TWEMOJI_FONT = "Twemoji"
except Exception:
    TWEMOJI_FONT = "Helvetica"


class FinoraBarChart(Flowable):
    """Small horizontal bar chart using the existing report palette."""

    def __init__(self, rows, width=240 * mm, bar_height=6, row_gap=7):
        super().__init__()
        self.rows = rows
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


class Concept3SpendingDonut(Flowable):
    """Premium spending-allocation donut used by the final Finora report."""
    def __init__(self, data, width=174 * mm, height=78 * mm):
        super().__init__()
        self.data = data
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        if not self.data:
            return
        c = self.canv
        cx, cy = 48 * mm, self.height / 2
        r, inner = 25 * mm, 15.5 * mm
        palette = [
            colors.HexColor("#7C5CFC"), colors.HexColor("#4F7CFF"),
            colors.HexColor("#19B5A5"), colors.HexColor("#F59E0B"),
            colors.HexColor("#EF5B5B"), colors.HexColor("#8B5CF6"),
            colors.HexColor("#14B8A6"), colors.HexColor("#64748B"),
            colors.HexColor("#F97316"), colors.HexColor("#94A3B8"),
        ]
        total = sum(v for _, v, _ in self.data) or 1.0
        start = 90
        for i, (_, value, _) in enumerate(self.data):
            extent = 360 * value / total
            c.setFillColor(palette[i % len(palette)])
            c.wedge(cx-r, cy-r, cx+r, cy+r, start-extent, start, fill=1, stroke=0)
            start -= extent
        c.setFillColor(colors.HexColor("#F5F6FA"))
        c.circle(cx, cy, inner, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#0B1220"))
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(cx, cy + 2.2 * mm, "SPENDING")
        c.setFillColor(colors.HexColor("#667085"))
        c.setFont("Helvetica", 6.2)
        c.drawCentredString(cx, cy - 3.7 * mm, "TOTAL SPENDING")

        lx, ly = 88 * mm, self.height - 5 * mm
        for i, (label, value, pct) in enumerate(self.data):
            col = palette[i % len(palette)]
            c.setFillColor(col)
            c.roundRect(lx, ly - 1.2 * mm, 3.1 * mm, 3.1 * mm, 0.7, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#172033"))
            c.setFont("Helvetica-Bold", 6.7)
            c.drawString(lx + 5 * mm, ly, str(label)[:28])
            c.setFillColor(colors.HexColor("#667085"))
            c.setFont("Helvetica", 6.5)
            c.drawRightString(self.width, ly, f"{pct:.1f}% · {value:,.2f}")
            ly -= 6.6 * mm


class Concept3BehaviourDiagram(Flowable):
    """Value-vs-frequency map: a genuinely different view of category behaviour."""
    def __init__(self, data, width=174 * mm, height=88 * mm):
        super().__init__()
        self.data = data
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        if not self.data:
            return
        c = self.canv
        x0, y0, w, h = 22 * mm, 17 * mm, 145 * mm, 55 * mm
        c.setStrokeColor(colors.HexColor("#E3E7EF"))
        c.setLineWidth(0.8)
        c.line(x0, y0, x0+w, y0)
        c.line(x0, y0, x0, y0+h)
        c.setFillColor(colors.HexColor("#667085"))
        c.setFont("Helvetica", 6.5)
        c.drawString(x0+w-40*mm, y0-6*mm, "TRANSACTION FREQUENCY →")
        c.saveState()
        c.translate(x0-9*mm, y0+24*mm)
        c.rotate(90)
        c.drawString(0, 0, "SPEND VALUE →")
        c.restoreState()

        max_value = max(v for _, v, _ in self.data) or 1.0
        max_count = max(n for _, _, n in self.data) or 1
        palette = [
            colors.HexColor("#7C5CFC"), colors.HexColor("#4F7CFF"),
            colors.HexColor("#19B5A5"), colors.HexColor("#F59E0B"),
            colors.HexColor("#EF5B5B"), colors.HexColor("#8B5CF6"),
            colors.HexColor("#14B8A6"), colors.HexColor("#64748B"),
            colors.HexColor("#F97316"), colors.HexColor("#94A3B8"),
        ]
        for i, (label, value, count) in enumerate(self.data):
            x = x0 + (count / max_count) * (w - 9*mm) + 4*mm
            y = y0 + (value / max_value) * (h - 7*mm) + 3*mm
            radius = 3.5*mm if value > max_value * 0.15 else 2.4*mm
            c.setFillColor(palette[i % len(palette)])
            c.circle(x, y, radius, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#172033"))
            c.setFont("Helvetica-Bold", 5.9)
            c.drawCentredString(x, min(y + radius + 1.7*mm, y0+h+3*mm), str(label)[:16])


class Concept3WeeklyChart(Flowable):
    """Weekly spending vs payments/credits chart."""
    def __init__(self, series, width=174 * mm, height=67 * mm):
        super().__init__()
        self.series = series
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        if not self.series:
            return
        c = self.canv
        left, bottom = 17*mm, 14*mm
        chart_w, chart_h = self.width-24*mm, 42*mm
        max_value = max(max((x[1] for x in self.series), default=0), max((x[2] for x in self.series), default=0), 1)
        for fraction in (0, .5, 1):
            y = bottom + chart_h*fraction
            c.setStrokeColor(colors.HexColor("#E3E7EF"))
            c.line(left, y, left+chart_w, y)
        slot = chart_w / max(1, len(self.series))
        bw = min(9*mm, slot*0.22)
        for i, (label, spent, payments) in enumerate(self.series):
            x = left + i*slot + slot*0.34
            c.setFillColor(colors.HexColor("#EF5B5B"))
            c.roundRect(x, bottom, bw, chart_h*spent/max_value, 1.5, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#22C55E"))
            c.roundRect(x+bw+1.5, bottom, bw, chart_h*payments/max_value, 1.5, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#667085"))
            c.setFont("Helvetica", 6)
            c.drawCentredString(x+bw, bottom-5.5*mm, str(label)[:8])
        c.setFillColor(colors.HexColor("#EF5B5B")); c.rect(2*mm, self.height-4*mm, 5*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#667085")); c.setFont("Helvetica", 6.5); c.drawString(9*mm, self.height-4.5*mm, "Spending")
        c.setFillColor(colors.HexColor("#22C55E")); c.rect(33*mm, self.height-4*mm, 5*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#667085")); c.drawString(40*mm, self.height-4.5*mm, "Payments / credits")


def _concept3_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("C3Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=colors.white),
        "heading": ParagraphStyle("C3Heading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=colors.HexColor("#0B1220")),
        "h2": ParagraphStyle("C3H2", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=colors.HexColor("#172033")),
        "body": ParagraphStyle("C3Body", parent=base["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#172033")),
        "small": ParagraphStyle("C3Small", parent=base["Normal"], fontName="Helvetica", fontSize=7.1, leading=9.2, textColor=colors.HexColor("#667085")),
        "metric_label": ParagraphStyle("C3MetricLabel", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=6.2, leading=7.5, textColor=colors.HexColor("#667085")),
        "metric": ParagraphStyle("C3Metric", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=15, leading=17, textColor=colors.HexColor("#0B1220")),
        "white": ParagraphStyle("C3White", parent=base["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#CBD5E1")),
        "table_head": ParagraphStyle("C3TableHead", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=6.1, leading=7.3, textColor=colors.white),
        "table": ParagraphStyle("C3Table", parent=base["Normal"], fontName="Helvetica", fontSize=6.2, leading=7.7, textColor=colors.HexColor("#172033")),
        "table_right": ParagraphStyle("C3TableRight", parent=base["Normal"], fontName="Helvetica", fontSize=6.2, leading=7.7, textColor=colors.HexColor("#172033"), alignment=TA_RIGHT),
    }


def _c3_card(title, value, note, accent, styles, width=53*mm):
    t = Table([
        [Paragraph(title.upper(), styles["metric_label"])],
        [Paragraph(value, styles["metric"])],
        [Paragraph(note, styles["small"])],
    ], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), .55, colors.HexColor("#E3E7EF")),
        ("LINEBEFORE", (0,0), (0,-1), 3, accent),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t


def _c3_box(content, widths, styles, background=None, padding=7):
    t = Table(content, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), background or colors.white),
        ("BOX", (0,0), (-1,-1), .55, colors.HexColor("#E3E7EF")),
        ("INNERGRID", (0,0), (-1,-1), .35, colors.HexColor("#E3E7EF")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), padding), ("RIGHTPADDING", (0,0), (-1,-1), padding),
        ("TOPPADDING", (0,0), (-1,-1), padding), ("BOTTOMPADDING", (0,0), (-1,-1), padding),
    ]))
    return t


def _c3_chrome(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFillColor(colors.HexColor("#F5F6FA"))
        canvas.rect(0,0,PAGE_W,PAGE_H,fill=1,stroke=0)
        canvas.setFillColor(colors.HexColor("#7C5CFC"))
        canvas.rect(0,PAGE_H-3*mm,PAGE_W,3*mm,fill=1,stroke=0)
        canvas.setFillColor(colors.HexColor("#0B1220")); canvas.setFont("Helvetica-Bold",7.5)
        canvas.drawString(18*mm,PAGE_H-11*mm,"FINORA AI")
        canvas.setFillColor(colors.HexColor("#667085")); canvas.setFont("Helvetica",7.5)
        canvas.drawRightString(PAGE_W-18*mm,PAGE_H-11*mm,"FINANCIAL INTELLIGENCE")
    canvas.setStrokeColor(colors.HexColor("#E3E7EF")); canvas.line(18*mm,12*mm,PAGE_W-18*mm,12*mm)
    canvas.setFillColor(colors.HexColor("#667085")); canvas.setFont("Helvetica",6.7)
    canvas.drawString(18*mm,7.5*mm,"CONFIDENTIAL · FINORA AI")
    canvas.drawRightString(PAGE_W-18*mm,7.5*mm,f"{doc.page:02d}")
    canvas.restoreState()



class Concept3MoneyFlowDashboard(Flowable):
    """Premium money-flow dashboard with aligned four-step statement cards."""
    def __init__(self, flow, health, formula, width=174*mm):
        super().__init__()
        self.flow = flow
        self.health = health
        self.formula = formula
        self.width = width
        self.height = 112*mm

    def wrap(self, availWidth, availHeight):
        self.width = min(self.width, availWidth)
        return self.width, self.height

    def draw(self):
        c = self.canv
        W = self.width

        # Four equal cards with fixed gutters.  The header is deliberately
        # two-line so long labels never collide with the icon or leave the
        # card boundary.
        gap = 5*mm
        card_w = (W - 3*gap) / 4
        card_h = 44*mm
        y = self.height - 48*mm
        radius = 4*mm

        accents = [
            colors.HexColor('#8B5CF6'),
            colors.HexColor('#4F7CFF'),
            colors.HexColor('#22B8A7'),
            colors.HexColor('#0B1220'),
        ]
        labels = [
            ('OPENING', 'BALANCE'),
            ('TOTAL', 'SPENDING'),
            ('PAYMENTS /', 'CREDITS'),
            ('CLOSING', 'BALANCE'),
        ]

        for i, (_, value, _) in enumerate(self.flow):
            x = i * (card_w + gap)
            dark = i == 3

            # Card.
            c.setFillColor(colors.HexColor('#0B1220') if dark else colors.white)
            c.setStrokeColor(colors.HexColor('#DDE3EF'))
            c.setLineWidth(0.8)
            c.roundRect(x, y, card_w, card_h, radius, fill=1, stroke=1)

            # Icon badge.
            icon_x = x + 10*mm
            icon_y = y + card_h - 11*mm
            c.setFillColor(accents[i])
            c.circle(icon_x, icon_y, 6*mm, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont('Helvetica-Bold', 9)
            c.drawCentredString(icon_x, icon_y - 3, '■')

            # Two-line label.  The second line is intentionally kept separate
            # so PAYMENTS / CREDITS and CLOSING BALANCE remain inside the card.
            label_color = colors.HexColor('#C4B5FD') if dark else accents[i]
            c.setFillColor(label_color)
            c.setFont('Helvetica-Bold', 6.2)
            c.drawString(x + 19*mm, y + card_h - 9*mm, labels[i][0])
            c.drawString(x + 19*mm, y + card_h - 13*mm, labels[i][1])

            # Amount and currency are aligned identically on every card.
            amount_color = colors.white if dark else colors.HexColor('#182235')
            muted_color = colors.HexColor('#CBD5E1') if dark else colors.HexColor('#667085')
            c.setFillColor(amount_color)
            c.setFont('Helvetica-Bold', 13)
            c.drawString(x + 8*mm, y + 17*mm, f'{float(value):,.2f}')
            c.setFillColor(muted_color)
            c.setFont('Helvetica', 6.3)
            c.drawString(x + 8*mm, y + 9*mm, 'AED')

            # Arrow sits exactly in the middle of each gutter, never inside a
            # card.  This removes the previous visual collision.
            if i < 3:
                arrow_x = x + card_w + gap/2
                c.setFillColor(colors.HexColor('#7C5CFC'))
                c.setFont('Helvetica-Bold', 13)
                c.drawCentredString(arrow_x, y + card_h/2 - 3, '→')

        # Statement health strip.
        hy = 16*mm
        hh = 22*mm
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.HexColor('#DDE3EF'))
        c.setLineWidth(0.7)
        c.roundRect(0, hy, W, hh, 3*mm, fill=1, stroke=1)

        health_labels = [
            ('Credit limit', self.health.get('limit')),
            ('Available', self.health.get('available')),
            ('Utilization', self.health.get('utilization')),
            ('Reconciled', self.health.get('reconciled')),
        ]
        seg = W / 4
        for i, (lab, val) in enumerate(health_labels):
            x = i * seg
            if i:
                c.setStrokeColor(colors.HexColor('#E5EAF2'))
                c.line(x, hy + 3*mm, x, hy + hh - 3*mm)
            c.setFillColor(colors.HexColor('#667085'))
            c.setFont('Helvetica', 6.4)
            c.drawString(x + 4*mm, hy + 13*mm, lab)
            c.setFillColor(colors.HexColor('#16A34A') if i == 3 else colors.HexColor('#182235'))
            c.setFont('Helvetica-Bold', 10)
            c.drawString(x + 4*mm, hy + 6*mm, str(val))

        c.setFillColor(colors.HexColor('#667085'))
        c.setFont('Helvetica', 6.1)
        c.drawString(0, 4.5*mm, self.formula)

class Concept3BehaviourDashboard(Flowable):
    """Premium dark category-spending card with emoji/category pictograms."""

    EMOJIS = {
        "uncategorized": "🗂️",
        "other": "📦",
        "food & dining": "🍽️",
        "food": "🍽️",
        "groceries": "🛒",
        "transportation": "🚗",
        "shopping": "🛍️",
        "bills & utilities": "💡",
        "healthcare": "💊",
        "saloon": "💇",
        "entertainment": "🎮",
        "education": "🎓",
        "housing": "🏠",
        "rent": "🏠",
        "travel": "✈️",
        "subscriptions": "📱",
        "insurance": "🛡️",
        "cash": "💵",
    }

    def __init__(self, data, spend, width=174 * mm):
        super().__init__()
        self.data = data
        self.spend = float(spend or 0)
        self.width = width
        self.height = 132 * mm

    def wrap(self, availWidth, availHeight):
        self.width = min(self.width, availWidth)
        return self.width, self.height

    @classmethod
    def emoji_for(cls, name):
        key = str(name).strip().lower()
        if key in cls.EMOJIS:
            return cls.EMOJIS[key]
        for token, emoji in cls.EMOJIS.items():
            if token in key:
                return emoji
        return "💳"

    def draw(self):
        c = self.canv
        W, H = self.width, self.height

        # Dark premium card, closely matching the supplied reference.
        c.setFillColor(colors.HexColor("#15171C"))
        c.setStrokeColor(colors.HexColor("#2B2E35"))
        c.setLineWidth(1.0)
        c.roundRect(0, 0, W, H, 7 * mm, fill=1, stroke=1)

        pad = 10 * mm
        title_y = H - 13 * mm

        # Chart emoji + title.
        c.setFont(TWEMOJI_FONT, 12)
        c.setFillColor(colors.white)
        c.drawString(pad, title_y, "📊")
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.HexColor("#F4F5F7"))
        c.drawString(pad + 10 * mm, title_y + 1, "Spending by category")

        rows = sorted(self.data, key=lambda x: float(x[1]), reverse=True)
        if not rows:
            return

        # Fit all categories cleanly without overlapping.
        max_amount = max(float(v) for _, v, _ in rows) or 1.0
        row_top = H - 27 * mm
        row_step = min(13.1 * mm, (H - 35 * mm) / max(len(rows), 1))
        label_x = pad + 11 * mm
        value_x = W - pad
        track_x = pad
        track_w = W - 2 * pad
        bar_h = 3.2 * mm

        for i, (name, amount, count) in enumerate(rows):
            amount = float(amount)
            cy = row_top - i * row_step

            # Emoji pictogram.
            c.setFont(TWEMOJI_FONT, 12)
            c.setFillColor(colors.white)
            c.drawString(track_x, cy - 2.5 * mm, self.emoji_for(name))

            # Category name.
            c.setFillColor(colors.HexColor("#F4F5F7"))
            c.setFont("Helvetica", 9.3)
            label = str(name)
            if len(label) > 25:
                label = label[:24] + "…"
            c.drawString(label_x, cy, label)

            # Amount.
            c.setFillColor(colors.HexColor("#AEB3BE"))
            c.setFont("Helvetica", 9.0)
            c.drawRightString(value_x, cy, f"{amount:,.2f}")

            # Track and fill.
            track_y = cy - 7.2 * mm
            c.setFillColor(colors.HexColor("#1D2026"))
            c.roundRect(track_x, track_y, track_w, bar_h, 1.6 * mm, fill=1, stroke=0)

            fill_w = track_w * max(0.0, min(1.0, amount / max_amount))
            c.setFillColor(colors.HexColor("#50D0D4"))
            c.roundRect(track_x, track_y, fill_w, bar_h, 1.6 * mm, fill=1, stroke=0)

        # Small explanatory footer.
        c.setFillColor(colors.HexColor("#7E8490"))
        c.setFont("Helvetica", 5.8)
        c.drawString(
            pad,
            5.5 * mm,
            "Ranked by total spend · transaction counts remain available in the report tables.",
        )


class Concept3SignalsDashboard(Flowable):
    """Four intelligence cards in a 2x2 grid plus a recommendation rail."""
    def __init__(self, signals, recommendations, width=174*mm):
        super().__init__(); self.signals=signals; self.recommendations=recommendations; self.width=width; self.height=92*mm
    def wrap(self, availWidth, availHeight): self.width=min(self.width,availWidth); return self.width,self.height
    def draw(self):
        c=self.canv; W=self.width; H=self.height; rail_w=58*mm; gap=5*mm; left_w=W-rail_w-gap; card_w=(left_w-4*mm)/2; card_h=37*mm
        for i,(num,title,body,accent,bg,metric) in enumerate(self.signals):
            col=i%2; row=i//2; x=col*(card_w+4*mm); y=H-((row+1)*card_h+row*4*mm)
            c.setFillColor(bg); c.setStrokeColor(colors.HexColor('#DDE3EF')); c.roundRect(x,y,card_w,card_h,3*mm,fill=1,stroke=1)
            c.setFillColor(accent); c.circle(x+8*mm,y+card_h-8*mm,4.5*mm,fill=1,stroke=0); c.setFillColor(colors.white); c.setFont('Helvetica-Bold',6); c.drawCentredString(x+8*mm,y+card_h-10*mm,num)
            c.setFillColor(accent); c.setFont('Helvetica-Bold',5.8); c.drawString(x+15*mm,y+card_h-6*mm,title[:24])
            c.setFillColor(colors.HexColor('#182235')); c.setFont('Helvetica',5.4); words=body.split(); line=''; yy=y+card_h-17*mm
            for w in words:
                test=(line+' '+w).strip()
                if len(test)>31: c.drawString(x+5*mm,yy,line); yy-=6.2; line=w
                else: line=test
            if line: c.drawString(x+5*mm,yy,line)
            if metric:
                c.setFillColor(accent); c.roundRect(x+5*mm,y+4*mm,card_w-10*mm,3.5*mm,1.5*mm,fill=1,stroke=0)
        rail_x=left_w+gap; c.setFillColor(colors.HexColor('#F8F9FC')); c.setStrokeColor(colors.HexColor('#DDE3EF')); c.roundRect(rail_x,0,rail_w,H-1*mm,3*mm,fill=1,stroke=1)
        c.setFillColor(colors.HexColor('#182235')); c.setFont('Helvetica-Bold',7); c.drawString(rail_x+5*mm,H-10*mm,'Recommended actions')
        yy=H-21*mm
        for i,rec in enumerate(self.recommendations[:3],1):
            c.setFillColor(colors.HexColor('#EEF0FF')); c.circle(rail_x+8*mm,yy,4*mm,fill=1,stroke=0); c.setFillColor(colors.HexColor('#7C5CFC')); c.setFont('Helvetica-Bold',6); c.drawCentredString(rail_x+8*mm,yy-2,str(i))
            c.setFillColor(colors.HexColor('#182235')); c.setFont('Helvetica',5.2); words=rec.split(); line=''; ry=yy+2
            for w in words:
                test=(line+' '+w).strip()
                if len(test)>28: c.drawString(rail_x+15*mm,ry,line); ry-=6; line=w
                else: line=test
            if line: c.drawString(rail_x+15*mm,ry,line)
            yy-=22*mm


def _build_report(transactions, file_name=None, statement_metadata=None):
    """Build the final Concept 3 Finora report from the live transaction set.

    The function intentionally does not deduplicate transactions. The parser's
    canonical transaction list is the single source of truth; repeated merchants
    remain when their underlying transactions differ.
    """
    txs = _sort_transactions(transactions)
    metadata = dict(statement_metadata or {})
    styles = _concept3_styles()

    currency_values = [str(getattr(t, "original_currency", "") or "").upper() for t in txs if getattr(t, "original_currency", None)]
    currency = Counter(currency_values).most_common(1)[0][0] if currency_values else "AED"

    debits = [t for t in txs if _enum(getattr(t, "direction", None)).lower() != "credit"]
    credits = [t for t in txs if _enum(getattr(t, "direction", None)).lower() == "credit"]
    spend = sum(abs(_n(getattr(t, "original_amount", 0))) for t in debits)
    credit_total = sum(abs(_n(getattr(t, "original_amount", 0))) for t in credits)

    categories = defaultdict(float); category_counts = Counter()
    merchants = defaultdict(float); merchant_counts = Counter()
    for t in debits:
        amount = abs(_n(getattr(t, "original_amount", 0)))
        categories[_cat_name(t)] += amount
        category_counts[_cat_name(t)] += 1
        merchants[_merchant_name(t)] += amount
        merchant_counts[_merchant_name(t)] += 1
    category_rows = sorted(categories.items(), key=lambda x:x[1], reverse=True)
    merchant_rows = sorted(merchants.items(), key=lambda x:x[1], reverse=True)
    category_data = [(name, amount, amount/spend*100 if spend else 0) for name, amount in category_rows]
    behaviour_data = [(name, amount, category_counts[name]) for name, amount in category_rows]

    dates = [getattr(t, "transaction_date", None) for t in txs if getattr(t, "transaction_date", None) is not None]
    period = f"{min(dates):%d %b %Y} – {max(dates):%d %b %Y}" if dates else "Statement period"

    statement_type = str(metadata.get("statement_type") or "").lower()
    credit_card = statement_type == "credit_card" or any(str(getattr(t, "statement_type", "")).lower().endswith("credit_card") for t in txs)
    card_limit = _n(metadata.get("card_limit")) if metadata.get("card_limit") is not None else None
    current_balance = _n(metadata.get("current_balance")) if metadata.get("current_balance") is not None else None
    available_credit = _n(metadata.get("available_limit")) if metadata.get("available_limit") is not None else None
    if credit_card and available_credit is None and card_limit is not None and current_balance is not None:
        available_credit = card_limit - current_balance
    utilization = (current_balance/card_limit*100) if credit_card and current_balance is not None and card_limit else None
    total_due = _n(metadata.get("total_payment_due")) if metadata.get("total_payment_due") is not None else current_balance
    minimum_due = _n(metadata.get("minimum_payment_due")) if metadata.get("minimum_payment_due") is not None else None
    opening = _n(metadata.get("opening_balance")) if metadata.get("opening_balance") is not None else None
    review_count = sum(bool(getattr(t, "requires_review", False)) for t in txs)

    # Weekly series.
    weekly = defaultdict(lambda:[0.0,0.0])
    for t in txs:
        d = getattr(t,"transaction_date",None)
        if d is None: continue
        week = d - timedelta(days=d.weekday())
        amount = abs(_n(getattr(t,"original_amount",0)))
        if _enum(getattr(t,"direction",None)).lower() == "credit": weekly[week][1] += amount
        else: weekly[week][0] += amount
    weekly_series = [(d.strftime("%d %b"),v[0],v[1]) for d,v in sorted(weekly.items())][-8:]

    def safe_money(v): return f"{currency} {v:,.2f}"
    top_category = category_rows[0] if category_rows else ("No category",0.0)
    top_merchant = merchant_rows[0] if merchant_rows else ("No merchant",0.0)
    repeated = [(m,merchant_counts[m],amount) for m,amount in merchant_rows if merchant_counts[m] >= 2][:8]
    category_coverage = (sum(category_counts[k] for k in categories if k.casefold() not in {"uncategorized","unknown",""}) / len(debits) * 100) if debits else 0

    if credit_card:
        executive = (
            f"Current balance is <b>{safe_money(current_balance or 0)}</b> against a <b>{safe_money(card_limit or 0)}</b> limit, "
            f"leaving <b>{safe_money(available_credit or 0)}</b> available. Total outgoing spending is <b>{safe_money(spend)}</b> "
            f"across {len(debits):,} transactions."
        )
    else:
        net = credit_total - spend
        executive = f"Received <b>{safe_money(credit_total)}</b> and spent <b>{safe_money(spend)}</b>. Net movement is <b>{safe_money(net)}</b>."

    story=[]
    # Cover
    cover = Table([
        [Paragraph("FINORA", ParagraphStyle("brand3",fontName="Helvetica-Bold",fontSize=12,textColor=colors.HexColor("#A78BFA")))],
        [Spacer(1,18*mm)],
        [Paragraph("Your money.<br/><font color='#A78BFA'>Explained.</font>",styles["title"])],
        [Spacer(1,5*mm)],
        [Paragraph("Financial Intelligence Report",ParagraphStyle("cover_sub",fontName="Helvetica",fontSize=13,textColor=colors.HexColor("#E2E8F0")))],
        [Spacer(1,8*mm)],
        [Paragraph(f"{escape(period)} · {escape(currency)} · {len(txs):,} transactions",styles["white"])],
        [Spacer(1,18*mm)],
        [Paragraph(f"<b>EXECUTIVE SIGNAL</b><br/><br/>{executive}",styles["white"])],
    ], colWidths=[174*mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#0B1220")),
        ("LEFTPADDING",(0,0),(-1,-1),15*mm),("RIGHTPADDING",(0,0),(-1,-1),15*mm),
        ("TOPPADDING",(0,0),(-1,-1),12*mm),("BOTTOMPADDING",(0,0),(-1,-1),12*mm),
    ]))
    story += [Spacer(1,7*mm),cover,Spacer(1,7*mm)]
    cards = Table([[
        _c3_card("Statement balance" if credit_card else "Total received", safe_money(current_balance if credit_card else credit_total), "Closing balance" if credit_card else "Credits / inflow", colors.HexColor("#7C5CFC"), styles),
        _c3_card("Total spending", safe_money(spend), f"{len(debits):,} outgoing", colors.HexColor("#4F7CFF"), styles),
        _c3_card("Utilization" if credit_card else "Net movement", f"{utilization:.1f}%" if utilization is not None else safe_money(credit_total-spend), safe_money(available_credit or 0)+" available" if credit_card else "Credits minus spending", colors.HexColor("#22C55E"), styles),
    ]], colWidths=[56*mm]*3)
    cards.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),4)]))
    story += [cards,PageBreak()]

    # Spending map
    story += [Spacer(1,4*mm),Paragraph("01 / SPENDING MAP",ParagraphStyle("kick1",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Where your money went",styles["heading"]),Paragraph("One visual map of the complete outgoing spend. Categories are not repeated elsewhere as another visual chart.",styles["small"]),Spacer(1,3*mm)]
    donut_data=[]
    for i,(name,amount,pct) in enumerate(category_data):
        donut_data.append((name,amount,pct))
    story += [Concept3SpendingDonut(donut_data),Spacer(1,2*mm)]
    pool_left=f"{escape(top_category[0])}: {safe_money(top_category[1])} ({(top_category[1]/spend*100 if spend else 0):.1f}%)"
    food_name="Food & Dining"; food_count=category_counts.get(food_name,0)
    pool_right=f"{food_name} has {food_count:,} transactions — the highest frequency category." if food_count else "Transaction frequency is not available for a category-level signal."
    call=_c3_box([[Paragraph("<b>BIGGEST VALUE POOL</b>",styles["h2"]),Paragraph("<b>FREQUENCY SIGNAL</b>",styles["h2"])],[Paragraph(pool_left,styles["body"]),Paragraph(pool_right,styles["body"])]],[83*mm,83*mm],styles,background=colors.white,padding=8)
    story += [call,PageBreak()]

    # Money flow — redesigned dashboard.
    if credit_card and opening is not None:
        flow=[("OPENING BALANCE",opening,colors.HexColor("#7C5CFC")),("TOTAL SPENDING",spend,colors.HexColor("#4F7CFF")),("PAYMENTS / CREDITS",credit_total,colors.HexColor("#19B5A5")),("CLOSING BALANCE",current_balance or (opening+spend-credit_total),colors.HexColor("#0B1220"))]
        calc=opening+spend-credit_total; diff=calc-(current_balance or calc)
    else:
        flow=[("OPENING BALANCE",opening or 0,colors.HexColor("#7C5CFC")),("RECEIVED",credit_total,colors.HexColor("#19B5A5")),("SPENT",spend,colors.HexColor("#EF5B5B")),("CLOSING BALANCE",_n(metadata.get("closing_balance")) if metadata.get("closing_balance") is not None else (opening+credit_total-spend if opening is not None else credit_total-spend),colors.HexColor("#0B1220"))]
        calc=(opening+credit_total-spend) if opening is not None else None; supplied=_n(metadata.get("closing_balance")) if metadata.get("closing_balance") is not None else None; diff=(calc-supplied) if calc is not None and supplied is not None else None
    health={"limit":safe_money(card_limit or 0),"available":safe_money(available_credit or 0),"utilization":f"{utilization:.1f}%" if utilization is not None else "N/A","reconciled":"YES" if diff is not None and abs(diff)<0.01 else "CHECK"}
    formula=(f"Formula: {safe_money(opening)}  +  {safe_money(spend)}  −  {safe_money(credit_total)}  =  {safe_money(current_balance or calc)}" if credit_card and opening is not None else "Statement-level reconciliation shown from the supplied metadata.")
    story += [Spacer(1,4*mm),Paragraph("02 / MONEY FLOW",ParagraphStyle("kick2",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Follow the statement",styles["heading"]),Paragraph("A visual reconciliation of the statement from opening position to closing balance.",styles["small"]),Spacer(1,4*mm),Concept3MoneyFlowDashboard(flow,health,formula),PageBreak()]

    # Category intelligence — page 3 after removing the redundant behaviour chart.
    # Category intelligence
    story += [Spacer(1,4*mm),Paragraph("03 / CATEGORY INTELLIGENCE",ParagraphStyle("kick3",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Category performance",styles["heading"]),Paragraph("Spend, transaction count, average transaction and share — one canonical category table.",styles["small"]),Spacer(1,3*mm)]
    cr=[[Paragraph(x,styles["table_head"]) for x in ["CATEGORY","SPEND","TXNS","AVG / TXN","SHARE"]]]
    for name,amount in category_rows:
        count=category_counts[name]; cr.append([Paragraph(escape(name),styles["table"]),Paragraph(safe_money(amount),styles["table_right"]),Paragraph(str(count),styles["table_right"]),Paragraph(safe_money(amount/count if count else 0),styles["table_right"]),Paragraph(f"{amount/spend*100:.1f}%" if spend else "0.0%",styles["table_right"])])
    ct=Table(cr,colWidths=[49*mm,34*mm,20*mm,37*mm,20*mm],repeatRows=1)
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0B1220")),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.3,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("ALIGN",(1,1),(-1,-1),"RIGHT"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [ct,PageBreak()]

    # Merchant intelligence
    story += [Spacer(1,4*mm),Paragraph("04 / MERCHANT INTELLIGENCE",ParagraphStyle("kick4",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Merchant concentration",styles["heading"]),Paragraph("Top spenders and repeat activity are highlighted; transaction evidence remains only in the ledger.",styles["small"]),Spacer(1,3*mm)]
    mr=[[Paragraph(x,styles["table_head"]) for x in ["MERCHANT","SPEND","TXNS","SHARE"]]]
    for name,amount in merchant_rows[:12]: mr.append([Paragraph(escape(name),styles["table"]),Paragraph(safe_money(amount),styles["table_right"]),Paragraph(str(merchant_counts[name]),styles["table_right"]),Paragraph(f"{amount/spend*100:.1f}%" if spend else "0.0%",styles["table_right"])])
    mt=Table(mr,colWidths=[96*mm,34*mm,20*mm,22*mm],repeatRows=1)
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0B1220")),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.3,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("ALIGN",(1,1),(-1,-1),"RIGHT"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [mt,Spacer(1,7*mm),Paragraph("Most frequent merchants",styles["h2"])]
    fr=Table([[Paragraph(escape(name),styles["body"]),Paragraph(f"{merchant_counts[name]:,} transactions",styles["body"]),Paragraph(safe_money(amount),ParagraphStyle("mright",parent=styles["body"],alignment=TA_RIGHT))] for name,amount in merchant_rows if merchant_counts[name]>=2][:8],colWidths=[98*mm,40*mm,36*mm])
    fr.setStyle(TableStyle([("BOX",(0,0),(-1,-1),.5,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.3,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    story += [fr,PageBreak()]

    # Trend
    story += [Spacer(1,4*mm),Paragraph("05 / SPENDING TREND",ParagraphStyle("kick5",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Weekly movement",styles["heading"]),Paragraph("Outgoing spending and credits/payments are shown together to reveal the statement rhythm.",styles["small"]),Spacer(1,3*mm),Concept3WeeklyChart(weekly_series),Spacer(1,2*mm)]
    wr=[[Paragraph(x,styles["table_head"]) for x in ["WEEK","SPENDING","PAYMENTS / CREDITS"]]]
    for label,spent,payments in weekly_series: wr.append([Paragraph(label,styles["table"]),Paragraph(safe_money(spent),styles["table_right"]),Paragraph(safe_money(payments),styles["table_right"])])
    wt=Table(wr,colWidths=[55*mm,55*mm,55*mm],repeatRows=1)
    wt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0B1220")),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.3,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("ALIGN",(1,1),(-1,-1),"RIGHT"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story += [wt,PageBreak()]

    # Signals — redesigned intelligence dashboard.
    signals=[
        ("01","CLASSIFICATION GAP",f"{safe_money(categories.get('Uncategorized',0))} is Uncategorized across {category_counts.get('Uncategorized',0):,} transactions.",colors.HexColor("#7C5CFC"),colors.HexColor("#F3F0FF"),True),
        ("02","FREQUENCY PATTERN",f"Food & Dining has {category_counts.get('Food & Dining',0):,} transactions and {safe_money(categories.get('Food & Dining',0))} of spend.",colors.HexColor("#4F7CFF"),colors.HexColor("#EEF4FF"),True),
        ("03","CREDIT POSITION",f"{utilization:.1f}% utilization leaves {safe_money(available_credit or 0)} available." if utilization is not None else "Credit utilization was not supplied.",colors.HexColor("#22C55E"),colors.HexColor("#ECFDF3"),True),
        ("04","STATEMENT INTEGRITY",("Calculated balance matches the supplied closing balance." if diff is not None and abs(diff)<0.01 else "A complete balance reconciliation could not be confirmed from the supplied metadata."),colors.HexColor("#F59E0B"),colors.HexColor("#FFF7E8"),False),
    ]
    recommendations=["Prioritize the largest Uncategorized merchants for classification.","Monitor Food & Dining by frequency, not only total value.","Use the ledger as the evidence layer; do not duplicate transactions in summary sections."]
    story += [Spacer(1,4*mm),Paragraph("06 / FINORA SIGNALS",ParagraphStyle("kick6",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Reconciliation & intelligence",styles["heading"]),Paragraph("Control layer first, interpretation second.",styles["small"]),Spacer(1,3*mm),Concept3SignalsDashboard(signals,recommendations),PageBreak()]

    # Complete canonical ledger. No deduplication here.
    show_review = review_count > 0
    ledger_headers=["DATE","MERCHANT","DESCRIPTION","AMOUNT","CATEGORY","TYPE","DIRECTION"] + (["REVIEW"] if show_review else [])
    widths=[15*mm,34*mm,38*mm,22*mm,31*mm,18*mm,14*mm] + ([12*mm] if show_review else [])
    per_page=29
    for start in range(0,len(txs),per_page):
        chunk=txs[start:start+per_page]
        story += [Spacer(1,4*mm),Paragraph(f"07 / CANONICAL LEDGER · {start//per_page+1}",ParagraphStyle("ledger_k",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("Complete transaction ledger",styles["heading"]),Paragraph("Every supplied canonical transaction is preserved once. Repeated merchant names remain when the underlying transaction is different.",styles["small"]),Spacer(1,3*mm)]
        rows=[[Paragraph(h,styles["table_head"]) for h in ledger_headers]]
        for t in chunk:
            date=getattr(t,"transaction_date",None); direction=_enum(getattr(t,"direction",None)) or "Debit"; amount=abs(_n(getattr(t,"original_amount",0)))
            amount_text=("+" if direction.lower()=="credit" else "-")+f"{amount:,.2f}"
            row=[Paragraph(date.strftime("%d %b") if date else "-",styles["table"]),Paragraph(escape(_merchant_name(t)),styles["table"]),Paragraph(escape(_safe_text(getattr(t,"description_raw",None))),styles["table"]),Paragraph(amount_text,styles["table_right"]),Paragraph(escape(_cat_name(t)),styles["table"]),Paragraph(escape(_safe_text(getattr(t,"transaction_type",None))),styles["table"]),Paragraph(escape(direction),styles["table"])]
            if show_review: row.append(Paragraph("Yes" if getattr(t,"requires_review",False) else "No",styles["table"]))
            rows.append(row)
        tbl=Table(rows,colWidths=widths,repeatRows=1,splitByRow=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0B1220")),("BOX",(0,0),(-1,-1),.45,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.25,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3.2),("BOTTOMPADDING",(0,0),(-1,-1),3.2)]))
        story.append(tbl)
        if start+per_page>=len(txs):
            story += [Spacer(1,3*mm),Paragraph("Ledger rule: only exact duplicate source records are candidates for deduplication. Different transactions from the same merchant are intentionally retained.",styles["small"])]
        story.append(PageBreak())

    # Final logic page
    story += [Spacer(1,4*mm),Paragraph("08 / REPORT LOGIC",ParagraphStyle("kick8",fontName="Helvetica-Bold",fontSize=7.5,textColor=colors.HexColor("#7C5CFC"))),Paragraph("One story, one source of truth",styles["heading"]),Paragraph("The final report separates decision-making from evidence while preserving the complete transaction set.",styles["small"]),Spacer(1,4*mm)]
    rules=[("EXECUTIVE","Balances, limit, utilization and statement-level read."),("SPENDING MAP","One visual composition of total outgoing spend."),("MONEY FLOW","Opening → spending → payments → closing."),("BEHAVIOUR","Value vs transaction frequency."),("MERCHANTS","Concentration and repeated activity."),("TREND","Weekly spending versus payments/credits."),("INTELLIGENCE","Reconciliation plus concise Finora signals."),("LEDGER",f"All {len(txs):,} supplied canonical transactions, once.")]
    rt=Table([[Paragraph(f"<b>{a}</b>",styles["small"]),Paragraph(b,styles["body"])] for a,b in rules],colWidths=[42*mm,124*mm])
    rt.setStyle(TableStyle([("BOX",(0,0),(-1,-1),.5,colors.HexColor("#E3E7EF")),("INNERGRID",(0,0),(-1,-1),.3,colors.HexColor("#E3E7EF")),("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,colors.HexColor("#FAFBFD")]),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    story += [rt,Spacer(1,10*mm)]
    endbox=Table([[Paragraph("FINORA AI",ParagraphStyle("endbrand",fontName="Helvetica-Bold",fontSize=19,textColor=colors.white,alignment=TA_CENTER)),Paragraph("Financial intelligence that is easy to scan, easy to verify, and difficult to misunderstand.",ParagraphStyle("endcopy",fontName="Helvetica",fontSize=9.2,leading=13,textColor=colors.HexColor("#D5DBE7"),alignment=TA_CENTER))]],colWidths=[58*mm,108*mm])
    endbox.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#0B1220")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),15),("BOTTOMPADDING",(0,0),(-1,-1),15)]))
    story.append(endbox)

    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm,title="Finora AI Financial Intelligence Report",author="Finora AI")
    doc.build(story,onFirstPage=_c3_chrome,onLaterPages=_c3_chrome)
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
