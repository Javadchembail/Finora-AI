
from __future__ import annotations

import io
import json
import os
import time
from collections import Counter
from decimal import Decimal
from html import escape

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from ingestion.pipeline import FinancialStatementPipeline
from transactions.models import Transaction, TransactionDirection
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
from ai.rag_service import StatementRAG

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
    "transactions_backup": [],
    "file_name": None,
    "chat_history": [],
    "ai_summary": None,
    "transaction_focus": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

requested_page = st.query_params.get("page")
if requested_page in {"Overview", "Transactions", "AI"}:
    st.session_state.page = requested_page


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


def get_currency(transactions):
    currencies = [
        str(getattr(t, "original_currency", "")).upper()
        for t in transactions
        if getattr(t, "original_currency", None)
    ]

    if not currencies:
        return "UNKNOWN"

    return Counter(currencies).most_common(1)[0][0]


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


class FinoraPDFPasswordError(Exception):
    """Customer-safe PDF password validation error."""


def _validate_pdf_password(uploaded, password):
    """Validate encrypted PDFs before running the universal pipeline.

    This gives the UI a precise message for a missing or incorrect password
    instead of allowing the PDF reader to fail later with a generic error.
    """
    filename = str(getattr(uploaded, "name", "")).lower()
    if not filename.endswith(".pdf") or PdfReader is None:
        return

    try:
        raw = uploaded.getvalue()
        reader = PdfReader(io.BytesIO(raw), strict=False)
    except Exception:
        # Let the universal pipeline handle malformed/non-standard PDFs.
        return

    if not getattr(reader, "is_encrypted", False):
        return

    clean_password = str(password or "")
    if not clean_password:
        raise FinoraPDFPasswordError("PDF_PASSWORD_REQUIRED")

    try:
        result = reader.decrypt(clean_password)
    except Exception:
        result = 0

    if not result:
        raise FinoraPDFPasswordError("INCORRECT_PDF_PASSWORD")


def _restore_transaction_session():
    """Restore transactions from the session backup after a Streamlit rerun."""
    current = st.session_state.get("transactions") or []
    backup = st.session_state.get("transactions_backup") or []

    if current:
        return current
    if not backup:
        return []

    restored = []
    for item in backup:
        try:
            restored.append(Transaction.model_validate(item))
        except Exception:
            # A stale backup should never crash the UI.
            continue

    st.session_state.transactions = restored
    return restored


def _analyze_uploaded_statement(uploaded, password, loader_placeholder=None):
    """Run the existing universal pipeline from any UI entry point."""
    os.makedirs("storage", exist_ok=True)

    file_path = os.path.join(
        "storage",
        uploaded.name,
    )

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

    # Validate encrypted PDFs before starting the expensive analysis.
    _validate_pdf_password(uploaded, password)

    pipeline = FinancialStatementPipeline(
        file_path=file_path,
        password=password or None,
    )

    result = pipeline.run()

    # Keep both the live objects and a Pydantic-safe backup. This prevents
    # navigation/reruns from making the Transactions and AI pages appear empty.
    st.session_state.transactions = list(result or [])
    st.session_state.transactions_backup = [
        transaction.model_dump(mode="json")
        for transaction in st.session_state.transactions
    ]
    st.session_state.file_name = uploaded.name

    # A newly analyzed statement starts with a clean transaction view.
    # Do not carry filters/focus from a previous statement into the new one.
    st.session_state.transaction_search = ""
    st.session_state.transaction_direction = "All"
    st.session_state.transaction_status = "All"
    st.session_state.transaction_category = "All"
    st.session_state.transaction_focus = None
    st.session_state.transaction_filter_file = uploaded.name

    st.session_state.chat_history = []
    st.session_state.ai_summary = None
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
# TOP NAVIGATION — SESSION-PRESERVING CONTROLS
# ============================================================

transactions = _restore_transaction_session()

nav_brand, nav_links, nav_status = st.columns([3.3, 5.2, 1.25])

with nav_brand:
    render("""
    <div class="topbar">
        <div class="brand">
            <div class="logo finora-f-logo">F</div>
            <div>
                <div class="brand-name">Finora AI</div>
                <div class="brand-sub">Intelligent Financial Intelligence</div>
            </div>
        </div>
    </div>
    """)

with nav_links:
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        if st.button("Overview", key="nav_overview", width="stretch"):
            st.session_state.page = "Overview" if transactions else "Home"
            st.query_params["page"] = st.session_state.page
            st.rerun()

    with nav2:
        if st.button("Transactions", key="nav_transactions", width="stretch"):
            if transactions:
                st.session_state.page = "Transactions"
                st.query_params["page"] = "Transactions"
                st.rerun()
            else:
                st.session_state.page = "Home"
                st.query_params["page"] = "Home"
                st.rerun()

    with nav3:
        if st.button("AI Intelligence", key="nav_ai", width="stretch"):
            if transactions:
                st.session_state.page = "AI"
                st.query_params["page"] = "AI"
                st.rerun()
            else:
                st.session_state.page = "Home"
                st.query_params["page"] = "Home"
                st.rerun()

with nav_status:
    render("""<div class="ai-ready" style="margin-top:9px;text-align:center;">● AI READY</div>""")


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

                except FinoraPDFPasswordError as exc:
                    loader.empty()

                    if str(exc) == "PDF_PASSWORD_REQUIRED":
                        st.warning(
                            "🔐 This PDF is password protected. Please enter the PDF password to continue."
                        )
                    else:
                        st.error(
                            "🔐 The password you entered is incorrect. Please check the password and try again."
                        )

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
                        st.error(
                            "🔐 The password you entered is incorrect. Please check the password and try again."
                        )
                    else:
                        st.error(
                            "Finora could not read this statement. Please check the file and try again."
                        )

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
            render(f"<div class='metric'><div class='metric-label'>TRANSACTIONS</div><div class='metric-value'>{len(transactions):,}</div><div class='metric-sub'>Extracted from the statement</div></div>")

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
# OVERVIEW
# ============================================================

if st.session_state.page == "Overview":

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

    if net_cash_flow > 0:
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
                <div class="cockpit-metric-label">Received</div>
                <div class="cockpit-metric-value">
                    {money(income, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    money coming in
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">Spent</div>
                <div class="cockpit-metric-value">
                    {money(expenses, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    money going out
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">Net position</div>
                <div class="cockpit-metric-value">
                    {money(net_cash_flow, currency)}
                </div>
                <div class="cockpit-metric-sub">
                    received minus spent
                </div>
            </div>

            <div class="cockpit-metric">
                <div class="cockpit-metric-label">Activity</div>
                <div class="cockpit-metric-value">
                    {len(transactions):,}
                </div>
                <div class="cockpit-metric-sub">
                    {review_count} needing review
                </div>
            </div>

        </div>

    </div>
    """)

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

                share = (
                    amount / total_spend * 100
                    if total_spend > 0
                    else 0
                )

                width = (
                    amount / max_amount * 100
                    if max_amount > 0
                    else 0
                )

                render(f"""
                <div class="rank-row">

                    <div class="rank-top">

                        <div class="rank-number">
                            {position:02d}
                        </div>

                        <div class="rank-name">
                            {display_name}
                        </div>

                        <div class="rank-amount">
                            {money(amount, currency)}
                        </div>

                    </div>

                    <div class="rank-track">
                        <span style="width:{min(100, width):.1f}%"></span>
                    </div>

                    <div class="rank-sub">
                        {share:.1f}% of outgoing money
                    </div>

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
                            "width":2.3,
                            "shape":"spline",
                        },
                        marker={"size":5},
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
                            "width":2.3,
                            "shape":"spline",
                        },
                        marker={"size":5},
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
                        mode="lines",
                        line={
                            "width":1.6,
                            "dash":"dot",
                        },
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Net: %{y:,.2f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.update_layout(
                    height=340,
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
                        "b":0,
                    },
                    hovermode="x unified",
                    hoverlabel={
                        "bgcolor":"#111827",
                        "bordercolor":"#334155",
                        "font":{"color":"#f8fafc"},
                    },
                    xaxis={
                        "showgrid":False,
                        "zeroline":False,
                        "fixedrange":True,
                        "tickfont":{"size":12},
                    },
                    yaxis={
                        "showgrid":True,
                        "gridcolor":"rgba(51,65,85,.22)",
                        "zeroline":False,
                        "fixedrange":True,
                        "tickfont":{"size":12},
                        "tickformat":"~s",
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
        Search, filter and inspect every transaction extracted by Finora.
    </div>
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        search = st.text_input(
            "Search",
            placeholder="Merchant or description...",
            key="transaction_search",
        )

    with c2:
        direction = st.selectbox(
            "Direction",
            ["All", "credit", "debit"],
            key="transaction_direction",
        )

    with c3:
        status = st.selectbox(
            "Status",
            ["All", "Needs Review", "Validated"],
            key="transaction_status",
        )

    with c4:
        categories = sorted({
            str(value).strip()
            for value in df.get("Category", pd.Series(dtype=str)).dropna().tolist()
            if str(value).strip()
        })
        category_filter = st.selectbox(
            "Category",
            ["All"] + categories,
            key="transaction_category",
        )

    # Transaction filters are statement-scoped. If this is the first visit to
    # Transactions for the currently loaded statement, initialize them cleanly.
    current_file = st.session_state.get("file_name") or ""
    filter_file = st.session_state.get("transaction_filter_file")
    if filter_file != current_file:
        st.session_state.transaction_search = ""
        st.session_state.transaction_direction = "All"
        st.session_state.transaction_status = "All"
        st.session_state.transaction_category = "All"
        st.session_state.transaction_focus = None
        st.session_state.transaction_filter_file = current_file
        st.rerun()

    filtered = df.copy()
    focus = st.session_state.get("transaction_focus")
    if focus and focus.get("value"):
        focus_value = _safe_focus_value(focus.get("value"))
        focus_kind = str(focus.get("kind") or "").casefold()

        if focus_kind == "category" and "Category" in filtered.columns:
            candidate = filtered[
                filtered["Category"].fillna("").astype(str).str.casefold()
                == focus_value.casefold()
            ]
        elif focus_kind == "merchant" and "Merchant" in filtered.columns:
            candidate = filtered[
                filtered["Merchant"].fillna("").astype(str).str.casefold()
                == focus_value.casefold()
            ]
        else:
            candidate = filtered

        # Never let an old/broken focus filter make the entire table appear
        # empty. A focus action is only applied when it actually matches data.
        if len(candidate) > 0:
            filtered = candidate
            render(
                f"<div class='focus-filter-banner'>"
                f"Showing transactions for <strong>{escape(focus_value)}</strong>. "
                f"<span>Clear the filter below to see all transactions.</span>"
                f"</div>"
            )
        else:
            st.session_state.transaction_focus = None

    if search.strip():
        query = search.strip()
        merchant_text = filtered["Merchant"].fillna("").astype(str)
        description_text = filtered["Description"].fillna("").astype(str)
        filtered = filtered[
            merchant_text.str.contains(query, case=False, na=False, regex=False)
            | description_text.str.contains(query, case=False, na=False, regex=False)
        ]

    if direction != "All":
        filtered = filtered[
            filtered["Direction"].fillna("").astype(str).str.casefold()
            == direction.casefold()
        ]

    if status == "Needs Review":
        filtered = filtered[filtered["Review"].fillna(False).astype(bool)]
    elif status == "Validated":
        filtered = filtered[~filtered["Review"].fillna(False).astype(bool)]

    if category_filter != "All" and "Category" in filtered.columns:
        filtered = filtered[
            filtered["Category"].fillna("").astype(str).str.casefold()
            == category_filter.casefold()
        ]

    st.caption(
        f"{len(filtered):,} transactions"
    )

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        height=600,
    )

    if focus and focus.get("value"):
        if st.button("Clear focused transaction view", key="clear_transaction_focus"):
            st.session_state.transaction_focus = None
            st.rerun()

    st.download_button(
        "Download CSV",
        filtered.to_csv(
            index=False
        ).encode("utf-8"),
        "finora_transactions.csv",
        "text/csv",
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
