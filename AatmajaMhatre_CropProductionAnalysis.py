"""
Indian Crop Production Analysis – Streamlit BI Dashboard (v2)
=============================================================
Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Indian Crop Production Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
PALETTE = [
    "#2E7D5B", "#3D9970", "#52B788", "#74C69D",
    "#95D5B2", "#1B5E45", "#A8DABC", "#D8F3DC",
    "#F2795B", "#E05C3A", "#F4A287", "#FAD2C8",
]
CHART_TEMPLATE = "plotly_white"
PRIMARY   = "#2E7D5B"
SECONDARY = "#F2795B"
SIDEBAR_BG = "#1F2430"
APP_BG     = "#EEF3F1"
CARD_BG    = "#FFFFFF"
TEXT_DARK  = "#2B2F36"
TEXT_MUTED = "#8A94A6"
BORDER     = "#E2E8E4"

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font: Inter ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── hide Streamlit chrome ── */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"]  { display: none !important; }

/* ── global typography & background ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #EEF3F1 !important;
    color: #2B2F36;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1F2430 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #D4DBE8 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small {
    color: #A8B4C8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
/* sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background-color: #2A3245 !important;
    border-color: #3A4560 !important;
    color: #D4DBE8 !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background-color: #2E7D5B !important;
}
/* sidebar radio active */
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    display: none;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: #2A3245;
    border-radius: 6px;
    padding: 7px 12px !important;
    margin-bottom: 4px;
    display: block;
    font-size: 0.82rem !important;
    color: #A8B4C8 !important;
    cursor: pointer;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    background: #2E7D5B !important;
    color: #FFFFFF !important;
}
/* sidebar reset button */
[data-testid="stSidebar"] .stButton > button {
    background-color: #2E7D5B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.80rem !important;
    font-weight: 600 !important;
    padding: 7px 0 !important;
    width: 100%;
    letter-spacing: 0.04em;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #245f47 !important;
}
/* sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #2A3245 !important;
    margin: 10px 0 !important;
}

/* ── main content padding ── */
.block-container {
    padding: 24px 32px 48px 32px !important;
    max-width: 1400px !important;
}

/* ── page top-bar ── */
.page-topbar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 14px;
    border-bottom: 1px solid #E2E8E4;
}
.page-topbar .breadcrumb {
    font-size: 0.72rem;
    color: #8A94A6;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.page-topbar .page-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #2B2F36;
    margin: 0;
    line-height: 1.2;
}
.page-topbar .meta-right {
    font-size: 0.72rem;
    color: #8A94A6;
    text-align: right;
    letter-spacing: 0.03em;
}

/* ── active filter badges ── */
.filter-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-bottom: 20px;
}
.filter-badge {
    display: inline-flex;
    align-items: center;
    background: #FFFFFF;
    border: 1px solid #2E7D5B;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #2E7D5B;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.filter-none {
    font-size: 0.75rem;
    color: #8A94A6;
}

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px 22px 16px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 0 0 1px #E2E8E4;
    position: relative;
    overflow: hidden;
    min-height: 110px;
}
.kpi-card .kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8A94A6;
    margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #2B2F36;
    line-height: 1.15;
    margin-bottom: 6px;
}
.kpi-card .kpi-delta {
    font-size: 0.75rem;
    font-weight: 500;
}
.kpi-card .kpi-delta.pos { color: #2E7D5B; }
.kpi-card .kpi-delta.neg { color: #D94F3D; }
.kpi-card .kpi-delta.neu { color: #8A94A6; }
.kpi-card .kpi-icon {
    position: absolute;
    top: 18px; right: 18px;
    width: 36px; height: 36px;
    border-radius: 50%;
    background: #EEF3F1;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    color: #2E7D5B;
}

/* ── chart section cards ── */
.chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 0 0 1px #E2E8E4;
    padding: 0 0 4px 0;
    margin-bottom: 24px;
    overflow: hidden;
}
.chart-card-header {
    padding: 11px 18px;
    border-bottom: 1px solid #F0F2F0;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8A94A6;
}
.chart-card-body {
    padding: 12px 12px 4px 12px;
}

/* ── section dividers ── */
.section-divider {
    height: 1px;
    background: #E2E8E4;
    margin: 28px 0 24px 0;
    border: none;
}

/* ── risk / opportunity / action cards ── */
.risk-card {
    background: #FFFFFF;
    border-left: 4px solid #D94F3D;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin-bottom: 8px;
    font-size: 0.84rem;
    color: #2B2F36;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.opp-card {
    background: #FFFFFF;
    border-left: 4px solid #2E7D5B;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin-bottom: 8px;
    font-size: 0.84rem;
    color: #2B2F36;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.action-card {
    background: #FFFFFF;
    border-left: 4px solid #3B6FCC;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    margin-bottom: 8px;
    font-size: 0.84rem;
    color: #2B2F36;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── inline alert replacements ── */
.inline-success {
    background: #FFFFFF;
    border-left: 4px solid #2E7D5B;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #2B2F36;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.inline-info {
    background: #FFFFFF;
    border-left: 4px solid #3B6FCC;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #2B2F36;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.inline-warn {
    background: #FFFFFF;
    border-left: 4px solid #D94F3D;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.84rem;
    color: #2B2F36;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── dataframe clean style ── */
[data-testid="stDataFrame"] table {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.80rem !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: #F5F7F6 !important;
    color: #8A94A6 !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background: #FAFBFA !important;
}

/* ── st.metric override ── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 0 0 1px #E2E8E4;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8A94A6 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #2B2F36 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* ── download button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #2E7D5B !important;
    color: #2E7D5B !important;
    border-radius: 6px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 5px 14px !important;
}
.stDownloadButton > button:hover {
    background: #2E7D5B !important;
    color: #FFFFFF !important;
}

/* ── predict button ── */
.stButton > button[kind="primary"] {
    background: #2E7D5B !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #245f47 !important;
}

/* ── footer ── */
.app-footer {
    text-align: center;
    color: #8A94A6;
    font-size: 0.72rem;
    padding: 20px 0 10px 0;
    border-top: 1px solid #E2E8E4;
    margin-top: 48px;
    letter-spacing: 0.03em;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_data(path: str = "APY.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Convert all object columns to string and strip whitespace
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    # Treat "nan" strings (from astype(str) on NaN) as actual NaN
    str_cols = ["State", "District", "Crop", "Season"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].replace({"nan": pd.NA, "": pd.NA, "NaN": pd.NA})
    # Drop rows where any key text column is missing
    df = df.dropna(subset=[c for c in str_cols if c in df.columns])
    # Ensure text columns are plain str after NA removal
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    # Coerce and validate numerics
    for col in ["Area", "Production", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Production", "Area"])
    df = df[(df["Area"] > 0) & (df["Production"] > 0)]
    df["Yield"] = df["Production"] / df["Area"]
    df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce")
    df = df.dropna(subset=["Crop_Year"])
    df["Crop_Year"] = df["Crop_Year"].astype(int)
    return df


@st.cache_data(show_spinner="Loading India GeoJSON…")
def load_india_geojson() -> dict | None:
    urls = [
        "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
        "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


df_raw = load_data()

# ── helpers ────────────────────────────────────────────────────────────────────
def excl_coconut(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Crop"].str.lower() != "coconut"]


def fmt_num(n: float, unit: str = "") -> str:
    """Format large numbers as readable strings."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f} Bn {unit}".strip()
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} Mn {unit}".strip()
    if n >= 1_000:
        return f"{n/1_000:.1f} K {unit}".strip()
    return f"{n:.2f} {unit}".strip()


def empty_fig(msg: str = "No data for current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#888"), xref="paper", yref="paper")
    fig.update_layout(template=CHART_TEMPLATE, height=300,
                      xaxis_visible=False, yaxis_visible=False)
    return fig


# ── global data statistics (for banners) ─────────────────────────────────────
YEAR_MIN = int(df_raw["Crop_Year"].min())
YEAR_MAX = int(df_raw["Crop_Year"].max())
N_STATES = df_raw["State"].nunique()
N_CROPS  = df_raw["Crop"].nunique()
DATA_META = f"{YEAR_MIN}–{YEAR_MAX} · {N_STATES} States · {N_CROPS}+ Crops"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – GLOBAL FILTERS (apply to all pages)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:18px 4px 6px 4px;'>"
        "<span style='font-size:1.05rem;font-weight:700;color:#FFFFFF;"
        "letter-spacing:0.06em;text-transform:uppercase;'>Crop Production</span>"
        "<br><span style='font-size:0.72rem;color:#6B7A99;'>India Agriculture BI</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#2A3245;margin:8px 0 14px 0;'>", unsafe_allow_html=True)

    # ── navigation ────────────────────────────────────────────────────────────
    PAGES = [
        "Executive Overview",
        "Crop, Season & Region",
        "Risks & Opportunities",
        "Yield Predictor",
    ]
    page = st.radio("Navigate to", PAGES, label_visibility="collapsed")
    st.markdown("<hr style='border-color:#2A3245;margin:14px 0 10px 0;'>", unsafe_allow_html=True)

    # ── global filters ────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.68rem;font-weight:700;letter-spacing:0.1em;"
        "text-transform:uppercase;color:#6B7A99;margin-bottom:10px;'>Filters</p>",
        unsafe_allow_html=True,
    )

    year_range = st.slider(
        "Crop Year range",
        min_value=YEAR_MIN, max_value=YEAR_MAX,
        value=(YEAR_MIN, YEAR_MAX),
        help="Filter all pages by harvest year range.",
    )

    all_states  = sorted(df_raw["State"].dropna().unique())
    all_crops   = sorted(df_raw["Crop"].dropna().unique())
    all_seasons = sorted(df_raw["Season"].dropna().unique())

    sel_states  = st.multiselect(
        "State(s)", all_states, default=[],
        help="Leave blank to include all states.",
    )
    sel_crops   = st.multiselect(
        "Crop(s)", all_crops, default=[],
        help="Leave blank to include all crops.",
    )
    sel_seasons = st.multiselect(
        "Season(s)", all_seasons, default=[],
        help="Kharif (Jun–Nov), Rabi (Nov–Apr), Summer, Autumn, Whole Year.",
    )

    if st.button("Reset Filters", use_container_width=True):
        st.session_state["_reset"] = True
        st.rerun()

    # honour reset
    if st.session_state.get("_reset"):
        sel_states  = []
        sel_crops   = []
        sel_seasons = []
        year_range  = (YEAR_MIN, YEAR_MAX)
        st.session_state["_reset"] = False

    st.markdown("<hr style='border-color:#2A3245;margin:14px 0 8px 0;'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.68rem;color:#6B7A99;text-align:center;"
        f"letter-spacing:0.02em;'>{DATA_META}</p>",
        unsafe_allow_html=True,
    )


# ── apply global filters ───────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Crop_Year"] >= year_range[0]) & (df["Crop_Year"] <= year_range[1])]
    if sel_states:
        df = df[df["State"].isin(sel_states)]
    if sel_crops:
        df = df[df["Crop"].isin(sel_crops)]
    if sel_seasons:
        df = df[df["Season"].isin(sel_seasons)]
    return df


df_filtered = apply_filters(df_raw)


# ── active-filter banner ───────────────────────────────────────────────────────
def active_filters_banner():
    badges = []
    if year_range != (YEAR_MIN, YEAR_MAX):
        badges.append(f"{year_range[0]}–{year_range[1]}")
    for s in sel_states:
        badges.append(s)
    for c in sel_crops:
        badges.append(c)
    for sn in sel_seasons:
        badges.append(sn)

    if badges:
        pills = "".join(f'<span class="filter-badge">{b}</span>' for b in badges)
        st.markdown(
            f'<div class="filter-bar">'
            f'<span style="font-size:0.72rem;color:#8A94A6;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.06em;">Active filters:</span> {pills}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="filter-bar"><span class="filter-none">No filters active — showing full dataset.</span></div>',
            unsafe_allow_html=True,
        )

    csv_bytes = df_filtered.to_csv(index=False).encode()
    st.download_button(
        "Download filtered dataset (CSV)",
        data=csv_bytes,
        file_name="crop_production_filtered.csv",
        mime="text/csv",
        help="Download the currently filtered dataset as a CSV file.",
    )


# ── page top-bar helper ────────────────────────────────────────────────────────
def page_banner(title: str, description: str):
    rows = len(df_filtered)
    crumb = "Home / " + title
    st.markdown(
        f"""<div class="page-topbar">
            <div>
                <div class="breadcrumb">{crumb}</div>
                <div class="page-title">{title}</div>
            </div>
            <div class="meta-right">{DATA_META}<br>{rows:,} rows after filters</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── section card helpers ───────────────────────────────────────────────────────
def card_header(label: str):
    st.markdown(
        f'<div class="chart-card-header">{label}</div>',
        unsafe_allow_html=True,
    )


# ── empty-data guard ───────────────────────────────────────────────────────────
def check_empty(df: pd.DataFrame, label: str = "selection") -> bool:
    if df.empty:
        st.markdown(
            f'<div class="inline-warn">No data found for the current {label}. '
            "Try adjusting the filters in the sidebar.</div>",
            unsafe_allow_html=True,
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    page_banner("Executive Overview",
                "National-level crop production KPIs, trends, and composition.")
    active_filters_banner()

    df = excl_coconut(df_filtered)
    if check_empty(df, "filters"):
        st.stop()

    # ── KPI calculations ──────────────────────────────────────────────────────
    total_prod    = df["Production"].sum()
    avg_yield     = df["Yield"].mean()
    total_area    = df["Area"].sum()

    years_sorted  = sorted(df["Crop_Year"].unique())
    crop_prod     = df.groupby("Crop")["Production"].sum()
    top_crop      = crop_prod.idxmax()
    top_share     = (crop_prod.max() / crop_prod.sum()) * 100

    # delta: compare last year vs year before
    def _yearly_mean(col: str, yr: int) -> float:
        sub = df[df["Crop_Year"] == yr]
        return sub[col].sum() if col == "Production" else sub[col].mean()

    if len(years_sorted) >= 2:
        yr_last  = years_sorted[-1]
        yr_prev  = years_sorted[-2]
        delta_prod  = _yearly_mean("Production", yr_last) - _yearly_mean("Production", yr_prev)
        delta_yield = _yearly_mean("Yield",      yr_last) - _yearly_mean("Yield",      yr_prev)
        delta_area  = df[df["Crop_Year"] == yr_last]["Area"].sum() - \
                      df[df["Crop_Year"] == yr_prev]["Area"].sum()
    else:
        delta_prod = delta_yield = delta_area = None

    # 5-year yield growth
    if len(years_sorted) >= 10:
        r5 = years_sorted[-5:]; p5 = years_sorted[-10:-5]
        yg = ((df[df["Crop_Year"].isin(r5)]["Yield"].mean() -
               df[df["Crop_Year"].isin(p5)]["Yield"].mean()) /
              (df[df["Crop_Year"].isin(p5)]["Yield"].mean() or 1)) * 100
    elif len(years_sorted) >= 2:
        yg = ((df[df["Crop_Year"] == years_sorted[-1]]["Yield"].mean() -
               df[df["Crop_Year"] == years_sorted[0]]["Yield"].mean()) /
              (df[df["Crop_Year"] == years_sorted[0]]["Yield"].mean() or 1)) * 100
    else:
        yg = 0.0

    # ── KPI row ───────────────────────────────────────────────────────────────
    def _delta_html(val, unit="") -> str:
        if val is None:
            return '<span class="kpi-delta neu">— no prior period</span>'
        sign = "+" if val >= 0 else ""
        cls  = "pos" if val >= 0 else "neg"
        arrow = "▲" if val >= 0 else "▼"
        return f'<span class="kpi-delta {cls}">{arrow} {sign}{fmt_num(abs(val), unit)} vs prev year</span>'

    def _delta_pct_html(val) -> str:
        if val is None:
            return '<span class="kpi-delta neu">—</span>'
        sign = "+" if val >= 0 else ""
        cls  = "pos" if val >= 0 else "neg"
        arrow = "▲" if val >= 0 else "▼"
        return f'<span class="kpi-delta {cls}">{arrow} {sign}{val:.4f} t/ha vs prev year</span>'

    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, "Total Production",      fmt_num(total_prod, "MT"),  _delta_html(delta_prod, "MT"),
         "bar_chart",
         "Sum of all crop production in metric tonnes (Coconut excluded). Delta = latest year vs prior year."),
        (k2, "Average Yield",          f"{avg_yield:.3f} t/ha",    _delta_pct_html(delta_yield),
         "agriculture",
         "Mean yield (Production / Area). Delta = latest year avg vs prior year avg."),
        (k3, "5-Year Yield Growth",    f"{yg:+.1f}%",              '<span class="kpi-delta neu">Recent 5 yrs vs prior 5 yrs</span>',
         "trending_up",
         "Percentage change in average yield: most-recent 5 crop-years vs the preceding 5 crop-years."),
        (k4, f"Top Crop — {top_crop}", f"{top_share:.1f}% share",  '<span class="kpi-delta neu">Share of national production</span>',
         "emoji_nature",
         "Crop with the highest total production share (Coconut excluded from tonnage)."),
    ]
    for col, label, value, delta_html, icon, tip in kpi_data:
        with col:
            st.markdown(
                f'<div class="kpi-card" title="{tip}">'
                f'  <div class="kpi-label">{label}</div>'
                f'  <div class="kpi-value">{value}</div>'
                f'  {delta_html}'
                f'  <div class="kpi-icon"><span class="material-symbols-outlined" '
                f'style="font-size:1.1rem;">{icon}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── production trend ──────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("National Production Trend by Year")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    trend = df.groupby("Crop_Year")["Production"].sum().reset_index()
    trend.columns = ["Crop_Year", "Total_Production"]
    fig_trend = px.area(
        trend, x="Crop_Year", y="Total_Production",
        template=CHART_TEMPLATE,
        color_discrete_sequence=[PRIMARY],
        labels={"Crop_Year": "Crop Year", "Total_Production": "Total Production (MT)"},
        hover_data={"Total_Production": ":,.0f"},
    )
    fig_trend.update_traces(
        line_width=2,
        fillcolor="rgba(46,125,91,0.12)",
        hovertemplate="<b>Year:</b> %{x}<br><b>Production:</b> %{y:,.0f} MT<extra></extra>",
    )
    fig_trend.update_layout(
        height=300, margin=dict(t=8, b=8, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=False, linecolor=BORDER),
        yaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── animated bar-chart race (top-10 states by production over years) ──────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Bar Chart Race — Top 10 States by Production")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    st.caption("Press Play to animate year-by-year.")

    with st.spinner("Building animation…"):
        race_df = (
            df.groupby(["Crop_Year", "State"])["Production"]
            .sum()
            .reset_index()
        )
        # keep only top-10 states per year
        race_top = (
            race_df.sort_values(["Crop_Year", "Production"], ascending=[True, False])
            .groupby("Crop_Year")
            .head(10)
            .reset_index(drop=True)
        )
        # rank within year for horizontal ordering
        race_top["Rank"] = race_top.groupby("Crop_Year")["Production"].rank(
            ascending=True, method="first"
        )
        if not race_top.empty:
            fig_race = px.bar(
                race_top,
                x="Production", y="State",
                animation_frame="Crop_Year",
                animation_group="State",
                orientation="h",
                color="State",
                color_discrete_sequence=PALETTE,
                template=CHART_TEMPLATE,
                labels={"Production": "Total Production (MT)", "State": ""},
                range_x=[0, race_top["Production"].max() * 1.1],
                hover_data={"Production": ":,.0f", "Crop_Year": True},
            )
            fig_race.update_traces(
                hovertemplate="<b>%{y}</b><br>Production: %{x:,.0f} MT<extra></extra>"
            )
            fig_race.update_layout(
                height=430,
                showlegend=False,
                margin=dict(t=8, b=8, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
                updatemenus=[dict(type="buttons", showactive=False,
                                  bgcolor="#2E7D5B", font=dict(color="#FFFFFF"),
                                  buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None, {"frame": {"duration": 600},
                                                             "transition": {"duration": 300},
                                                             "fromcurrent": True}]),
                                           dict(label="Pause",
                                                method="animate",
                                                args=[[None], {"frame": {"duration": 0},
                                                               "mode": "immediate"}])])],
            )
            st.plotly_chart(fig_race, use_container_width=True)
        else:
            st.plotly_chart(empty_fig(), use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    col_l, col_r = st.columns(2, gap="large")

    _chart_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
        xaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
        yaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)"),
        margin=dict(t=8, b=8, l=0, r=0),
    )

    # ── top 10 states by production ───────────────────────────────────────────
    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        card_header("Top 10 States — Production")
        st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
        sp = (df.groupby("State")["Production"].sum()
              .nlargest(10).reset_index().sort_values("Production"))
        fig_sp = px.bar(
            sp, x="Production", y="State", orientation="h",
            template=CHART_TEMPLATE,
            labels={"Production": "Total Production (MT)", "State": ""},
            color_discrete_sequence=[PRIMARY],
        )
        fig_sp.update_traces(
            marker_color=PRIMARY,
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} MT<extra></extra>",
        )
        fig_sp.update_layout(height=340, **_chart_layout)
        st.plotly_chart(fig_sp, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # ── top 10 states by yield ────────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        card_header("Top 10 States — Avg Yield")
        st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
        sy = (df.groupby("State")["Yield"].mean()
              .nlargest(10).reset_index().sort_values("Yield"))
        fig_sy = px.bar(
            sy, x="Yield", y="State", orientation="h",
            template=CHART_TEMPLATE,
            labels={"Yield": "Avg Yield (t/ha)", "State": ""},
            color_discrete_sequence=[SECONDARY],
        )
        fig_sy.update_traces(
            marker_color=SECONDARY,
            hovertemplate="<b>%{y}</b><br>%{x:.3f} t/ha<extra></extra>",
        )
        fig_sy.update_layout(height=340, **_chart_layout)
        st.plotly_chart(fig_sy, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── treemap: State → Crop production ─────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Production Treemap — State / Crop  (click to drill down)")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    with st.spinner("Building treemap…"):
        tm = (df.groupby(["State", "Crop"])["Production"].sum().reset_index())
        # limit to manageable size: top 15 states by production
        top_states_tm = df.groupby("State")["Production"].sum().nlargest(15).index
        tm = tm[tm["State"].isin(top_states_tm)]
        if not tm.empty:
            fig_tm = px.treemap(
                tm,
                path=[px.Constant("India"), "State", "Crop"],
                values="Production",
                color="Production",
                color_continuous_scale=[[0, "#D8F3DC"], [0.5, "#52B788"], [1, "#1B5E45"]],
                template=CHART_TEMPLATE,
                hover_data={"Production": ":,.0f"},
            )
            fig_tm.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Production: %{value:,.0f} MT<br>"
                    "Parent: %{parent}<extra></extra>"
                )
            )
            fig_tm.update_layout(
                height=480,
                margin=dict(t=8, b=8, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=11),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_tm, use_container_width=True)
        else:
            st.plotly_chart(empty_fig(), use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – CROP, SEASON & REGION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    page_banner("Crop, Season & Region Analysis",
                "Season comparisons, crop yield trends, district drill-down & map.")
    active_filters_banner()

    df  = apply_filters(df_raw)
    dfn = excl_coconut(df)
    if check_empty(dfn, "filters"):
        st.stop()

    _chart_layout2 = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
        margin=dict(t=8, b=8, l=0, r=0),
    )

    # ── season-wise comparison ────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Season-wise Production & Yield")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    sea = (dfn.groupby("Season")
           .agg(Total_Production=("Production","sum"), Avg_Yield=("Yield","mean"))
           .reset_index().sort_values("Total_Production", ascending=False))

    fig_sea = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sea.add_trace(
        go.Bar(x=sea["Season"], y=sea["Total_Production"],
               name="Total Production (MT)", marker_color=PRIMARY,
               hovertemplate="<b>%{x}</b><br>Production: %{y:,.0f} MT<extra></extra>"),
        secondary_y=False)
    fig_sea.add_trace(
        go.Scatter(x=sea["Season"], y=sea["Avg_Yield"],
                   name="Avg Yield (t/ha)", mode="lines+markers",
                   line=dict(color=SECONDARY, width=2.5),
                   hovertemplate="<b>%{x}</b><br>Yield: %{y:.3f} t/ha<extra></extra>"),
        secondary_y=True)
    fig_sea.update_layout(
        template=CHART_TEMPLATE, height=340,
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        **_chart_layout2,
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F0F2F0"),
    )
    fig_sea.update_yaxes(title_text="Total Production (MT)", secondary_y=False,
                         title_font=dict(size=11, color=TEXT_MUTED))
    fig_sea.update_yaxes(title_text="Avg Yield (t/ha)", secondary_y=True,
                         title_font=dict(size=11, color=TEXT_MUTED))
    st.plotly_chart(fig_sea, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── yield trend per crop ──────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Yield Trend Over Years — Top Crops")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    top_crops_default = (
        dfn.groupby("Crop").size().nlargest(10).index.tolist()
    )
    crops_for_trend = (
        [c for c in sel_crops if c.lower() != "coconut"]
        if sel_crops
        else top_crops_default
    )[:10]

    yt = (dfn[dfn["Crop"].isin(crops_for_trend)]
          .groupby(["Crop_Year", "Crop"])["Yield"].mean().reset_index())
    if not yt.empty:
        fig_yt = px.line(
            yt, x="Crop_Year", y="Yield", color="Crop",
            template=CHART_TEMPLATE,
            color_discrete_sequence=PALETTE,
            labels={"Crop_Year": "Crop Year", "Yield": "Avg Yield (t/ha)"},
        )
        fig_yt.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>"
                          "Year: %{x}<br>Yield: %{y:.3f} t/ha<extra></extra>"
        )
        fig_yt.update_layout(
            height=340,
            **_chart_layout2,
            xaxis=dict(showgrid=False, linecolor=BORDER),
            yaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig_yt, use_container_width=True)
    else:
        st.plotly_chart(empty_fig(), use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── district drill-down ───────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("District-level Drill-down")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    crop_opts = sel_crops if sel_crops else sorted(dfn["Crop"].dropna().unique())
    crop_for_dist = st.selectbox(
        "Select crop for district ranking",
        options=crop_opts,
        help="Choose a crop to rank districts by average yield.",
    )

    # state drill-down
    state_opts = sel_states if sel_states else sorted(dfn["State"].dropna().unique())
    state_for_dist = st.selectbox(
        "Drill into a specific state (optional — blank = all)",
        options=["All states"] + list(state_opts),
        help="Filter district ranking to a single state.",
    )

    df_dist_base = dfn[dfn["Crop"] == crop_for_dist]
    if state_for_dist != "All states":
        df_dist_base = df_dist_base[df_dist_base["State"] == state_for_dist]

    if check_empty(df_dist_base, f"crop '{crop_for_dist}'"):
        pass
    else:
        dist_yield = (df_dist_base.groupby(["State", "District"])["Yield"]
                      .mean().reset_index().sort_values("Yield", ascending=False))
        dist_yield["Label"] = dist_yield["District"] + " (" + dist_yield["State"] + ")"

        ct, cb = st.columns(2, gap="large")
        _dist_layout = dict(
            **_chart_layout2,
            xaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)"),
        )
        with ct:
            st.markdown(
                f'<p style="font-size:0.72rem;font-weight:600;color:{TEXT_MUTED};'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
                f'Top 10 Districts — {crop_for_dist}</p>',
                unsafe_allow_html=True,
            )
            t10 = dist_yield.head(10).sort_values("Yield")
            ft = px.bar(t10, x="Yield", y="Label", orientation="h",
                        template=CHART_TEMPLATE,
                        labels={"Yield": "Avg Yield (t/ha)", "Label": ""},
                        color_discrete_sequence=[PRIMARY])
            ft.update_traces(marker_color=PRIMARY,
                             hovertemplate="<b>%{y}</b><br>%{x:.3f} t/ha<extra></extra>")
            ft.update_layout(height=340, **_dist_layout)
            st.plotly_chart(ft, use_container_width=True)

        with cb:
            st.markdown(
                f'<p style="font-size:0.72rem;font-weight:600;color:{TEXT_MUTED};'
                f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">'
                f'Bottom 10 Districts — {crop_for_dist}</p>',
                unsafe_allow_html=True,
            )
            b10 = dist_yield.tail(10).sort_values("Yield", ascending=False)
            fb = px.bar(b10, x="Yield", y="Label", orientation="h",
                        template=CHART_TEMPLATE,
                        labels={"Yield": "Avg Yield (t/ha)", "Label": ""},
                        color_discrete_sequence=[SECONDARY])
            fb.update_traces(marker_color=SECONDARY,
                             hovertemplate="<b>%{y}</b><br>%{x:.3f} t/ha<extra></extra>")
            fb.update_layout(height=340, **_dist_layout)
            st.plotly_chart(fb, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── compare two states ────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Compare Two States — Yield Trend")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    state_list = sorted(dfn["State"].dropna().unique())
    c_sa, c_sb = st.columns(2)
    with c_sa:
        state_a = st.selectbox("State A", state_list,
                               index=0, key="cmp_a",
                               help="First state to compare.")
    with c_sb:
        state_b = st.selectbox("State B", state_list,
                               index=min(1, len(state_list) - 1), key="cmp_b",
                               help="Second state to compare.")

    cmp_df = dfn[dfn["State"].isin([state_a, state_b])]
    if not cmp_df.empty:
        cmp_trend = (cmp_df.groupby(["Crop_Year", "State"])["Yield"]
                     .mean().reset_index())
        fig_cmp = px.line(
            cmp_trend, x="Crop_Year", y="Yield", color="State",
            template=CHART_TEMPLATE,
            color_discrete_sequence=[PRIMARY, SECONDARY],
            labels={"Crop_Year": "Crop Year", "Yield": "Avg Yield (t/ha)"},
            markers=True,
        )
        fig_cmp.update_traces(
            hovertemplate="<b>%{fullData.name}</b> · %{x}<br>"
                          "Yield: %{y:.3f} t/ha<extra></extra>"
        )
        fig_cmp.update_layout(
            height=300,
            **_chart_layout2,
            xaxis=dict(showgrid=False, linecolor=BORDER),
            yaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.plotly_chart(empty_fig(), use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── choropleth map (with graceful fallback) ───────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("India State-level Map")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    map_metric = st.radio(
        "Map metric",
        ["Total Production (MT)", "Avg Yield (t/ha)", "Total Area (Ha)"],
        horizontal=True,
        help="Choose what each state's colour represents.",
    )

    state_map = (dfn.groupby("State")
                 .agg(
                     Total_Production=("Production", "sum"),
                     Avg_Yield=("Yield", "mean"),
                     Total_Area=("Area", "sum"),
                 )
                 .reset_index())
    col_map = {
        "Total Production (MT)": "Total_Production",
        "Avg Yield (t/ha)": "Avg_Yield",
        "Total Area (Ha)": "Total_Area",
    }[map_metric]

    with st.spinner("Loading GeoJSON…"):
        geojson = load_india_geojson()

    if geojson is not None:
        # ── normalise GeoJSON state names to match dataset ────────────────────
        # build a best-effort mapping from GeoJSON properties
        name_keys = ["NAME_1", "ST_NM", "name", "NAME", "state"]
        rename_map = {}  # geo_name -> dataset state name
        ds_states_lower = {s.lower(): s for s in dfn["State"].unique()}

        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            geo_name = None
            for k in name_keys:
                if k in props:
                    geo_name = str(props[k])
                    break
            if geo_name:
                low = geo_name.lower()
                # exact
                if low in ds_states_lower:
                    rename_map[geo_name] = ds_states_lower[low]
                else:
                    # partial: find best match
                    for ds_l, ds_orig in ds_states_lower.items():
                        if low in ds_l or ds_l in low:
                            rename_map[geo_name] = ds_orig
                            break

        # identify the property key used for state names in this GeoJSON
        prop_key = "ST_NM"
        for k in name_keys:
            if any(k in (f.get("properties") or {}) for f in geojson.get("features", [])):
                prop_key = k
                break

        # apply renames in-place to the geojson
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            old = props.get(prop_key, "")
            if old in rename_map:
                props[prop_key] = rename_map[old]

        # add feature id from the (now renamed) name
        for i, feat in enumerate(geojson.get("features", [])):
            feat["id"] = feat.get("properties", {}).get(prop_key, str(i))

        state_map["geo_id"] = state_map["State"]

        # check coverage
        matched = state_map["State"].isin(
            [f["id"] for f in geojson.get("features", [])]
        ).sum()

        if matched < 3:
            # fallback
            geojson = None
        else:
            fig_map = px.choropleth(
                state_map,
                geojson=geojson,
                locations="geo_id",
                featureidkey="id",
                color=col_map,
                hover_name="State",
                color_continuous_scale="Greens",
                template=CHART_TEMPLATE,
                labels={col_map: map_metric},
                hover_data={
                    "Total_Production": ":,.0f",
                    "Avg_Yield": ":.3f",
                    "Total_Area": ":,.0f",
                },
            )
            fig_map.update_geos(
                fitbounds="locations", visible=False
            )
            fig_map.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    f"{map_metric}: %{{z:,.2f}}<extra></extra>"
                )
            )
            fig_map.update_layout(height=520, margin=dict(t=10, b=10))
            st.plotly_chart(fig_map, use_container_width=True)

    if geojson is None:
        st.markdown(
            '<div class="inline-info">GeoJSON could not be loaded or matched — showing ranked bar chart instead.</div>',
            unsafe_allow_html=True,
        )
        state_map_s = state_map.sort_values(col_map, ascending=True)
        fig_fallback = px.bar(
            state_map_s, x=col_map, y="State", orientation="h",
            template=CHART_TEMPLATE,
            labels={col_map: map_metric, "State": ""},
            color_discrete_sequence=[PRIMARY],
        )
        fig_fallback.update_traces(
            marker_color=PRIMARY,
            hovertemplate=f"<b>%{{y}}</b><br>{map_metric}: %{{x:,.2f}}<extra></extra>",
        )
        fig_fallback.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
            margin=dict(t=8, b=8, l=0, r=0),
            xaxis=dict(gridcolor="#F0F2F0"), yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_fallback, use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – RISKS, OPPORTUNITIES & ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    page_banner("Risks & Opportunities",
                "Automated risk detection, opportunity flagging, and recommended actions.")
    active_filters_banner()

    df = excl_coconut(df_filtered)
    if check_empty(df, "filters"):
        st.stop()

    # ── RISK: 3+ consecutive yield decline ────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Risk — Sustained Yield Decline (3+ Consecutive Years)")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def find_yield_declines(_df_key: str, df_in: pd.DataFrame, min_consec: int = 3):
        results = []
        for (state, crop), grp in df_in.groupby(["State", "Crop"]):
            yearly = grp.groupby("Crop_Year")["Yield"].mean().sort_index()
            if len(yearly) < min_consec + 1:
                continue
            best = streak = 0
            for i in range(1, len(yearly)):
                if yearly.iloc[i] < yearly.iloc[i - 1]:
                    streak += 1
                    best = max(best, streak)
                else:
                    streak = 0
            if best >= min_consec:
                dec = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100
                       if yearly.iloc[0] != 0 else 0)
                severity = "🔴 High" if best >= 6 else ("🟠 Medium" if best >= 4 else "🟡 Low")
                results.append({
                    "State": state, "Crop": crop,
                    "Max Decline Streak (yrs)": best,
                    "Overall Yield Change (%)": round(dec, 1),
                    "Latest Yield (t/ha)": round(yearly.iloc[-1], 3),
                    "Severity": severity,
                })
        return (pd.DataFrame(results)
                .sort_values("Max Decline Streak (yrs)", ascending=False)
                if results else pd.DataFrame())

    with st.spinner("Scanning for yield declines…"):
        # use a hashable key to allow caching
        df_risks = find_yield_declines(
            f"{year_range}|{'|'.join(sel_states)}|{'|'.join(sel_crops)}|{'|'.join(sel_seasons)}",
            df,
        )

    if df_risks.empty:
        st.markdown(
            '<div class="inline-success">No sustained yield declines detected for current filters.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="font-size:0.84rem;color:{TEXT_DARK};margin-bottom:12px;">'
            f'<b>{len(df_risks)} state-crop pairs</b> detected with ≥3 consecutive years of yield decline.</p>',
            unsafe_allow_html=True,
        )

        # search + sort
        risk_search = st.text_input(
            "Search risks (State or Crop)",
            placeholder="e.g. Punjab, Wheat…",
            help="Filter the risk table by state or crop name.",
        )
        df_risks_show = df_risks.copy()
        if risk_search:
            mask = (df_risks_show["State"].str.contains(risk_search, case=False) |
                    df_risks_show["Crop"].str.contains(risk_search, case=False))
            df_risks_show = df_risks_show[mask]

        col_sort = st.selectbox(
            "Sort by", df_risks_show.columns.tolist(),
            index=df_risks_show.columns.tolist().index("Max Decline Streak (yrs)"),
            key="risk_sort",
        )
        df_risks_show = df_risks_show.sort_values(col_sort, ascending=False)

        # colour-coded severity column
        def colour_severity(val: str) -> str:
            if "High"   in val: return "background-color: #ffcdd2"
            if "Medium" in val: return "background-color: #fff9c4"
            if "Low"    in val: return "background-color: #f1f8e9"
            return ""

        st.dataframe(
            df_risks_show.head(50).style.map(colour_severity, subset=["Severity"]),
            use_container_width=True, height=320,
        )
        st.download_button(
            "⬇️ Download risk table (CSV)",
            data=df_risks_show.to_csv(index=False).encode(),
            file_name="crop_yield_risks.csv",
            mime="text/csv",
        )

        st.markdown(
            f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:{TEXT_MUTED};margin:16px 0 8px 0;">Risk Alerts — Top 5</p>',
            unsafe_allow_html=True,
        )
        for _, row in df_risks.head(5).iterrows():
            st.markdown(
                f'<div class="risk-card"><b>{row["Crop"]}</b> in <b>{row["State"]}</b> — '
                f'<b>{row["Max Decline Streak (yrs)"]} consecutive years</b> of decline '
                f'(overall: <b>{row["Overall Yield Change (%)"]:.1f}%</b>, '
                f'severity: {row["Severity"]}). Agronomic review advised.</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── OPPORTUNITY ───────────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Opportunity — High Yield Growth with Untapped Area")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def find_opportunities(_df_key: str, df_in: pd.DataFrame, pct: float = 0.35):
        ys = sorted(df_in["Crop_Year"].unique())
        if len(ys) < 4:
            return pd.DataFrame()
        r5 = ys[-min(5, len(ys)):]
        p5 = ys[-min(10, len(ys)):-min(5, len(ys))] or ys[:min(5, len(ys))]
        ry = df_in[df_in["Crop_Year"].isin(r5)].groupby("Crop")["Yield"].mean()
        py = df_in[df_in["Crop_Year"].isin(p5)].groupby("Crop")["Yield"].mean()
        growth = ((ry - py) / py.replace(0, np.nan) * 100).dropna()
        area_c = df_in.groupby("Crop")["Area"].sum()
        thresh = area_c.quantile(pct)
        rows = []
        for crop, g in growth.items():
            if g > 10 and area_c.get(crop, 0) < thresh:
                rows.append({
                    "Crop": crop,
                    "Yield Growth (5yr %)": round(g, 1),
                    "Total Area (Ha)": int(area_c.get(crop, 0)),
                    "Avg Recent Yield (t/ha)": round(ry.get(crop, 0), 3),
                    "Potential": "⭐⭐⭐" if g > 50 else ("⭐⭐" if g > 25 else "⭐"),
                })
        return (pd.DataFrame(rows).sort_values("Yield Growth (5yr %)", ascending=False)
                if rows else pd.DataFrame())

    with st.spinner("Scanning for opportunities…"):
        df_opps = find_opportunities(
            f"{year_range}|{'|'.join(sel_states)}|{'|'.join(sel_crops)}|{'|'.join(sel_seasons)}",
            df,
        )

    if df_opps.empty:
        st.markdown(
            '<div class="inline-info">No strong opportunity crops identified with current filters.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="font-size:0.84rem;color:{TEXT_DARK};margin-bottom:12px;">'
            f'<b>{len(df_opps)} crops</b> show >10% yield growth yet occupy &lt; 35th-percentile area.</p>',
            unsafe_allow_html=True,
        )
        opp_search = st.text_input("Search opportunities", key="opp_search",
                                   placeholder="e.g. Jowar, Summer…")
        df_opps_show = df_opps.copy()
        if opp_search:
            df_opps_show = df_opps_show[
                df_opps_show["Crop"].str.contains(opp_search, case=False)
            ]

        st.dataframe(df_opps_show.head(30), use_container_width=True, height=280)
        st.download_button(
            "⬇️ Download opportunity table (CSV)",
            data=df_opps_show.to_csv(index=False).encode(),
            file_name="crop_opportunities.csv",
            mime="text/csv",
        )

        st.markdown(
            f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:{TEXT_MUTED};margin:16px 0 8px 0;">Opportunity Highlights — Top 5</p>',
            unsafe_allow_html=True,
        )
        for _, row in df_opps.head(5).iterrows():
            st.markdown(
                f'<div class="opp-card"><b>{row["Crop"]}</b> — '
                f'+{row["Yield Growth (5yr %)"]:.1f}% yield growth over 5 years, '
                f'only {row["Total Area (Ha)"]:,} Ha cultivated. '
                f'Potential: {row["Potential"]}. Scale-up recommended.</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── RECOMMENDED ACTIONS ───────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("Recommended Actions")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    actions = []

    if not df_risks.empty:
        w = df_risks.iloc[0]
        actions.append(
            f"<b>Soil & water audit</b> — <b>{w['Crop']}</b> in <b>{w['State']}</b>: "
            f"{w['Max Decline Streak (yrs)']} consecutive years of yield decline. "
            "Investigate pest pressure, soil degradation, or irrigation failure."
        )

    if not df_opps.empty:
        b = df_opps.iloc[0]
        actions.append(
            f"<b>Scale up {b['Crop']}</b> cultivation: "
            f"+{b['Yield Growth (5yr %)']:.1f}% yield improvement in 5 years but only "
            f"{b['Total Area (Ha)']:,} Ha allocated. "
            "Policy incentives and credit support recommended."
        )

    sy = df.groupby(["State", "Crop_Year"])["Production"].sum().reset_index()
    sy_piv = sy.pivot(index="Crop_Year", columns="State", values="Production")
    if len(sy_piv) >= 3:
        chg = (sy_piv.iloc[-1] - sy_piv.iloc[-3]) / sy_piv.iloc[-3].replace(0, np.nan) * 100
        ws  = chg.idxmin()
        wc  = chg.min()
        if pd.notna(wc) and wc < 0:
            actions.append(
                f"<b>Policy focus: {ws}</b> — production down {abs(wc):.1f}% over 3 years. "
                "Evaluate irrigation, climate risk, and farmer support."
            )

    sea_yld = df.groupby("Season")["Yield"].mean()
    if not sea_yld.empty:
        ls = sea_yld.idxmin()
        actions.append(
            f"<b>Invest in {ls}-season productivity</b> "
            f"(lowest avg yield: {sea_yld.min():.3f} t/ha). "
            "Improved varieties and advisory services can raise output."
        )

    top_area_crop  = df.groupby("Crop")["Area"].sum().idxmax()
    top_yield_crop = df.groupby("Crop")["Yield"].mean().idxmax()
    if top_area_crop != top_yield_crop:
        actions.append(
            f"<b>Portfolio rebalancing</b> — <b>{top_area_crop}</b> dominates area but "
            f"<b>{top_yield_crop}</b> delivers the highest yield. "
            "Applying high-yield best practices to large-area crops can lift national output."
        )

    for a in actions:
        st.markdown(f'<div class="action-card">{a}</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── heatmap ────────────────────────────────────────────────────────────────
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    card_header("State × Season Yield Heatmap")
    st.markdown('<div class="chart-card-body">', unsafe_allow_html=True)
    piv = df.pivot_table(index="State", columns="Season", values="Yield", aggfunc="mean")
    piv = piv.loc[piv.mean(axis=1).nlargest(20).index]
    if not piv.empty:
        fig_heat = px.imshow(
            piv,
            color_continuous_scale=[[0, "#FAD2C8"], [0.5, "#74C69D"], [1, "#1B5E45"]],
            aspect="auto", template=CHART_TEMPLATE,
            labels={"color": "Avg Yield (t/ha)"},
        )
        fig_heat.update_traces(
            hovertemplate="State: <b>%{y}</b><br>Season: <b>%{x}</b><br>"
                          "Yield: %{z:.3f} t/ha<extra></extra>"
        )
        fig_heat.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
            margin=dict(t=8, b=8, l=0, r=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.plotly_chart(empty_fig(), use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – YIELD PREDICTOR (ML)
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    page_banner("Yield Predictor",
                "Predict expected yield for any State / Crop / Season / Area combination.")
    active_filters_banner()

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # always train on FULL cleaned data (not filtered) for best model
        df_model = excl_coconut(df_raw).copy()

        @st.cache_resource(show_spinner="Training Random Forest model…")
        def train_model(n_rows: int):
            d = excl_coconut(load_data()).copy()
            feats = ["State", "Crop", "Season", "Area", "Crop_Year"]
            d = d.dropna(subset=feats + ["Yield"])
            encoders = {}
            for c in ["State", "Crop", "Season"]:
                le = LabelEncoder()
                d[c + "_enc"] = le.fit_transform(d[c].astype(str))
                encoders[c] = le
            X = d[["State_enc", "Crop_enc", "Season_enc", "Area", "Crop_Year"]].values
            y = d["Yield"].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            m = RandomForestRegressor(n_estimators=120, max_depth=12,
                                      random_state=42, n_jobs=-1)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            return m, encoders, r2_score(y_te, y_pred), mean_absolute_error(y_te, y_pred)

        model, encoders, r2, mae = train_model(len(df_model))

        # ── model metrics ──────────────────────────────────────────────────────
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        card_header("Model Performance — held-out 20% test set")
        st.markdown('<div class="chart-card-body" style="padding:16px 20px;">', unsafe_allow_html=True)
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("R² Score", f"{r2:.4f}",
                   help="1.0 = perfect; fraction of variance explained by the model.")
        mm2.metric("Mean Absolute Error", f"{mae:.4f} t/ha",
                   help="Average absolute difference between predicted and actual yield.")
        mm3.metric("Training samples", fmt_num(len(df_model) * 0.8, "rows"),
                   help="80% of cleaned data used to train the Random Forest.")

        st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        card_header("Make a Prediction")
        st.markdown('<div class="chart-card-body" style="padding:16px 20px;">', unsafe_allow_html=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            pred_state  = st.selectbox("State",  sorted(df_model["State"].dropna().unique()),
                                       help="Select the state for prediction.")
            pred_crop   = st.selectbox("Crop",   sorted(df_model["Crop"].dropna().unique()),
                                       help="Select the crop.")
        with pc2:
            pred_season = st.selectbox("Season", sorted(df_model["Season"].dropna().unique()),
                                       help="Select the growing season.")
            pred_area   = st.number_input(
                "Area (Ha)", min_value=1.0, value=100.0, step=10.0,
                help="Area under cultivation in hectares.",
            )
            pred_year   = st.number_input(
                "Crop Year",
                min_value=int(df_model["Crop_Year"].min()),
                max_value=int(df_model["Crop_Year"].max()) + 5,
                value=int(df_model["Crop_Year"].max()),
                help="Year of harvest (can be a future year for forecasting).",
            )

        def encode_safe(le: LabelEncoder, val: str) -> int:
            cls = list(le.classes_)
            return le.transform([val])[0] if val in cls else \
                le.transform([min(cls, key=lambda x: abs(ord(x[0]) - ord(val[0])))])[0]

        if st.button("Predict Yield", type="primary", use_container_width=True):
            with st.spinner("Predicting…"):
                X_in = np.array([[
                    encode_safe(encoders["State"],  pred_state),
                    encode_safe(encoders["Crop"],   pred_crop),
                    encode_safe(encoders["Season"], pred_season),
                    pred_area,
                    pred_year,
                ]])
                pred_yield = model.predict(X_in)[0]
                pred_prod  = pred_yield * pred_area

                # historical average for same crop + state
                hist = df_model[
                    (df_model["State"] == pred_state) &
                    (df_model["Crop"]  == pred_crop)
                ]["Yield"].mean()
                delta_vs_hist = pred_yield - hist if not np.isnan(hist) else None

            st.markdown(
                f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_MUTED};margin:20px 0 10px 0;">Prediction Results</p>',
                unsafe_allow_html=True,
            )
            r1, r2_col, r3 = st.columns(3)
            r1.metric(
                "Predicted Yield",
                f"{pred_yield:.4f} t/ha",
                delta=f"{delta_vs_hist:+.4f} vs hist. avg" if delta_vs_hist is not None else None,
                help="Model prediction for yield (production ÷ area).",
            )
            r2_col.metric(
                "Predicted Production",
                fmt_num(pred_prod, "MT"),
                help=f"Predicted yield × {pred_area:,.0f} Ha.",
            )
            r3.metric(
                "Historical Avg Yield",
                f"{hist:.4f} t/ha" if not np.isnan(hist) else "N/A",
                help=f"Mean yield for {pred_crop} in {pred_state} across all years in dataset.",
            )

            # ── feature importance ─────────────────────────────────────────────
            st.markdown(
                f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:{TEXT_MUTED};margin:20px 0 8px 0;">Feature Importances</p>',
                unsafe_allow_html=True,
            )
            imp_df = pd.DataFrame({
                "Feature": ["State", "Crop", "Season", "Area", "Crop Year"],
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                template=CHART_TEMPLATE,
                labels={"Importance": "Importance Score", "Feature": ""},
                color_discrete_sequence=[PRIMARY],
            )
            fig_imp.update_traces(
                marker_color=PRIMARY,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            )
            fig_imp.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
                margin=dict(t=4, b=4, l=0, r=0),
                xaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # ── historical vs predicted sparkline ─────────────────────────────
            hist_trend = (
                df_model[
                    (df_model["State"] == pred_state) &
                    (df_model["Crop"]  == pred_crop)
                ]
                .groupby("Crop_Year")["Yield"].mean()
                .reset_index()
            )
            if not hist_trend.empty:
                st.markdown(
                    f'<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;'
                    f'text-transform:uppercase;color:{TEXT_MUTED};margin:20px 0 8px 0;">'
                    f'Historical Yield — {pred_crop} in {pred_state}</p>',
                    unsafe_allow_html=True,
                )
                pred_point = pd.DataFrame([{"Crop_Year": pred_year, "Yield": pred_yield}])
                fig_hs = px.line(
                    hist_trend, x="Crop_Year", y="Yield",
                    template=CHART_TEMPLATE,
                    color_discrete_sequence=[PRIMARY],
                    labels={"Crop_Year": "Year", "Yield": "Avg Yield (t/ha)"},
                )
                fig_hs.update_traces(
                    hovertemplate="Year: %{x}<br>Yield: %{y:.4f} t/ha<extra></extra>"
                )
                fig_hs.add_scatter(
                    x=pred_point["Crop_Year"], y=pred_point["Yield"],
                    mode="markers",
                    marker=dict(color=SECONDARY, size=14, symbol="star"),
                    name=f"Predicted ({pred_year})",
                    hovertemplate=f"Predicted {pred_year}: %{{y:.4f}} t/ha<extra></extra>",
                )
                fig_hs.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
                    margin=dict(t=4, b=4, l=0, r=0),
                    xaxis=dict(showgrid=False, linecolor=BORDER),
                    yaxis=dict(gridcolor="#F0F2F0", linecolor="rgba(0,0,0,0)"),
                    legend=dict(font=dict(size=11)),
                )
                st.plotly_chart(fig_hs, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    except ImportError:
        st.markdown(
            '<div class="inline-warn">scikit-learn is not installed. '
            'Run <code>pip install scikit-learn</code> and restart to use the predictor.</div>',
            unsafe_allow_html=True,
        )


# ── Material Symbols font (for icon badges) ────────────────────────────────────
st.markdown(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />',
    unsafe_allow_html=True,
)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="app-footer">
    Data source: APY.csv &nbsp;·&nbsp; India Agriculture &nbsp;|&nbsp;
    Built with Python, Streamlit &amp; Plotly
    </div>""",
    unsafe_allow_html=True,
)
