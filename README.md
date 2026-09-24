# 🌾 Indian Crop Production Analysis — BI Dashboard

A multi-page, interactive Business Intelligence dashboard built with **Streamlit** and **Plotly**,
analysing India's agricultural crop production data across states, districts, seasons, and years.

---

## Features

### 🏠 Page 1 — Executive Overview
- Green-gradient page header banner with dataset metadata (year range, state count, crop count) computed from data
- KPI cards using `st.metric` with **delta values** (latest year vs prior year) for Total Production, Average Yield, 5-Year Yield Growth, and Top Crop Share
- Large numbers formatted as `1.2 Mn MT`, `45.3 K Ha` for readability
- National production **area chart** by year with formatted hover tooltips
- **Animated bar chart race** — top 10 states by production over years (Plotly `animation_frame`), with Play/Pause controls
- **Top 10 states** ranked by total production and by average yield (side-by-side)
- Interactive **treemap** (State → Crop) allowing click-through drill-down into crop composition

### 🌱 Page 2 — Crop, Season & Region Analysis
- **Season-wise comparison**: dual-axis bar + line (total production vs avg yield), hover tooltips with units
- **Yield trend per crop** over years for top 10 crops (or selected crops)
- **District drill-down**: select crop and optionally filter to a state; top/bottom 10 districts ranked by yield with district + state labels
- **Compare two states**: side-by-side yield trend chart with markers
- **India choropleth map**: state-level production, yield, or area, loading a public India states GeoJSON; gracefully falls back to a ranked bar chart if GeoJSON fails to load or state names cannot be matched

### ⚠️ Page 3 — Risks, Opportunities & Actions
- **Risk detector**: state-crop pairs with 3+ consecutive years of yield decline; severity colour-coded (High/Medium/Low) with styled `st.dataframe`
- **Searchable + sortable risk table** with `st.text_input` and `st.selectbox` sort control
- **Download button** (CSV) for risk and opportunity tables
- **Opportunity detector**: crops with >10% 5-year yield growth but below 35th-percentile area, with star-rated potential
- **5 categories of plain-English recommended actions** generated from the data
- **State × Season yield heatmap** with tooltip units

### 🤖 Page 4 — Yield Predictor (ML)
- `RandomForestRegressor` (120 trees, depth 12) trained on full cleaned dataset; cached with `@st.cache_resource`
- Model metrics: R² and MAE displayed as `st.metric` cards
- Prediction form: State, Crop, Season, Area (Ha), Crop Year
- Prediction output as `st.metric` with **delta vs historical average** for same crop/state
- **Feature importance** horizontal bar chart
- **Historical yield sparkline** with predicted point plotted as an orange star

### 🔍 Global Sidebar Filters (all pages)
- **Crop Year range slider** filtering all pages
- **State, Crop, Season multiselects** with `help=` tooltips
- **"↺ Reset filters" button**
- **Active filters badges** displayed at the top of every page
- **⬇️ Download filtered dataset** button on every page

### 🎨 Design
- `.streamlit/config.toml` with deep-green agriculture theme
- Consistent `plotly_white` chart template across all charts
- `PALETTE` colour list used across all bar/line charts
- `fmt_num()` helper for human-readable large numbers
- `empty_fig()` fallback for charts with no data
- Friendly `st.warning` shown instead of errors when filters produce no data
- Footer: *Data source: APY.csv — India Agriculture | Built with Python, Streamlit & Plotly*

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the dataset

Ensure `APY.csv` is in the same directory as `app.py`.

### 3. Launch the dashboard

```bash
streamlit run AatmajaMhatre_CropProductionAnalysis.py
```

The app will open automatically at `http://localhost:8501`.

---

## Project Structure

```
crop production/
├── app.py                             # All-in-one Streamlit dashboard (v2)
├── APY.csv                            # Dataset (see Dataset section)
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── Indian_Crop_Production_Report.docx # Project report
└── .streamlit/
    └── config.toml                    # Streamlit theme (deep green)
```

---

## Dataset

| Field | Details |
|-------|---------|
| **Name** | Area, Production, Yield (APY) — Indian Agriculture |
| **Source** | [Add dataset link here] |
| **Rows** | ~246,091 |
| **Columns** | State, District, Crop, Crop_Year, Season, Area, Production, Yield |
| **Coverage** | Multiple Indian states, years ~2000–2020, all major crops and seasons |

> **Note on Coconut**: Coconut production is measured in *nuts*, not metric tonnes.
> The dashboard excludes it from all tonnage-based KPIs and aggregations to avoid
> misleading comparisons, but it remains available in crop-level filtering.

---

## Data Cleaning (applied at load time via `@st.cache_data`)

- Strip whitespace from all column names and string values
- Drop rows where `Production` is missing or `Area ≤ 0`
- Recompute `Yield = Production / Area`
- Exclude Coconut from tonnage totals

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `plotly` | Interactive charts (bar, area, scatter, treemap, choropleth, heatmap, animation) |
| `scikit-learn` | Random Forest yield predictor |
| `requests` | Fetch India GeoJSON for choropleth map |

---

## License

For educational and analytical use. Refer to the dataset source for data licensing terms.
