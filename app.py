"""
app.py
======
AI Air Quality Monitoring & PM2.5 Prediction
Streamlit Application — 6-page interactive dashboard

Run with:
    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import (
    load_processed, run_preprocessing, POLLUTANT_COLS,
    TARGET, pm25_category, inspect_all_files,
)
from src.feature_engineering import (
    run_feature_engineering, build_single_row, SEASON_MAP,
)
from src.evaluate import compare_models
from src.explain import get_feature_importance, shap_explanation, shap_summary_df, SHAP_AVAILABLE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Air Quality Monitoring",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — polished card/badge styles, no animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── KPI cards ──────────────────────────────────────────────────────────── */
.kpi-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 12px 14px;
    text-align: center;
}
.kpi-icon  { font-size: 1.5rem; line-height: 1; margin-bottom: 4px; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #1f2328; line-height: 1.2; }
.kpi-label { font-size: 0.78rem; color: #57606a; margin-top: 4px; letter-spacing: 0.02em; }

/* ── Prediction result box ──────────────────────────────────────────────── */
.pred-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e8f4fd 100%);
    border: 2px solid #3b82d4;
    border-radius: 14px;
    padding: 28px 24px 22px;
    text-align: center;
}
.pred-icon  { font-size: 2rem; margin-bottom: 8px; }
.pred-label { font-size: 0.9rem; color: #57606a; font-weight: 500; margin-bottom: 4px; }
.pred-value { font-size: 3.2rem; font-weight: 800; color: #1f2328; line-height: 1.1; }
.pred-unit  { font-size: 1.1rem; color: #57606a; margin-top: 2px; }
.pred-city  { font-size: 0.85rem; color: #57606a; margin-top: 10px; }

/* ── Pollution category badge ───────────────────────────────────────────── */
.cat-badge {
    display: inline-block;
    padding: 5px 20px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #fff;
    margin-top: 10px;
    letter-spacing: 0.02em;
}

/* ── Section subtitle ───────────────────────────────────────────────────── */
.page-subtitle {
    color: #57606a;
    font-size: 0.95rem;
    margin-top: -10px;
    margin-bottom: 16px;
}

/* ── SDG pill ───────────────────────────────────────────────────────────── */
.sdg-pill {
    display: inline-block;
    background: #e8f5e9;
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    color: #2e7d32;
    font-weight: 600;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: page header with subtitle
# ---------------------------------------------------------------------------
def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f"## {icon} {title}")
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    """Load or regenerate processed data."""
    return load_processed()


@st.cache_resource(show_spinner=False)
def get_artefact():
    """Load trained model artefact."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_pm25_model.joblib")
    if not os.path.exists(model_path):
        st.error("Model not found. Please run: python -m src.train")
        st.stop()
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def get_file_summaries():
    return inspect_all_files()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🌍 Overview",
    "📊 Air Quality Analytics",
    "🔮 PM2.5 Prediction",
    "📈 Next-Day Forecast",
    "🤖 Model Performance",
    "🧹 Data Quality",
]

with st.sidebar:
    st.markdown("### 🌬️ AI Air Quality")
    st.markdown("**Monitoring & PM2.5 Prediction**")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        '<span class="sdg-pill">🌱 SDG 11</span>'
        '<span class="sdg-pill">🌱 SDG 13</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "🇮🇳 India Air Quality 2015–2020  \n"
        "🧠 XGBoost PM2.5 Regression  \n"
        "📍 26 cities · 24,933 records",
    )

# Load data & model
df = get_data()
art = get_artefact()
pipeline    = art["pipeline"]
feat_cols   = art["feat_cols"]
encoders    = art["encoders"]
all_metrics = art["all_metrics"]
best_name   = art["model_name"]
best_metrics = art["metrics"]


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "🌍 Overview":
    page_header(
        "🌍", "AI Air Quality Monitoring & PM2.5 Prediction",
        "Understanding pollution patterns across 26 Indian cities · 2015–2020"
    )

    # SDG alignment strip
    st.markdown(
        '<span class="sdg-pill">🌱 SDG 11 — Sustainable Cities</span>'
        '<span class="sdg-pill">🌱 SDG 13 — Climate Action</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This system analyses historical air-quality data from 26 Indian cities "
        "and provides XGBoost-based PM2.5 concentration predictions. "
        "Data: [Air Quality Data in India — Kaggle]"
        "(https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)"
    )
    st.markdown("---")

    # KPI cards
    kpis = [
        ("📋", f"{len(df):,}",                          "Records"),
        ("📍", str(df["City"].nunique()),                "Cities"),
        ("📅", f"{df['Date'].min().year}–{df['Date'].max().year}", "Date Range"),
        ("💨", f"{df['PM2.5'].mean():.1f} µg/m³",       "Avg PM2.5"),
        ("⚠️", f"{df['PM2.5'].max():.0f} µg/m³",        "Max PM2.5"),
        ("🎯", f"{best_metrics['R2']:.4f}",              "Best Model R²"),
    ]
    cols = st.columns(6)
    for col, (icon, value, label) in zip(cols, kpis):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Best model strip
    st.markdown("#### 🤖 Best Model Summary")
    m_cols = st.columns(4)
    m_cols[0].metric("Model", best_name)
    m_cols[1].metric("MAE",  f"{best_metrics['MAE']:.2f}",  help="Mean Absolute Error (µg/m³)")
    m_cols[2].metric("RMSE", f"{best_metrics['RMSE']:.2f}", help="Root Mean Squared Error (µg/m³)")
    m_cols[3].metric("R²",   f"{best_metrics['R2']:.4f}",   help="Coefficient of determination")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📈 PM2.5 Trend Over Time")
        monthly = (
            df.set_index("Date")
            .resample("MS")["PM2.5"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            monthly, x="Date", y="PM2.5",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)"},
            template="plotly_white",
        )
        fig.update_traces(line_color="#3b82d4", line_width=2)
        fig.add_hline(
            y=60, line_dash="dash", line_color="orange",
            annotation_text="WHO guideline (60 µg/m³)",
            annotation_position="top left",
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🏙️ Average PM2.5 by City")
        city_avg = df.groupby("City")["PM2.5"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            city_avg, x="PM2.5", y="City", orientation="h",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)", "City": ""},
            template="plotly_white",
            color="PM2.5",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### 📊 PM2.5 Distribution")
        fig = px.histogram(
            df, x="PM2.5", nbins=60,
            labels={"PM2.5": "PM2.5 (µg/m³)"},
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("#### 🌦️ Average PM2.5 by Season")
        df["season"] = df["Date"].dt.month.map(SEASON_MAP)
        season_avg = df.groupby("season")["PM2.5"].mean().reset_index()
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
        season_avg["season"] = pd.Categorical(
            season_avg["season"], categories=season_order, ordered=True
        )
        season_avg = season_avg.sort_values("season")
        fig = px.bar(
            season_avg, x="season", y="PM2.5",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)", "season": "Season"},
            template="plotly_white",
            color="PM2.5",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

    # SDG impact section
    st.markdown("---")
    st.markdown("#### 🌱 Sustainability Impact")
    sdg_col1, sdg_col2 = st.columns(2)
    with sdg_col1:
        st.info(
            "**SDG 11 — Sustainable Cities and Communities**  \n"
            "Monitoring PM2.5 across 26 Indian cities helps identify pollution hotspots, "
            "supports evidence-based urban planning, and raises public awareness of air quality risks."
        )
    with sdg_col2:
        st.info(
            "**SDG 13 — Climate Action**  \n"
            "PM2.5 and co-pollutants (NO₂, SO₂, CO) are closely linked to fossil fuel combustion. "
            "Reducing these pollutants delivers co-benefits for both public health and climate goals."
        )

    st.caption(
        "⚠️ Disclaimer: Predictions are estimates based on historical patterns (2015–2020). "
        "This is an academic demonstration — not a real-time monitoring system."
    )


# ===========================================================================
# PAGE 2 — AIR QUALITY ANALYTICS
# ===========================================================================
elif page == "📊 Air Quality Analytics":
    page_header(
        "📊", "Air Quality Analytics",
        "Explore pollution patterns, trends, and correlations across Indian cities"
    )

    df["season"]     = df["Date"].dt.month.map(SEASON_MAP)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"]      = df["Date"].dt.month
    df["year"]       = df["Date"].dt.year

    # Sidebar filters
    cities = sorted(df["City"].unique().tolist())
    sel_cities = st.sidebar.multiselect("📍 Filter Cities", cities, default=cities[:5])
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    import datetime
    d_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        d_start, d_end = pd.Timestamp(d_range[0]), pd.Timestamp(d_range[1])
    else:
        d_start, d_end = pd.Timestamp(date_min), pd.Timestamp(date_max)

    pollutant_opts = ["PM2.5"] + [c for c in POLLUTANT_COLS if c in df.columns]
    sel_pollutant = st.sidebar.selectbox("💨 Pollutant Focus", pollutant_opts)

    df_filt = df[
        (df["City"].isin(sel_cities if sel_cities else cities)) &
        (df["Date"] >= d_start) &
        (df["Date"] <= d_end)
    ].copy()

    if df_filt.empty:
        st.warning("⚠️ No data matches the current filter. Adjust the sidebar filters.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏙️ City Comparison",
        "📈 Trends",
        "🌦️ Seasonal & Calendar",
        "🔬 Correlations",
        "🔍 Selected City",
    ])

    # ---- Tab 1: City Comparison ----
    with tab1:
        st.markdown("#### 🏙️ PM2.5 City Comparison")
        col_l, col_r = st.columns(2)
        with col_l:
            top_n = st.slider("Top N cities by avg PM2.5", 5, 26, 10)
            top_cities = (
                df_filt.groupby("City")["PM2.5"].mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )
            fig = px.bar(
                top_cities, x="City", y="PM2.5",
                color="PM2.5", color_continuous_scale="RdYlGn_r",
                template="plotly_white",
                labels={"PM2.5": "Avg PM2.5 (µg/m³)"},
                title=f"Top {top_n} Cities by Avg PM2.5",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**PM2.5 spread by city (box plot)**")
            fig = px.box(
                df_filt.sort_values("City"), x="City", y="PM2.5",
                template="plotly_white",
                color="City",
                title="PM2.5 Distribution per City",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"#### 📈 {sel_pollutant} — Time Series by City")
        city_ts = df_filt.groupby(["Date", "City"])[sel_pollutant].mean().reset_index()
        fig = px.line(
            city_ts, x="Date", y=sel_pollutant, color="City",
            template="plotly_white",
            labels={sel_pollutant: f"{sel_pollutant} (µg/m³)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Trends ----
    with tab2:
        st.markdown(f"#### 📈 Monthly {sel_pollutant} Trend")
        monthly = (
            df_filt.set_index("Date")
            .resample("MS")[sel_pollutant]
            .mean()
            .reset_index()
        )
        fig = px.line(
            monthly, x="Date", y=sel_pollutant,
            template="plotly_white",
            labels={sel_pollutant: f"Avg {sel_pollutant} (µg/m³)"},
        )
        fig.update_traces(line_color="#3b82d4", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Yearly Average PM2.5")
        yearly = df_filt.groupby("year")["PM2.5"].mean().reset_index()
        fig = px.bar(
            yearly, x="year", y="PM2.5",
            template="plotly_white",
            color="PM2.5", color_continuous_scale="RdYlGn_r",
            labels={"year": "Year", "PM2.5": "Avg PM2.5 (µg/m³)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: Seasonal & Calendar ----
    with tab3:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 📅 Average PM2.5 by Month")
            monthly_avg = df_filt.groupby("month")["PM2.5"].mean().reset_index()
            month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly_avg["month_name"] = monthly_avg["month"].apply(lambda x: month_names[x-1])
            fig = px.bar(
                monthly_avg, x="month_name", y="PM2.5",
                color="PM2.5", color_continuous_scale="RdYlGn_r",
                template="plotly_white",
                labels={"PM2.5": "Avg PM2.5 (µg/m³)", "month_name": "Month"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("#### 📆 PM2.5 by Day of Week")
            dow_avg = df_filt.groupby("day_of_week")["PM2.5"].mean().reset_index()
            dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dow_avg["day_name"] = dow_avg["day_of_week"].apply(lambda x: dow_names[x])
            fig = px.bar(
                dow_avg, x="day_name", y="PM2.5",
                color="PM2.5", color_continuous_scale="RdYlGn_r",
                template="plotly_white",
                labels={"PM2.5": "Avg PM2.5 (µg/m³)", "day_name": "Day"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🌦️ Seasonal PM2.5 Variation")
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
        season_avg = df_filt.groupby("season")["PM2.5"].mean().reset_index()
        season_avg["season"] = pd.Categorical(
            season_avg["season"], categories=season_order, ordered=True
        )
        season_avg = season_avg.sort_values("season")
        fig = px.bar(
            season_avg, x="season", y="PM2.5",
            template="plotly_white",
            color="PM2.5", color_continuous_scale="RdYlGn_r",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)", "season": "Season"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 4: Correlations ----
    with tab4:
        st.markdown("#### 🔬 Pollutant Correlation Heatmap")
        corr_cols = [c for c in ["PM2.5"] + POLLUTANT_COLS if c in df_filt.columns]
        corr = df_filt[corr_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        ))
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 PM2.5 vs Key Pollutants")
        pair_plots = [("PM10", "PM2.5"), ("NO2", "PM2.5"), ("CO", "PM2.5"), ("SO2", "PM2.5")]
        cols = st.columns(2)
        for i, (x_col, y_col) in enumerate(pair_plots):
            if x_col in df_filt.columns:
                sample = df_filt.sample(min(2000, len(df_filt)), random_state=42)
                fig = px.scatter(
                    sample,
                    x=x_col, y=y_col,
                    color="City", opacity=0.5,
                    template="plotly_white",
                    title=f"{y_col} vs {x_col}",
                    labels={x_col: f"{x_col} (µg/m³)", y_col: f"{y_col} (µg/m³)"},
                )
                # Binned mean trend — no statsmodels required
                _s = sample[[x_col, y_col]].dropna().sort_values(x_col)
                if len(_s) > 10:
                    _bins = np.array_split(_s, min(30, len(_s) // 20 or 1))
                    _tx = [float(b[x_col].mean()) for b in _bins if len(b) > 0]
                    _ty = [float(b[y_col].mean()) for b in _bins if len(b) > 0]
                    fig.add_trace(go.Scatter(
                        x=_tx, y=_ty,
                        mode="lines", name="Trend (mean)",
                        line=dict(color="#1f2328", width=2, dash="dot"),
                        showlegend=False,
                    ))
                cols[i % 2].plotly_chart(fig, use_container_width=True)

    # ---- Tab 5: Selected City ----
    with tab5:
        sel_city = st.selectbox("🏙️ Select City", sorted(df["City"].unique()))
        df_city = df[df["City"] == sel_city].sort_values("Date")

        st.markdown(f"#### 📈 Monthly PM2.5 Trend — {sel_city}")
        monthly_city = df_city.set_index("Date").resample("MS")["PM2.5"].mean().reset_index()
        fig = px.line(
            monthly_city, x="Date", y="PM2.5",
            template="plotly_white",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)"},
        )
        fig.update_traces(line_color="#3b82d4", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"#### 🧪 Multi-Pollutant Comparison — {sel_city}")
        available_pols = [
            c for c in POLLUTANT_COLS
            if c in df_city.columns and df_city[c].notna().sum() > 10
        ]
        sel_pols = st.multiselect("Pollutants", available_pols, default=available_pols[:3])
        if sel_pols:
            plot_df = df_city.set_index("Date")[sel_pols].resample("MS").mean().reset_index()
            fig = px.line(
                plot_df.melt(id_vars="Date", var_name="Pollutant", value_name="Concentration"),
                x="Date", y="Concentration", color="Pollutant",
                template="plotly_white",
                labels={"Concentration": "Concentration (µg/m³)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"#### 📋 Data Summary — {sel_city}")
        st.dataframe(df_city.describe().round(2), use_container_width=True)


# ===========================================================================
# PAGE 3 — PM2.5 PREDICTION
# ===========================================================================
elif page == "🔮 PM2.5 Prediction":
    page_header(
        "🔮", "PM2.5 Prediction",
        "Estimate particulate pollution from historical air-quality conditions"
    )

    st.info(
        "🔒 **Leakage-free inputs:** AQI and AQI_Bucket are **not** used — "
        "they are derived from PM2.5 and would trivially leak the answer into the model."
    )

    import datetime
    cities = sorted(df["City"].unique().tolist())

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown("#### 🗂️ Input Features")
        sel_city = st.selectbox("📍 City", cities, key="pred_city")
        sel_date = st.date_input(
            "📅 Date",
            value=datetime.date(2020, 1, 15),
            min_value=datetime.date(2015, 1, 1),
            max_value=datetime.date(2025, 12, 31),
        )

        st.markdown("**💨 Pollutant Readings** *(leave at city median if unknown)*")
        pol_cols_available = [c for c in POLLUTANT_COLS if c in df.columns]
        city_medians = df[df["City"] == sel_city][pol_cols_available].median().fillna(0)

        pol_vals = {}
        c1, c2 = st.columns(2)
        for i, col_name in enumerate(pol_cols_available):
            default_val = float(city_medians.get(col_name, 0.0))
            widget = c1 if i % 2 == 0 else c2
            pol_vals[col_name] = widget.number_input(
                f"{col_name} (µg/m³)",
                min_value=0.0,
                max_value=5000.0,
                value=round(default_val, 2),
                step=0.1,
                key=f"pol_{col_name}",
            )

        st.markdown("**🕐 Recent PM2.5 History** *(optional — improves accuracy)*")
        hc1, hc2, hc3 = st.columns(3)
        _pm_default = float(city_medians.get("PM2.5", 50.0)) if "PM2.5" in city_medians else 50.0
        d7 = hc1.number_input("7 days ago", 0.0, 999.0, _pm_default, step=0.1, key="h7")
        d3 = hc2.number_input("3 days ago", 0.0, 999.0, _pm_default, step=0.1, key="h3")
        d1 = hc3.number_input("Yesterday",  0.0, 999.0, _pm_default, step=0.1, key="h1")
        hist_vals = [d7, 0, 0, 0, d3, 0, d1]

        predict_btn = st.button("🔮 Predict PM2.5", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### 📊 Prediction Result")
        if predict_btn:
            try:
                row_df = build_single_row(
                    city=sel_city,
                    date=pd.Timestamp(sel_date),
                    pollutant_values=pol_vals,
                    historical_pm25=hist_vals,
                    encoders=encoders,
                    feat_cols=feat_cols,
                )
                pred_pm25  = max(0.0, float(pipeline.predict(row_df)[0]))
                cat_label, cat_colour = pm25_category(pred_pm25)

                st.markdown(
                    f'<div class="pred-box">'
                    f'<div class="pred-icon">🔮</div>'
                    f'<div class="pred-label">Predicted PM2.5</div>'
                    f'<div class="pred-value">{pred_pm25:.2f}</div>'
                    f'<div class="pred-unit">µg/m³</div>'
                    f'<div><span class="cat-badge" style="background:{cat_colour}">'
                    f'{cat_label}</span></div>'
                    f'<div class="pred-city">📍 {sel_city} · 📅 {sel_date}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("")
                st.markdown(
                    "**Indian CPCB PM2.5 categories:**  \n"
                    "🟢 Good (0–30) &nbsp;|&nbsp; 🟡 Satisfactory (30–60) &nbsp;|&nbsp; "
                    "🟠 Moderately Polluted (60–90) &nbsp;|&nbsp; 🔴 Poor (90–120)  \n"
                    "🔴 Very Poor (120–250) &nbsp;|&nbsp; ⚫ Severe (>250)"
                )
                st.caption(
                    "ℹ️ PM2.5 concentration ≠ AQI. AQI is computed from multiple pollutants "
                    "using CPCB sub-index formulas and is not predicted here."
                )

                st.markdown("---")
                st.markdown("#### 🔍 Top Contributing Features")
                df_fi = get_feature_importance(pipeline, feat_cols)
                fig = px.bar(
                    df_fi.head(10), x="importance", y="feature", orientation="h",
                    template="plotly_white",
                    labels={"importance": "Importance", "feature": "Feature"},
                    color="importance", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(t=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Feature importance reflects predictive power — it does NOT imply causation."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.markdown(
                "👈 Fill in the form and click **🔮 Predict PM2.5** to see the result."
            )
            st.markdown("---")
            st.markdown("#### 🔍 Model Feature Importance")
            df_fi = get_feature_importance(pipeline, feat_cols)
            fig = px.bar(
                df_fi.head(12), x="importance", y="feature", orientation="h",
                template="plotly_white",
                labels={"importance": "Importance", "feature": "Feature"},
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(t=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The model's top predictors are recent PM2.5 lag features — "
                "reflecting strong temporal autocorrelation in air quality."
            )


# ===========================================================================
# PAGE 4 — NEXT-DAY FORECAST
# ===========================================================================
elif page == "📈 Next-Day Forecast":
    page_header(
        "📈", "Next-Day PM2.5 Forecast",
        "Time-aware prediction of upcoming PM2.5 levels using lag & rolling features"
    )

    st.info(
        "ℹ️ This forecast uses only information available **before** the prediction date: "
        "historical PM2.5 lags, rolling averages, city identity, and calendar features. "
        "Pollutant co-readings for future dates are approximated with city-level medians."
    )

    cities = sorted(df["City"].unique().tolist())
    import datetime
    sel_city_fc   = st.sidebar.selectbox("📍 City (Forecast)", cities, key="fc_city")
    forecast_days = st.sidebar.slider("📅 Days to Forecast", 1, 14, 7)

    st.markdown(f"#### 📊 Recent Historical PM2.5 — {sel_city_fc}")
    df_city = df[df["City"] == sel_city_fc].sort_values("Date").tail(90)
    fig = px.line(
        df_city, x="Date", y="PM2.5",
        template="plotly_white",
        labels={"PM2.5": "PM2.5 (µg/m³)"},
    )
    fig.update_traces(line_color="#3b82d4", line_width=2)
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("📈 Generate Forecast", type="primary"):
        city_medians      = df[df["City"] == sel_city_fc][POLLUTANT_COLS].median().fillna(0)
        pol_cols_available = [c for c in POLLUTANT_COLS if c in df.columns]
        pol_vals           = {c: float(city_medians.get(c, 0.0)) for c in pol_cols_available}

        last_known   = df_city["PM2.5"].tail(7).tolist()
        last_date    = df_city["Date"].max()
        forecast_rows = []
        rolling_hist = last_known.copy()

        with st.spinner("Computing forecast…"):
            for i in range(1, forecast_days + 1):
                fc_date = last_date + pd.Timedelta(days=i)
                row_df  = build_single_row(
                    city=sel_city_fc,
                    date=fc_date,
                    pollutant_values=pol_vals,
                    historical_pm25=rolling_hist,
                    encoders=encoders,
                    feat_cols=feat_cols,
                )
                pred = max(0.0, float(pipeline.predict(row_df)[0]))
                cat_label, _ = pm25_category(pred)
                forecast_rows.append({
                    "Date": fc_date.date(),
                    "Forecast PM2.5": round(pred, 2),
                    "Category": cat_label,
                    "Type": "Forecast",
                })
                rolling_hist.append(pred)
                rolling_hist = rolling_hist[-7:]

        df_fc = pd.DataFrame(forecast_rows)

        st.markdown("#### 📈 Forecast Results")

        hist_plot = df_city[["Date", "PM2.5"]].tail(30).copy()
        hist_plot["Type"] = "Historical"
        hist_plot.rename(columns={"PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
        fc_plot = df_fc.copy()
        fc_plot["Date"] = pd.to_datetime(fc_plot["Date"])
        fc_plot.rename(columns={"Forecast PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
        combined = pd.concat([
            hist_plot[["Date", "PM2.5 (µg/m³)", "Type"]],
            fc_plot[["Date",   "PM2.5 (µg/m³)", "Type"]],
        ], ignore_index=True)

        fig = px.line(
            combined, x="Date", y="PM2.5 (µg/m³)", color="Type",
            color_discrete_map={"Historical": "#3b82d4", "Forecast": "#dc2626"},
            template="plotly_white",
        )
        fig.add_vline(
            x=last_date, line_dash="dash", line_color="#57606a",
            annotation_text="Forecast start", annotation_position="top right",
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Forecast Table")
        st.dataframe(df_fc, use_container_width=True)
        st.caption(
            "⚠️ Forecasts are model predictions, not measurements. "
            "Uncertainty grows with forecast horizon as errors compound through the lag features."
        )


# ===========================================================================
# PAGE 5 — MODEL PERFORMANCE
# ===========================================================================
elif page == "🤖 Model Performance":
    page_header(
        "🤖", "Model Performance",
        "Evaluate and understand the machine-learning models trained for PM2.5 prediction"
    )

    # Model comparison table
    st.markdown("#### 📊 Model Comparison")
    df_cmp = compare_models(all_metrics)
    st.dataframe(
        df_cmp.style
        .highlight_min(subset=["MAE", "RMSE"], color="#d1fae5")
        .highlight_max(subset=["R²"],           color="#d1fae5"),
        use_container_width=True,
    )

    # Grouped bar chart
    fig = go.Figure()
    palette = ["#3b82d4", "#7c5cd8", "#059669", "#d97706"]
    for j, (_, row) in enumerate(df_cmp.iterrows()):
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=["MAE", "RMSE"],
            y=[row["MAE"], row["RMSE"]],
            marker_color=palette[j % len(palette)],
        ))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title="MAE & RMSE Comparison Across Models",
        margin=dict(t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Split info
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Best Model", best_name)
    col_info2.metric(
        "Train period",
        f"{art['train_date_range'][0][:10]}  →  {art['train_date_range'][1][:10]}",
    )
    col_info3.metric(
        "Test period",
        f"{art['test_date_range'][0][:10]}  →  {art['test_date_range'][1][:10]}",
    )
    st.caption(
        "🕐 Chronological 80/20 split — earlier dates train, later dates test. "
        "No future information leaks into training."
    )

    st.markdown("---")

    # Actual vs Predicted
    y_test     = np.array(art["y_test"])
    y_pred     = np.array(art["y_pred_best"])
    test_dates  = art["test_dates"]
    test_cities = art["test_cities"]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🎯 Actual vs Predicted PM2.5")
        sample_n = min(2000, len(y_test))
        idx = np.random.RandomState(42).choice(len(y_test), sample_n, replace=False)
        fig = px.scatter(
            x=y_test[idx], y=y_pred[idx],
            labels={"x": "Actual PM2.5 (µg/m³)", "y": "Predicted PM2.5 (µg/m³)"},
            template="plotly_white",
            opacity=0.45,
            color_discrete_sequence=["#3b82d4"],
        )
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=float(y_test.max()), y1=float(y_test.max()),
            line=dict(dash="dash", color="red", width=1.5),
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red dashed line = perfect prediction. Points above/below = over/under-prediction.")

    with col_right:
        st.markdown("#### 📉 Residual Distribution")
        residuals = y_test - y_pred
        fig = px.histogram(
            x=residuals,
            nbins=60,
            labels={"x": "Residual (Actual − Predicted)"},
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1.5)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Mean residual: {residuals.mean():.2f} µg/m³ · "
            f"Std: {residuals.std():.2f} µg/m³"
        )

    # Time-series actual vs predicted
    st.markdown("#### 📈 Actual vs Predicted — Test Set Time Series")
    df_test_plot = pd.DataFrame({
        "Date":      pd.to_datetime(test_dates),
        "Actual":    y_test,
        "Predicted": y_pred,
        "City":      test_cities,
    }).sort_values("Date")

    test_cities_uniq = sorted(df_test_plot["City"].unique())
    sel_city_perf = st.selectbox("📍 City (Performance view)", test_cities_uniq)
    df_city_perf = df_test_plot[df_test_plot["City"] == sel_city_perf]

    if not df_city_perf.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_city_perf["Date"], y=df_city_perf["Actual"],
            mode="lines", name="Actual",
            line=dict(color="#3b82d4", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df_city_perf["Date"], y=df_city_perf["Predicted"],
            mode="lines", name="Predicted",
            line=dict(color="#dc2626", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="PM2.5 (µg/m³)",
            margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("#### 🔍 Feature Importance (XGBoost split gain)")
    df_fi = get_feature_importance(pipeline, feat_cols)
    fig = px.bar(
        df_fi.head(15), x="importance", y="feature", orientation="h",
        template="plotly_white",
        color="importance", color_continuous_scale="Blues",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "ℹ️ Feature importance indicates contribution to prediction error reduction. "
        "It does NOT imply causation."
    )

    # SHAP (optional)
    st.markdown("---")
    st.markdown("#### 🧩 SHAP Explainability")
    if not SHAP_AVAILABLE:
        st.info(
            "SHAP is not installed in this environment. "
            "The feature importance chart above provides equivalent model explainability."
        )
    else:
        with st.spinner("Computing SHAP values…"):
            try:
                df_full = load_processed()
                df_full_eng, _, _ = run_feature_engineering(df_full, POLLUTANT_COLS)
                X_sample = df_full_eng[feat_cols].values[:300]
                sv, _ = shap_explanation(pipeline, X_sample, feat_cols, max_samples=300)
                if sv is not None:
                    df_shap = shap_summary_df(sv, feat_cols)
                    fig = px.bar(
                        df_shap.head(15), x="mean_abs_shap", y="feature", orientation="h",
                        template="plotly_white",
                        color="mean_abs_shap", color_continuous_scale="Purples",
                        labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}, height=500, margin=dict(t=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "SHAP values measure each feature's average contribution to the model output. "
                        "Computed on a 300-sample subset."
                    )
                else:
                    st.info("SHAP not available for this model type.")
            except Exception as e:
                st.warning(f"SHAP computation skipped: {e}")


# ===========================================================================
# PAGE 6 — DATA QUALITY
# ===========================================================================
elif page == "🧹 Data Quality":
    page_header(
        "🧹", "Data Quality & Preprocessing",
        "Dataset inventory, missing-value audit, and preprocessing decisions"
    )

    # Dataset selection rationale
    st.markdown("#### ✅ Selected Modelling Dataset")
    st.success(
        "**`city_day.csv` — Daily city-level pollution data**  \n"
        "📋 29,531 daily observations · 📍 26 Indian cities · 📅 2015–2020  \n"
        "All required pollutant features present: PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene  \n"
        "✅ Manageable 2.45 MB — no aggregation from hourly data needed  \n"
        "✅ `station_day.csv` aggregated at station level — `city_day.csv` already consolidates across stations  \n"
        "❌ `station_hour.csv` (210 MB) and `city_hour.csv` (63 MB) excluded — excessive for this task"
    )

    st.markdown("#### 🔒 Leakage Prevention")
    st.error(
        "**Excluded features (data leakage risk):**  \n"
        "- `AQI` — computed directly from PM2.5 via the Indian CPCB sub-index formula  \n"
        "- `AQI_Bucket` — categorical label derived from AQI, and therefore from PM2.5  \n\n"
        "Using these as inputs would let the model trivially learn the answer from the label, "
        "making it useless in production where AQI is unknown before PM2.5 is measured."
    )

    # All files summary
    st.markdown("#### 📁 Dataset File Inventory")
    with st.spinner("Reading file metadata…"):
        summaries = get_file_summaries()

    for fname, info in summaries.items():
        icon = "✅" if "error" not in info else "❌"
        with st.expander(f"{icon} {fname}"):
            if "error" in info:
                st.warning(f"Not present in this environment: {info['error']}")
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Size (MB)",    info["size_mb"])
            c2.metric("Rows",         f"{info['rows']:,}")
            c3.metric("Columns",      info["cols"])
            c4.metric("Granularity",  info["granularity"])
            st.markdown(f"**Columns:** {', '.join(info['columns'])}")
            if info["missing_pct_sample"]:
                miss_df = pd.DataFrame.from_dict(
                    {k: [v] for k, v in info["missing_pct_sample"].items() if v > 0},
                    orient="columns",
                )
                if not miss_df.empty:
                    st.markdown("**Missing % (first 5,000 rows sample):**")
                    st.dataframe(miss_df, use_container_width=True)

    # Missing-value chart
    st.markdown("#### 📊 Missing Values in city_day.csv (Before Cleaning)")
    _raw_path = os.path.join(os.path.dirname(__file__), "city_day.csv")
    if os.path.exists(_raw_path):
        raw_df = pd.read_csv(_raw_path)
        miss = raw_df.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing Count"]
        miss["Missing %"] = (miss["Missing Count"] / len(raw_df) * 100).round(1)
        miss = miss[miss["Missing Count"] > 0].sort_values("Missing %", ascending=False)
        fig = px.bar(
            miss, x="Column", y="Missing %",
            template="plotly_white",
            color="Missing %", color_continuous_scale="Reds",
            labels={"Missing %": "Missing %", "Column": "Column"},
            title="Missing Value % per Column (raw data)",
        )
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "city_day.csv is not bundled in this deployment. "
            "Showing pre-computed statistics from the training run."
        )
        _known_missing = {
            "Xylene": 61.3, "PM10": 37.7, "Toluene": 27.2,
            "NH3": 34.9, "Benzene": 19.0, "AQI": 15.8,
            "AQI_Bucket": 15.8, "PM2.5": 15.6, "O3": 13.6,
            "SO2": 13.0, "NOx": 14.2, "NO2": 12.1,
            "NO": 12.1, "CO": 7.0,
        }
        miss = pd.DataFrame(
            list(_known_missing.items()), columns=["Column", "Missing %"]
        ).sort_values("Missing %", ascending=False)
        miss["Missing Count"] = (miss["Missing %"] / 100 * 29531).astype(int)
        fig = px.bar(
            miss, x="Column", y="Missing %",
            template="plotly_white",
            color="Missing %", color_continuous_scale="Reds",
            labels={"Missing %": "Missing %", "Column": "Column"},
            title="Missing Value % per Column (pre-computed)",
        )
        fig.update_layout(margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if "miss" in dir():
            st.dataframe(miss, use_container_width=True)
    with col_b:
        st.markdown("#### 🛠️ Preprocessing Actions")
        steps = [
            ("🔒", "AQI and AQI_Bucket dropped — data leakage"),
            ("🗑️", "4,598 rows with missing PM2.5 removed"),
            ("📌", "Extreme PM2.5 > 1,500 µg/m³ capped at 999"),
            ("♻️", "Duplicate rows removed (0 found)"),
            ("📝", "City names standardised (title-case, trimmed)"),
            ("🔢", "Non-numeric pollutant values coerced to NaN"),
            ("✂️", "Negative pollutant values clipped to 0"),
            ("📊", "Missing pollutants filled with city-level median"),
            ("📊", "Remaining NaN filled with global column median"),
        ]
        for icon, step in steps:
            st.markdown(f"{icon} {step}")

        st.markdown("---")
        st.markdown(f"**Cleaned records:** {len(df):,}")
        st.markdown(f"**Unique cities:** {df['City'].nunique()}")
        st.markdown(
            f"**Date range:** {df['Date'].min().date()} — {df['Date'].max().date()}"
        )

    # City × Year data coverage heatmap
    st.markdown("---")
    st.markdown("#### 🗺️ City × Year Data Coverage")
    df["year"] = df["Date"].dt.year
    coverage = df.groupby(["City", "year"])["PM2.5"].count().reset_index()
    coverage.columns = ["City", "Year", "Record Count"]
    pivot = coverage.pivot(index="City", columns="Year", values="Record Count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values.astype(int),
        texttemplate="%{text}",
    ))
    fig.update_layout(template="plotly_white", height=600, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
