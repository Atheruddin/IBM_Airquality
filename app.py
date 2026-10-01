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
    page_icon="🌬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2328;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #57606a;
        margin-top: 4px;
    }
    .prediction-box {
        background: #f0f9ff;
        border: 2px solid #3b82d4;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .pred-value {
        font-size: 3rem;
        font-weight: 800;
        color: #1f2328;
    }
    .pred-unit {
        font-size: 1.2rem;
        color: #57606a;
    }
    .category-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


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
    "Overview",
    "Air Quality Analytics",
    "PM2.5 Prediction",
    "Next-Day Forecast",
    "Model Performance",
    "Data Quality",
]

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/UN_emblem_blue.svg/240px-UN_emblem_blue.svg.png",
    width=60,
)
st.sidebar.markdown("## AI Air Quality\nMonitoring & Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** Air Quality India 2015–2020  \n"
    "**Model:** PM2.5 Regression  \n"
    "**SDG:** 11 & 13"
)

# Load data & model
df = get_data()
art = get_artefact()
pipeline = art["pipeline"]
feat_cols = art["feat_cols"]
encoders = art["encoders"]
all_metrics = art["all_metrics"]
best_name = art["model_name"]
best_metrics = art["metrics"]


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "Overview":
    st.title("AI Air Quality Monitoring & PM2.5 Prediction")
    st.markdown(
        "> **UN SDG 11** – Sustainable Cities and Communities &nbsp;|&nbsp; "
        "**UN SDG 13** – Climate Action"
    )
    st.markdown(
        "This system analyses historical air-quality data from 26 Indian cities "
        "(2015–2020) and provides ML-based PM2.5 concentration predictions. "
        "Data source: [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)"
    )

    # KPI metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    kpis = [
        (col1, "Records", f"{len(df):,}", "#3b82d4"),
        (col2, "Cities", str(df["City"].nunique()), "#7c5cd8"),
        (col3, "Date Range", f"{df['Date'].min().year}–{df['Date'].max().year}", "#059669"),
        (col4, "Avg PM2.5", f"{df['PM2.5'].mean():.1f} µg/m³", "#d97706"),
        (col5, "Max PM2.5", f"{df['PM2.5'].max():.0f} µg/m³", "#dc2626"),
        (col6, f"Best Model R²", f"{best_metrics['R2']:.4f}", "#0284c7"),
    ]
    for col, label, value, colour in kpis:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{colour}">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Model performance summary
    st.subheader("Best Model Summary")
    m_cols = st.columns(4)
    m_cols[0].metric("Model", best_name)
    m_cols[1].metric("MAE", f"{best_metrics['MAE']:.2f}")
    m_cols[2].metric("RMSE", f"{best_metrics['RMSE']:.2f}")
    m_cols[3].metric("R²", f"{best_metrics['R2']:.4f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("PM2.5 Trend Over Time (All Cities)")
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
        fig.update_traces(line_color="#3b82d4")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="WHO guideline (60)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Average PM2.5 by City")
        city_avg = df.groupby("City")["PM2.5"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            city_avg, x="PM2.5", y="City", orientation="h",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)", "City": ""},
            template="plotly_white",
            color="PM2.5",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("PM2.5 Distribution")
        fig = px.histogram(
            df, x="PM2.5", nbins=60,
            labels={"PM2.5": "PM2.5 (µg/m³)"},
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Average PM2.5 by Season")
        df["season"] = df["Date"].dt.month.map(SEASON_MAP)
        season_avg = df.groupby("season")["PM2.5"].mean().reset_index()
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
        season_avg["season"] = pd.Categorical(season_avg["season"], categories=season_order, ordered=True)
        season_avg = season_avg.sort_values("season")
        fig = px.bar(
            season_avg, x="season", y="PM2.5",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)", "season": "Season"},
            template="plotly_white",
            color="PM2.5",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "---\n**Disclaimer:** Predictions are estimates based on historical patterns. "
        "This is an academic demonstration using the Air Quality Data in India (2015–2020) dataset."
    )


# ===========================================================================
# PAGE 2 — AIR QUALITY ANALYTICS
# ===========================================================================
elif page == "Air Quality Analytics":
    st.title("Air Quality Analytics")

    df["season"] = df["Date"].dt.month.map(SEASON_MAP)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year

    # Sidebar filters
    cities = sorted(df["City"].unique().tolist())
    sel_cities = st.sidebar.multiselect("Filter Cities", cities, default=cities[:5])
    date_min = df["Date"].min().date()
    date_max = df["Date"].max().date()
    import datetime
    d_range = st.sidebar.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(d_range, (list, tuple)) and len(d_range) == 2:
        d_start, d_end = pd.Timestamp(d_range[0]), pd.Timestamp(d_range[1])
    else:
        d_start, d_end = pd.Timestamp(date_min), pd.Timestamp(date_max)

    pollutant_opts = ["PM2.5"] + [c for c in POLLUTANT_COLS if c in df.columns]
    sel_pollutant = st.sidebar.selectbox("Pollutant Focus", pollutant_opts)

    df_filt = df[
        (df["City"].isin(sel_cities if sel_cities else cities)) &
        (df["Date"] >= d_start) &
        (df["Date"] <= d_end)
    ].copy()

    if df_filt.empty:
        st.warning("No data matches the current filter. Adjust filters in the sidebar.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "City Comparison", "Trends", "Seasonal & Calendar", "Correlations", "Selected City"
    ])

    # ---- Tab 1: City Comparison ----
    with tab1:
        st.subheader("PM2.5 City Comparison")
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
                labels={"PM2.5": f"Avg PM2.5 (µg/m³)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**City-wise box plot**")
            fig = px.box(
                df_filt.sort_values("City"), x="City", y="PM2.5",
                template="plotly_white",
                color="City",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"{sel_pollutant} — Time Series Comparison")
        city_ts = df_filt.groupby(["Date", "City"])[sel_pollutant].mean().reset_index()
        fig = px.line(
            city_ts, x="Date", y=sel_pollutant, color="City",
            template="plotly_white",
            labels={sel_pollutant: f"{sel_pollutant} (µg/m³)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Trends ----
    with tab2:
        st.subheader("Monthly Trend")
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
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Yearly Average PM2.5")
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
            st.subheader("Average PM2.5 by Month")
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
            st.subheader("PM2.5 by Day of Week")
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

        st.subheader("PM2.5 by Season")
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post-Monsoon"]
        season_avg = df_filt.groupby("season")["PM2.5"].mean().reset_index()
        season_avg["season"] = pd.Categorical(season_avg["season"], categories=season_order, ordered=True)
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
        st.subheader("Pollutant Correlation Heatmap")
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

        pair_plots = [("PM10", "PM2.5"), ("NO2", "PM2.5"), ("CO", "PM2.5"), ("SO2", "PM2.5")]
        cols = st.columns(2)
        for i, (x_col, y_col) in enumerate(pair_plots):
            if x_col in df_filt.columns:
                fig = px.scatter(
                    df_filt.sample(min(2000, len(df_filt)), random_state=42),
                    x=x_col, y=y_col,
                    color="City", opacity=0.5,
                    template="plotly_white",
                    title=f"{y_col} vs {x_col}",
                    trendline="ols",
                )
                cols[i % 2].plotly_chart(fig, use_container_width=True)

    # ---- Tab 5: Selected City ----
    with tab5:
        sel_city = st.selectbox("Select City", sorted(df["City"].unique()))
        df_city = df[df["City"] == sel_city].sort_values("Date")

        st.subheader(f"Monthly PM2.5 Trend — {sel_city}")
        monthly_city = df_city.set_index("Date").resample("MS")["PM2.5"].mean().reset_index()
        fig = px.line(
            monthly_city, x="Date", y="PM2.5",
            template="plotly_white",
            labels={"PM2.5": "Avg PM2.5 (µg/m³)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Multi-Pollutant Comparison — {sel_city}")
        available_pols = [c for c in POLLUTANT_COLS if c in df_city.columns and df_city[c].notna().sum() > 10]
        sel_pols = st.multiselect("Pollutants", available_pols, default=available_pols[:3])
        if sel_pols:
            plot_df = df_city.set_index("Date")[sel_pols].resample("MS").mean().reset_index()
            fig = px.line(
                plot_df.melt(id_vars="Date", var_name="Pollutant", value_name="Concentration"),
                x="Date", y="Concentration", color="Pollutant",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Data Summary — {sel_city}")
        st.dataframe(df_city.describe().round(2), use_container_width=True)


# ===========================================================================
# PAGE 3 — PM2.5 PREDICTION
# ===========================================================================
elif page == "PM2.5 Prediction":
    st.title("PM2.5 Prediction")
    st.markdown(
        "Enter the pollutant readings and select a city/date. "
        "The model will predict the expected PM2.5 concentration. "
        "Only features that would realistically be known before measurement are used."
    )
    st.info(
        "**Leakage note:** AQI and AQI_Bucket are NOT used as inputs — "
        "they are derived from PM2.5 and would leak the target into predictions."
    )

    import datetime
    cities = sorted(df["City"].unique().tolist())

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Input Features")
        sel_city = st.selectbox("City", cities, key="pred_city")
        sel_date = st.date_input(
            "Date",
            value=datetime.date(2020, 1, 15),
            min_value=datetime.date(2015, 1, 1),
            max_value=datetime.date(2025, 12, 31),
        )

        st.markdown("**Pollutant Readings** (leave at 0 if unknown)")
        pol_cols_available = [c for c in POLLUTANT_COLS if c in df.columns]
        # Use city-level medians as defaults
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

        st.markdown("**Recent PM2.5 History** (optional — improves prediction)")
        hist_vals = []
        hc1, hc2, hc3 = st.columns(3)
        d7 = hc1.number_input("7 days ago", 0.0, 999.0, float(city_medians.get("PM2.5", 50.0)) if "PM2.5" in city_medians else 50.0, step=0.1, key="h7")
        d3 = hc2.number_input("3 days ago", 0.0, 999.0, float(city_medians.get("PM2.5", 50.0)) if "PM2.5" in city_medians else 50.0, step=0.1, key="h3")
        d1 = hc3.number_input("Yesterday", 0.0, 999.0, float(city_medians.get("PM2.5", 50.0)) if "PM2.5" in city_medians else 50.0, step=0.1, key="h1")
        hist_vals = [d7, 0, 0, 0, d3, 0, d1]

        predict_btn = st.button("Predict PM2.5", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")
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
                pred_pm25 = float(pipeline.predict(row_df)[0])
                pred_pm25 = max(0.0, pred_pm25)

                cat_label, cat_colour = pm25_category(pred_pm25)

                st.markdown(
                    f'<div class="prediction-box">'
                    f'<div style="font-size:1rem;color:#57606a">Predicted PM2.5</div>'
                    f'<div class="pred-value">{pred_pm25:.2f}</div>'
                    f'<div class="pred-unit">µg/m³</div>'
                    f'<br><span class="category-badge" style="background:{cat_colour};color:white">'
                    f'{cat_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown(f"**City:** {sel_city}  |  **Date:** {sel_date}")
                st.markdown(
                    "**PM2.5 Categories (Indian CPCB standard):**  \n"
                    "Good (0–30) | Satisfactory (30–60) | Moderately Polluted (60–90) | "
                    "Poor (90–120) | Very Poor (120–250) | Severe (>250)"
                )
                st.markdown(
                    "> *Note: PM2.5 concentration ≠ AQI. AQI is computed from multiple "
                    "pollutants using CPCB sub-index formulas and is NOT predicted here.*"
                )

                # Top features
                st.markdown("---")
                st.subheader("Top Contributing Features")
                df_fi = get_feature_importance(pipeline, feat_cols)
                fig = px.bar(
                    df_fi.head(10), x="importance", y="feature", orientation="h",
                    template="plotly_white",
                    labels={"importance": "Importance", "feature": "Feature"},
                    color="importance", color_continuous_scale="Blues",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Feature importance reflects predictive power in the model — "
                    "it does NOT imply causation."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.markdown(
                "Fill in the form on the left and click **Predict PM2.5** to see the result."
            )
            st.markdown("---")
            st.subheader("Feature Importance (Model)")
            df_fi = get_feature_importance(pipeline, feat_cols)
            fig = px.bar(
                df_fi.head(12), x="importance", y="feature", orientation="h",
                template="plotly_white",
                labels={"importance": "Importance", "feature": "Feature"},
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4 — NEXT-DAY FORECAST
# ===========================================================================
elif page == "Next-Day Forecast":
    st.title("Next-Day PM2.5 Forecast")
    st.markdown(
        "This component uses historical PM2.5 lags, rolling averages, "
        "calendar features, and city information — all available **before** "
        "the target date — to forecast the next day's PM2.5 concentration.  \n"
        "Forecast values are **model predictions**, not measurements."
    )
    st.info(
        "Because pollutant co-readings for future dates are unknown, "
        "this forecast uses the rolling-average and lag features as the primary "
        "signal, supplemented by city-level median pollutant values."
    )

    cities = sorted(df["City"].unique().tolist())
    import datetime
    sel_city_fc = st.sidebar.selectbox("City (Forecast)", cities, key="fc_city")
    forecast_days = st.sidebar.slider("Days to Forecast", 1, 14, 7)

    st.subheader(f"Historical PM2.5 — {sel_city_fc}")
    df_city = df[df["City"] == sel_city_fc].sort_values("Date").tail(90)
    fig = px.line(
        df_city, x="Date", y="PM2.5",
        template="plotly_white",
        labels={"PM2.5": "PM2.5 (µg/m³)"},
    )
    fig.update_traces(line_color="#3b82d4")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Generate Forecast", type="primary"):
        city_medians = df[df["City"] == sel_city_fc][POLLUTANT_COLS].median().fillna(0)
        pol_cols_available = [c for c in POLLUTANT_COLS if c in df.columns]
        pol_vals = {c: float(city_medians.get(c, 0.0)) for c in pol_cols_available}

        last_known = df_city["PM2.5"].tail(7).tolist()

        last_date = df_city["Date"].max()
        forecast_rows = []
        rolling_hist = last_known.copy()

        for i in range(1, forecast_days + 1):
            fc_date = last_date + pd.Timedelta(days=i)
            row_df = build_single_row(
                city=sel_city_fc,
                date=fc_date,
                pollutant_values=pol_vals,
                historical_pm25=rolling_hist,
                encoders=encoders,
                feat_cols=feat_cols,
            )
            pred = float(pipeline.predict(row_df)[0])
            pred = max(0.0, pred)
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

        st.subheader("Forecast Results")

        # Combine historical and forecast for plot
        hist_plot = df_city[["Date", "PM2.5"]].tail(30).copy()
        hist_plot["Type"] = "Historical"
        hist_plot.rename(columns={"PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
        fc_plot = df_fc.copy()
        fc_plot["Date"] = pd.to_datetime(fc_plot["Date"])
        fc_plot.rename(columns={"Forecast PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
        combined = pd.concat([
            hist_plot[["Date", "PM2.5 (µg/m³)", "Type"]],
            fc_plot[["Date", "PM2.5 (µg/m³)", "Type"]],
        ], ignore_index=True)

        fig = px.line(
            combined, x="Date", y="PM2.5 (µg/m³)", color="Type",
            color_discrete_map={"Historical": "#3b82d4", "Forecast": "#dc2626"},
            template="plotly_white",
        )
        fig.add_vline(x=last_date, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_fc, use_container_width=True)
        st.caption(
            "Forecasts are generated by iterating the model forward using predicted "
            "values as lag features. Uncertainty increases with forecast horizon."
        )


# ===========================================================================
# PAGE 5 — MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":
    st.title("Model Performance")

    # Model comparison table
    st.subheader("Model Comparison")
    df_cmp = compare_models(all_metrics)
    st.dataframe(df_cmp.style.highlight_min(subset=["MAE","RMSE"], color="#d1fae5")
                 .highlight_max(subset=["R²"], color="#d1fae5"),
                 use_container_width=True)

    # Model comparison chart
    fig = go.Figure()
    for _, row in df_cmp.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=["MAE", "RMSE"],
            y=[row["MAE"], row["RMSE"]],
        ))
    fig.update_layout(barmode="group", template="plotly_white", title="MAE & RMSE by Model")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Selected model:** `{best_name}` (lowest RMSE on chronological test set)")
    st.markdown(
        "**Train/Test split:** Chronological 80/20 — earlier dates for training, "
        "later dates for testing. No future data leaks into training."
    )
    st.markdown(
        f"Train: {art['train_date_range'][0][:10]} to {art['train_date_range'][1][:10]}  |  "
        f"Test: {art['test_date_range'][0][:10]} to {art['test_date_range'][1][:10]}"
    )

    st.markdown("---")

    # Actual vs Predicted
    y_test = np.array(art["y_test"])
    y_pred = np.array(art["y_pred_best"])
    test_dates = art["test_dates"]
    test_cities = art["test_cities"]

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Actual vs Predicted PM2.5")
        sample_n = min(2000, len(y_test))
        idx = np.random.RandomState(42).choice(len(y_test), sample_n, replace=False)
        fig = px.scatter(
            x=y_test[idx], y=y_pred[idx],
            labels={"x": "Actual PM2.5", "y": "Predicted PM2.5"},
            template="plotly_white",
            opacity=0.5,
        )
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=float(y_test.max()), y1=float(y_test.max()),
            line=dict(dash="dash", color="red"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Residuals")
        residuals = y_test - y_pred
        fig = px.histogram(
            x=residuals,
            nbins=60,
            labels={"x": "Residual (Actual − Predicted)"},
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    # Time-series actual vs predicted
    st.subheader("Actual vs Predicted — Test Set Time Series")
    df_test_plot = pd.DataFrame({
        "Date": pd.to_datetime(test_dates),
        "Actual": y_test,
        "Predicted": y_pred,
        "City": test_cities,
    }).sort_values("Date")

    test_cities_uniq = sorted(df_test_plot["City"].unique())
    sel_city_perf = st.selectbox("City (Performance)", test_cities_uniq)
    df_city_perf = df_test_plot[df_test_plot["City"] == sel_city_perf]

    if not df_city_perf.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_city_perf["Date"], y=df_city_perf["Actual"],
                                  mode="lines", name="Actual", line=dict(color="#3b82d4")))
        fig.add_trace(go.Scatter(x=df_city_perf["Date"], y=df_city_perf["Predicted"],
                                  mode="lines", name="Predicted", line=dict(color="#dc2626", dash="dot")))
        fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.subheader("Feature Importance")
    df_fi = get_feature_importance(pipeline, feat_cols)
    fig = px.bar(
        df_fi.head(15), x="importance", y="feature", orientation="h",
        template="plotly_white",
        color="importance", color_continuous_scale="Blues",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance indicates each variable's contribution to reducing "
        "prediction error in the tree model. Importance does NOT imply causation."
    )

    # SHAP (optional)
    st.markdown("---")
    st.subheader("SHAP Explainability")
    if not SHAP_AVAILABLE:
        st.info(
            "SHAP is not installed in this environment. "
            "The feature importance chart above provides equivalent model explainability."
        )
    else:
        with st.spinner("Computing SHAP values (this may take a moment)..."):
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
                    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "SHAP (SHapley Additive exPlanations) measures the average contribution "
                        "of each feature to the model output. Computed on a 300-sample subset."
                    )
                else:
                    st.info("SHAP not available for this model type.")
            except Exception as e:
                st.warning(f"SHAP computation skipped: {e}")


# ===========================================================================
# PAGE 6 — DATA QUALITY
# ===========================================================================
elif page == "Data Quality":
    st.title("Data Quality & Preprocessing")

    # Dataset selection rationale
    st.subheader("Dataset Selection")
    st.success(
        "**Selected dataset for ML modelling: `city_day.csv`**  \n"
        "- 29,531 daily observations across 26 Indian cities (2015–2020)  \n"
        "- All required pollutant features present (PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene)  \n"
        "- Manageable size (2.45 MB) for academic ML demonstration  \n"
        "- No aggregation from hourly data needed  \n"
        "- `station_hour.csv` (210 MB) and `city_hour.csv` (63 MB) are excessive for this task  \n"
        "- `station_day.csv` is at station level — city_day already aggregates across stations"
    )

    st.subheader("Leakage Prevention")
    st.error(
        "**Excluded features (data leakage):**  \n"
        "- `AQI` — computed directly from PM2.5 and other pollutants using CPCB formula  \n"
        "- `AQI_Bucket` — categorical label derived from AQI, therefore from PM2.5  \n\n"
        "Using these as inputs would tell the model the answer before it predicts — "
        "making the model useless in real deployment where AQI is unknown."
    )

    # All files summary
    st.subheader("All Dataset Files")
    with st.spinner("Reading file metadata..."):
        summaries = get_file_summaries()

    for fname, info in summaries.items():
        with st.expander(f"📄 {fname}"):
            if "error" in info:
                st.error(info["error"])
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Size (MB)", info["size_mb"])
            c2.metric("Rows", f"{info['rows']:,}")
            c3.metric("Columns", info["cols"])
            c4.metric("Granularity", info["granularity"])
            st.markdown(f"**Columns:** {', '.join(info['columns'])}")
            if info["missing_pct_sample"]:
                miss_df = pd.DataFrame.from_dict(
                    {k: [v] for k, v in info["missing_pct_sample"].items() if v > 0},
                    orient="columns"
                )
                if not miss_df.empty:
                    st.markdown("**Missing % (sample):**")
                    st.dataframe(miss_df, use_container_width=True)

    # Missing-value visualization
    st.subheader("Missing Values in city_day.csv (Before Cleaning)")
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
            title="Missing Value % per Column",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # city_day.csv not present in this deployment — show pre-computed summary
        st.info(
            "city_day.csv is not bundled in this deployment. "
            "Showing pre-computed missing-value statistics from the training run."
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
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if "miss" in dir():
            st.dataframe(miss, use_container_width=True)
    with col_b:
        st.markdown("**Preprocessing Actions Taken:**")
        steps = [
            "1. AQI and AQI_Bucket dropped (leakage)",
            "2. Rows with missing PM2.5 removed (4,598 rows)",
            "3. Extreme PM2.5 values (>1500) capped at 999",
            "4. Duplicate rows removed",
            "5. City names standardized (title-case, whitespace stripped)",
            "6. Non-numeric pollutant values coerced to NaN",
            "7. Negative pollutant values clipped to 0",
            "8. Missing pollutants filled with city-level median",
            "9. Remaining NaN filled with global median",
        ]
        for s in steps:
            st.markdown(f"✓ {s}")

        st.markdown("---")
        st.markdown(f"**Cleaned record count:** {len(df):,}")
        st.markdown(f"**Unique cities:** {df['City'].nunique()}")
        st.markdown(f"**Date range:** {df['Date'].min().date()} to {df['Date'].max().date()}")

    # City coverage heatmap
    st.subheader("City × Year Data Coverage")
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
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)
