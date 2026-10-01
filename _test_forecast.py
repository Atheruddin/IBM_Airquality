"""
_test_forecast.py
-----------------
Exercises the full Next-Day Forecast logic (without Streamlit) to verify:
  1. No TypeError from Plotly Timestamp arithmetic
  2. Date column in df_fc is uniform datetime64
  3. df_fc_display Date column is string (Arrow-safe)
  4. add_shape / add_annotation work without errors
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import load_processed, POLLUTANT_COLS, pm25_category
from src.feature_engineering import run_feature_engineering, build_single_row, SEASON_MAP
from src.train import load_artefact

df = load_processed()
art = load_artefact()
pipeline  = art["pipeline"]
feat_cols = art["feat_cols"]
encoders  = art["encoders"]

# ── Replicate the forecast page logic exactly ────────────────────────────────
sel_city_fc   = "Delhi"
forecast_days = 7

df_city = df[df["City"] == sel_city_fc].sort_values("Date").tail(90)

city_medians      = df[df["City"] == sel_city_fc][POLLUTANT_COLS].median().fillna(0)
pol_cols_available = [c for c in POLLUTANT_COLS if c in df.columns]
pol_vals           = {c: float(city_medians.get(c, 0.0)) for c in pol_cols_available}

last_known   = df_city["PM2.5"].tail(7).tolist()
last_date    = df_city["Date"].max()
forecast_rows = []
rolling_hist = last_known.copy()

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
        "Date": pd.Timestamp(fc_date),        # uniform Timestamp
        "Forecast PM2.5": round(pred, 2),
        "Category": cat_label,
        "Type": "Forecast",
    })
    rolling_hist.append(pred)
    rolling_hist = rolling_hist[-7:]

df_fc = pd.DataFrame(forecast_rows)
df_fc["Date"] = pd.to_datetime(df_fc["Date"])

# ── Check 1: dtype must be datetime64 ────────────────────────────────────────
assert df_fc["Date"].dtype == "datetime64[ns]", f"Bad dtype: {df_fc['Date'].dtype}"
print(f"Check 1 PASS: Date dtype = {df_fc['Date'].dtype}")

# ── Check 2: no Python date objects in the column ─────────────────────────────
import datetime
for v in df_fc["Date"]:
    assert not isinstance(v, datetime.date) or isinstance(v, pd.Timestamp), \
        f"Found raw date object: {v!r}"
print("Check 2 PASS: no raw datetime.date objects in Date column")

# ── Build the combined chart dataframe ───────────────────────────────────────
hist_plot = df_city[["Date", "PM2.5"]].tail(30).copy()
hist_plot["Type"] = "Historical"
hist_plot.rename(columns={"PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
fc_plot = df_fc.copy()
fc_plot.rename(columns={"Forecast PM2.5": "PM2.5 (µg/m³)"}, inplace=True)
combined = pd.concat([
    hist_plot[["Date", "PM2.5 (µg/m³)", "Type"]],
    fc_plot[["Date",   "PM2.5 (µg/m³)", "Type"]],
], ignore_index=True)

# ── Check 3: Plotly chart + add_shape/add_annotation without TypeError ────────
fig = px.line(
    combined, x="Date", y="PM2.5 (µg/m³)", color="Type",
    color_discrete_map={"Historical": "#3b82d4", "Forecast": "#dc2626"},
    template="plotly_white",
)
_vline_x = last_date.isoformat()    # ISO string — no integer arithmetic in Plotly
fig.add_shape(
    type="line",
    x0=_vline_x, x1=_vline_x,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(dash="dash", color="#57606a", width=1.5),
)
fig.add_annotation(
    x=_vline_x, y=1,
    xref="x", yref="paper",
    text="Forecast start",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(size=11, color="#57606a"),
    bgcolor="rgba(255,255,255,0.7)",
)
# Serialise to dict — would raise if Timestamp arithmetic was triggered
_ = fig.to_dict()
print("Check 3 PASS: add_shape/add_annotation no TypeError")

# ── Check 4: display df has string Date (Arrow-safe) ─────────────────────────
df_fc_display = df_fc.copy()
df_fc_display["Date"] = df_fc_display["Date"].dt.strftime("%Y-%m-%d")
assert df_fc_display["Date"].dtype == object, f"Expected str, got {df_fc_display['Date'].dtype}"
# All values must be strings
for v in df_fc_display["Date"]:
    assert isinstance(v, str), f"Not a string: {v!r}"
print("Check 4 PASS: display Date column is uniform string")

print()
print("Forecast table preview:")
print(df_fc_display.to_string(index=False))
print()
print("ALL FORECAST CHECKS PASSED")
