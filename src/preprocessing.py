"""
preprocessing.py
================
Loads and cleans city_day.csv for PM2.5 prediction.

Selected dataset: city_day.csv
Reason:
  - 29,531 daily observations across 26 Indian cities (2015-2020)
  - All required pollutant features present
  - Manageable size for academic ML demonstration
  - No need to aggregate hourly data
  - station_hour.csv (210 MB) would be excessive for this task

Leakage prevention:
  AQI and AQI_Bucket are computed directly from PM2.5 (and other pollutants)
  using the Indian CPCB formula.  Using them as model inputs would leak the
  target into features.  Both columns are dropped before any modelling step.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(ROOT_DIR, "city_day.csv")
PROCESSED_DATA = os.path.join(ROOT_DIR, "data", "city_day_processed.csv")

# ---------------------------------------------------------------------------
# Pollutant columns (model feature candidates)
# ---------------------------------------------------------------------------
POLLUTANT_COLS = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
                  "Benzene", "Toluene", "Xylene"]

# Leakage columns — NEVER use as model inputs
LEAKAGE_COLS = ["AQI", "AQI_Bucket"]

TARGET = "PM2.5"


# ---------------------------------------------------------------------------
# Helper: load raw data
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load city_day.csv and return as DataFrame with parsed Date."""
    df = pd.read_csv(RAW_DATA)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Inspection helpers (used by the Data Quality page)
# ---------------------------------------------------------------------------
def inspect_all_files() -> dict:
    """
    Return a summary dict for all five CSV files.
    Uses chunked reading for large files to avoid memory pressure.
    """
    import csv

    summaries = {}
    files = {
        "city_day.csv": {"granularity": "daily", "key_id": "City"},
        "city_hour.csv": {"granularity": "hourly", "key_id": "City"},
        "stations.csv": {"granularity": "reference", "key_id": "StationId"},
        "station_day.csv": {"granularity": "daily", "key_id": "StationId"},
        "station_hour.csv": {"granularity": "hourly", "key_id": "StationId"},
    }

    for fname, meta in files.items():
        fpath = os.path.join(ROOT_DIR, fname)
        if not os.path.exists(fpath):
            summaries[fname] = {"error": "File not found"}
            continue

        size_mb = os.path.getsize(fpath) / (1024 * 1024)

        # Count rows cheaply
        with open(fpath, "r", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            headers = next(reader)
            row_count = sum(1 for _ in reader)

        # Read a small sample to inspect dtypes / missing
        sample = pd.read_csv(fpath, encoding="utf-8-sig", nrows=5000)

        missing_pct = (sample.isnull().sum() / len(sample) * 100).round(1).to_dict()

        summaries[fname] = {
            "size_mb": round(size_mb, 2),
            "rows": row_count,
            "cols": len(headers),
            "columns": headers,
            "granularity": meta["granularity"],
            "key_id": meta["key_id"],
            "missing_pct_sample": missing_pct,
        }

    return summaries


# ---------------------------------------------------------------------------
# Core cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean city_day DataFrame.

    Returns
    -------
    cleaned_df : pd.DataFrame
    report     : dict  — actions taken, counts before/after
    """
    report = {}
    report["rows_raw"] = len(df)
    report["leakage_cols_dropped"] = LEAKAGE_COLS

    # 1. Drop leakage columns
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    # 2. Drop rows where target (PM2.5) is missing — cannot train without it
    missing_target = df[TARGET].isnull().sum()
    df = df.dropna(subset=[TARGET])
    report["rows_dropped_missing_pm25"] = int(missing_target)

    # 3. Clip extreme PM2.5 values
    #    Values > 1500 µg/m³ are implausible; cap at 999 to retain them
    extreme_mask = df[TARGET] > 1500
    report["extreme_pm25_capped"] = int(extreme_mask.sum())
    df[TARGET] = df[TARGET].clip(upper=999)

    # 4. Drop duplicate rows
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(dups)

    # 5. Standardize City names (strip whitespace, title-case)
    df["City"] = df["City"].str.strip().str.title()

    # 6. Ensure numeric pollutant columns
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Clip negative sensor errors to 0
            df[col] = df[col].clip(lower=0)

    # 7. Fill remaining missing pollutant values with city-level median
    #    We use forward/backward fill within each city first, then median
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = df.groupby("City")[col].transform(
                lambda s: s.fillna(s.median() if not s.dropna().empty else 0)
            )
            # Any remaining (city has all NaN for that column) → global median
            global_med = df[col].median()
            df[col] = df[col].fillna(global_med if pd.notna(global_med) else 0)

    report["rows_after_cleaning"] = len(df)
    report["unique_cities"] = df["City"].nunique()
    report["date_range"] = f"{df['Date'].min().date()} — {df['Date'].max().date()}"

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_preprocessing(save: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Load raw data, clean it, optionally save to data/.

    Returns
    -------
    df     : cleaned DataFrame
    report : preprocessing report dict
    """
    os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

    df_raw = load_raw()
    df_clean, report = clean(df_raw)

    if save:
        df_clean.to_csv(PROCESSED_DATA, index=False)
        report["saved_to"] = PROCESSED_DATA

    return df_clean, report


def load_processed() -> pd.DataFrame:
    """Load previously processed data, or run preprocessing if missing."""
    if os.path.exists(PROCESSED_DATA):
        df = pd.read_csv(PROCESSED_DATA)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    df, _ = run_preprocessing(save=True)
    return df


# ---------------------------------------------------------------------------
# Indian AQI classification (display only — NOT used as model feature)
# ---------------------------------------------------------------------------
PM25_CATEGORIES = [
    (0, 30, "Good", "#00b050"),
    (30, 60, "Satisfactory", "#92d050"),
    (60, 90, "Moderately Polluted", "#ffff00"),
    (90, 120, "Poor", "#ff9900"),
    (120, 250, "Very Poor", "#ff0000"),
    (250, float("inf"), "Severe", "#c00000"),
]


def pm25_category(value: float) -> tuple[str, str]:
    """
    Return (category_label, hex_colour) for a PM2.5 value.
    Based on Indian CPCB AQI breakpoints for PM2.5 (µg/m³, 24-hr average).
    NOTE: this is a display classification only — not the same as computing AQI.
    """
    if pd.isna(value):
        return "Unknown", "#888888"
    for lo, hi, label, colour in PM25_CATEGORIES:
        if lo <= value < hi:
            return label, colour
    return "Severe", "#c00000"


if __name__ == "__main__":
    df, report = run_preprocessing(save=True)
    print("Preprocessing complete.")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(df.shape)
    print(df.dtypes)
