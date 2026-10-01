"""
feature_engineering.py
======================
Creates time-based features, lag features, and rolling averages
for PM2.5 prediction.

All lag/rolling features use ONLY information that would be known
BEFORE the prediction timestamp — no target leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Season mapping (Indian meteorological seasons)
# ---------------------------------------------------------------------------
SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring",
    5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
    12: "Winter",
}

FEATURE_COLS = None  # set after engineering


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features from the Date column."""
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek          # 0=Mon
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["season"] = df["month"].map(SEASON_MAP)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "PM2.5") -> pd.DataFrame:
    """
    Add lagged PM2.5 and rolling window averages per city.
    All lags look BACKWARDS — they use past observations only.
    """
    df = df.copy().sort_values(["City", "Date"])

    for lag in [1, 3, 7]:
        col_name = f"pm25_lag_{lag}d"
        df[col_name] = df.groupby("City")[target].shift(lag)

    for window in [3, 7]:
        col_name = f"pm25_roll_{window}d"
        df[col_name] = (
            df.groupby("City")[target]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode City and season.
    Returns augmented df and a dict of encoders.
    """
    encoders = {}

    for col in ["City", "season"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def build_feature_matrix(
    df: pd.DataFrame,
    pollutant_cols: list[str],
    drop_lag_na: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the final feature matrix X and return feature column names.

    Parameters
    ----------
    df             : cleaned DataFrame with lag features already added
    pollutant_cols : list of pollutant column names to include
    drop_lag_na    : whether to drop rows where lag features are NaN
                     (the first few rows per city won't have lags)

    Returns
    -------
    df        : DataFrame with all features
    feat_cols : list of feature column names used for X
    """
    time_feats = [
        "year", "month", "day", "day_of_week", "week_of_year", "is_weekend",
        "City_enc", "season_enc",
    ]
    lag_feats = [c for c in df.columns if c.startswith("pm25_lag_") or
                 c.startswith("pm25_roll_")]

    feat_cols = time_feats + [c for c in pollutant_cols if c in df.columns] + lag_feats

    if drop_lag_na and lag_feats:
        df = df.dropna(subset=lag_feats)

    return df, feat_cols


def run_feature_engineering(
    df: pd.DataFrame,
    pollutant_cols: list[str],
    drop_lag_na: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Full feature engineering pipeline.

    Returns
    -------
    df        : feature-engineered DataFrame
    feat_cols : list of feature column names
    encoders  : dict of LabelEncoders
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df, encoders = encode_categorical(df)
    df, feat_cols = build_feature_matrix(df, pollutant_cols, drop_lag_na)
    return df, feat_cols, encoders


# ---------------------------------------------------------------------------
# Inference helper — build features for a single prediction row
# ---------------------------------------------------------------------------
def build_single_row(
    city: str,
    date: pd.Timestamp,
    pollutant_values: dict,
    historical_pm25: list[float],   # ordered oldest → newest
    encoders: dict,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for model inference.

    Parameters
    ----------
    city              : city name (must match training set)
    date              : prediction date
    pollutant_values  : dict of {col: value} for pollutant features
    historical_pm25   : list of recent daily PM2.5 values (up to 7)
    encoders          : dict returned by run_feature_engineering
    feat_cols         : list of feature columns used during training

    Returns
    -------
    row_df : single-row DataFrame aligned to feat_cols
    """
    row = {}

    # Time features
    row["year"] = date.year
    row["month"] = date.month
    row["day"] = date.day
    row["day_of_week"] = date.dayofweek
    row["week_of_year"] = date.isocalendar()[1]
    row["is_weekend"] = int(date.dayofweek >= 5)
    row["season_enc"] = encoders["season"].transform(
        [SEASON_MAP[date.month]]
    )[0] if "season" in encoders else 0

    # City encoding
    city_norm = city.strip().title()
    city_classes = list(encoders["City"].classes_)
    if city_norm in city_classes:
        row["City_enc"] = int(encoders["City"].transform([city_norm])[0])
    else:
        row["City_enc"] = 0   # fallback

    # Pollutant features
    for col, val in pollutant_values.items():
        row[col] = val

    # Lag features from historical PM2.5
    pm25_hist = list(historical_pm25)
    # Pad with NaN if not enough history
    while len(pm25_hist) < 7:
        pm25_hist.insert(0, np.nan)

    row["pm25_lag_1d"] = pm25_hist[-1] if not np.isnan(pm25_hist[-1]) else 0
    row["pm25_lag_3d"] = pm25_hist[-3] if not np.isnan(pm25_hist[-3]) else 0
    row["pm25_lag_7d"] = pm25_hist[-7] if not np.isnan(pm25_hist[-7]) else 0

    recent = [x for x in pm25_hist[-3:] if not np.isnan(x)]
    row["pm25_roll_3d"] = float(np.mean(recent)) if recent else 0
    recent7 = [x for x in pm25_hist[-7:] if not np.isnan(x)]
    row["pm25_roll_7d"] = float(np.mean(recent7)) if recent7 else 0

    # Build DataFrame aligned to feat_cols
    row_df = pd.DataFrame([row])
    for col in feat_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feat_cols]

    # Return as numpy array to avoid sklearn feature-name warnings
    return row_df.values


if __name__ == "__main__":
    # Quick sanity check
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing import load_processed, POLLUTANT_COLS

    df = load_processed()
    df_eng, feats, enc = run_feature_engineering(df, POLLUTANT_COLS)
    print(f"Feature-engineered shape: {df_eng.shape}")
    print(f"Feature columns ({len(feats)}):")
    for f in feats:
        print(" ", f)
