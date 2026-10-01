"""
train.py
========
Trains multiple regression models for PM2.5 prediction using a
chronological (time-aware) train/test split.

Models trained:
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor
  4. XGBoost Regressor (if available)

The best model is saved to models/best_pm25_model.joblib along with
metadata (feature columns, encoders, scaler, metrics).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.preprocessing import load_processed, POLLUTANT_COLS, TARGET
from src.evaluate import evaluate_model
from src.feature_engineering import run_feature_engineering

MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_pm25_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "all_model_metrics.json")

# Chronological split ratio
TRAIN_RATIO = 0.80


def get_models() -> dict:
    """Return dict of model name → sklearn-compatible estimator."""
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=12,
                min_samples_leaf=3, n_jobs=-1, random_state=42
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=5,
                learning_rate=0.05, subsample=0.8, random_state=42
            )),
        ]),
    }

    # Attempt to add XGBoost
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=200, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist", n_jobs=-1,
                random_state=42, verbosity=0,
            )),
        ])
        print("  XGBoost available — will train.")
    except ImportError:
        print("  XGBoost not available — skipping.")

    return models


def chronological_split(df: pd.DataFrame, ratio: float = TRAIN_RATIO):
    """
    Split df chronologically by date.
    Returns (df_train, df_test, train_end_date).
    """
    df_sorted = df.sort_values("Date")
    n = len(df_sorted)
    split_idx = int(n * ratio)
    df_train = df_sorted.iloc[:split_idx].copy()
    df_test = df_sorted.iloc[split_idx:].copy()
    return df_train, df_test


def run_training(verbose: bool = True) -> dict:
    """
    Full training pipeline.

    Returns
    -------
    results : dict with keys = model names, values = metric dicts
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if verbose:
        print("Loading processed data...")
    df = load_processed()

    if verbose:
        print("Running feature engineering...")
    df_eng, feat_cols, encoders = run_feature_engineering(df, POLLUTANT_COLS)

    # Chronological split
    df_train, df_test = chronological_split(df_eng)
    if verbose:
        print(f"  Train: {df_train['Date'].min().date()} to {df_train['Date'].max().date()}  ({len(df_train)} rows)")
        print(f"  Test : {df_test['Date'].min().date()} to {df_test['Date'].max().date()}  ({len(df_test)} rows)")

    X_train = df_train[feat_cols].values
    y_train = df_train[TARGET].values
    X_test = df_test[feat_cols].values
    y_test = df_test[TARGET].values

    all_metrics = {}
    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    models = get_models()

    for name, pipeline in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)   # no negative PM2.5

        metrics = evaluate_model(y_test, y_pred, name)
        all_metrics[name] = metrics

        if verbose:
            print(f"  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  R²={metrics['R2']:.4f}")

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_name = name
            best_pipeline = pipeline

    if verbose:
        print(f"\nBest model: {best_name}  (RMSE={best_rmse:.2f})")

    # Save best model + metadata
    y_pred_best = best_pipeline.predict(X_test)
    y_pred_best = np.clip(y_pred_best, 0, None)

    artefact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "feat_cols": feat_cols,
        "encoders": encoders,
        "metrics": all_metrics[best_name],
        "all_metrics": all_metrics,
        "train_date_range": (
            df_train["Date"].min().isoformat(),
            df_train["Date"].max().isoformat(),
        ),
        "test_date_range": (
            df_test["Date"].min().isoformat(),
            df_test["Date"].max().isoformat(),
        ),
        "y_test": y_test.tolist(),
        "y_pred_best": y_pred_best.tolist(),
        "test_dates": df_test["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "test_cities": df_test["City"].tolist(),
    }
    joblib.dump(artefact, MODEL_PATH)
    if verbose:
        print(f"  Saved to {MODEL_PATH}")

    # Also save JSON metrics for easy reading
    metrics_json = {
        k: {mk: float(mv) for mk, mv in v.items() if isinstance(mv, float)}
        for k, v in all_metrics.items()
    }
    metrics_json["best_model"] = best_name
    metrics_json["train_date_range"] = artefact["train_date_range"]
    metrics_json["test_date_range"] = artefact["test_date_range"]
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics_json, fh, indent=2)
    if verbose:
        print(f"  Metrics saved to {METRICS_PATH}")

    return all_metrics


def load_artefact() -> dict:
    """Load saved model artefact from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    run_training(verbose=True)
