"""
evaluate.py
===========
Model evaluation utilities.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict with MAE, RMSE, R2, model_name
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "model_name": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def compare_models(all_metrics: dict) -> "pd.DataFrame":
    """
    Convert all_metrics dict to a sorted DataFrame for display.
    """
    import pandas as pd
    rows = []
    for name, m in all_metrics.items():
        if isinstance(m, dict) and "MAE" in m:
            rows.append({
                "Model": name,
                "MAE": round(m["MAE"], 3),
                "RMSE": round(m["RMSE"], 3),
                "R²": round(m["R2"], 4),
            })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
