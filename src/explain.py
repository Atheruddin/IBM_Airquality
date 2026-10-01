"""
explain.py
==========
Model explainability utilities.
Provides feature importance, permutation importance, and SHAP (if available).
"""

import os
import sys
import numpy as np
import pandas as pd

# SHAP is fully optional.
# The import can fail with ImportError (package absent) or OSError/other
# (e.g. llvmlite.dll blocked by Windows Application Control).
# We catch the base Exception so any failure sets SHAP_AVAILABLE = False
# and the rest of the module works without it.
_shap = None
SHAP_AVAILABLE: bool = False
try:
    import shap as _shap  # type: ignore[no-redef]
    SHAP_AVAILABLE = True
except Exception:           # ImportError, OSError, or anything llvmlite raises
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_feature_importance(pipeline, feat_cols: list[str]) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models or coefficients
    from linear models.

    Returns
    -------
    DataFrame with columns ['feature', 'importance'] sorted desc.
    """
    # Navigate through pipeline to get the final estimator
    final_step = pipeline.steps[-1][1]

    importance_vals = None

    # Tree-based: feature_importances_
    if hasattr(final_step, "feature_importances_"):
        importance_vals = final_step.feature_importances_

    # Linear models: coef_
    elif hasattr(final_step, "coef_"):
        importance_vals = np.abs(final_step.coef_)

    if importance_vals is None:
        return pd.DataFrame({"feature": feat_cols, "importance": [0] * len(feat_cols)})

    # Align length (imputer may not change feature count, but be safe)
    if len(importance_vals) != len(feat_cols):
        min_len = min(len(importance_vals), len(feat_cols))
        importance_vals = importance_vals[:min_len]
        feat_cols = feat_cols[:min_len]

    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": importance_vals,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df_imp


def permutation_importance(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance using sklearn.

    Returns
    -------
    DataFrame with columns ['feature', 'importance', 'std'] sorted desc.
    """
    from sklearn.inspection import permutation_importance as sk_perm_imp

    result = sk_perm_imp(
        pipeline, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )

    df = pd.DataFrame({
        "feature": feat_cols,
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def shap_explanation(
    pipeline,
    X: np.ndarray,
    feat_cols: list[str],
    max_samples: int = 500,
) -> tuple:
    """
    Compute SHAP values for tree-based or linear models.

    Returns
    -------
    (shap_values, explainer) or (None, None) if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        return None, None

    final_step = pipeline.steps[-1][1]

    # Apply pre-processing steps to X
    X_transformed = X.copy()
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            X_transformed = step.transform(X_transformed)

    sample = X_transformed[:max_samples]

    try:
        if hasattr(final_step, "feature_importances_"):
            # Tree explainer
            explainer = _shap.TreeExplainer(final_step)
            shap_values = explainer.shap_values(sample)
        else:
            # Generic
            explainer = _shap.Explainer(final_step, sample)
            shap_values = explainer(sample).values

        return shap_values, explainer
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None, None


def shap_summary_df(
    shap_values: np.ndarray,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Summarise mean absolute SHAP values per feature.

    Returns
    -------
    DataFrame with columns ['feature', 'mean_abs_shap'] sorted desc.
    """
    if shap_values is None:
        return pd.DataFrame()

    mean_abs = np.abs(shap_values).mean(axis=0)
    if len(mean_abs) != len(feat_cols):
        min_len = min(len(mean_abs), len(feat_cols))
        mean_abs = mean_abs[:min_len]
        feat_cols = feat_cols[:min_len]

    return pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    sys.path.insert(0, ROOT_DIR)
    from src.train import load_artefact
    from src.preprocessing import load_processed, POLLUTANT_COLS
    from src.feature_engineering import run_feature_engineering

    art = load_artefact()
    pipeline = art["pipeline"]
    feat_cols = art["feat_cols"]

    df = load_processed()
    df_eng, _, _ = run_feature_engineering(df, POLLUTANT_COLS)
    X = df_eng[feat_cols].values

    print("Feature importance:")
    df_fi = get_feature_importance(pipeline, feat_cols)
    print(df_fi.head(10))

    print("\nSHAP:")
    sv, exp = shap_explanation(pipeline, X, feat_cols)
    if sv is not None:
        df_shap = shap_summary_df(sv, feat_cols)
        print(df_shap.head(10))
    else:
        print("SHAP not computed.")
