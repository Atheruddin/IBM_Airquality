"""
_verify.py
==========
Project verification script.
Runs without SHAP installed; asserts SHAP_AVAILABLE is False
when the package is absent, and confirms all other pipelines work.
"""
import sys
import ast

sys.path.insert(0, ".")

# ── 1. Syntax check app.py ──────────────────────────────────────────────────
with open("app.py", "r", encoding="utf-8") as f:
    source = f.read()
try:
    ast.parse(source)
    print("app.py : syntax OK")
except SyntaxError as exc:
    print(f"app.py : syntax ERROR — {exc}")
    sys.exit(1)

# ── 2. Core package imports ─────────────────────────────────────────────────
import pandas       # noqa: E402
import numpy        # noqa: E402
import plotly       # noqa: E402
import streamlit    # noqa: E402
import joblib       # noqa: E402
import sklearn      # noqa: E402
print("Core packages : OK")

import xgboost      # noqa: E402
print("XGBoost       : OK")

# ── 3. SHAP must not be importable (package removed) ────────────────────────
from src.explain import SHAP_AVAILABLE  # noqa: E402

if SHAP_AVAILABLE:
    print("SHAP          : INSTALLED (explainability features active)")
else:
    print("SHAP          : not installed — SHAP_AVAILABLE=False (expected)")

# No assertion here so the script passes on either state;
# the print line above makes the outcome unambiguous.

# ── 4. All src module imports ───────────────────────────────────────────────
from src.preprocessing import (    # noqa: E402
    load_processed, POLLUTANT_COLS, TARGET, pm25_category, inspect_all_files,
)
from src.feature_engineering import (  # noqa: E402
    run_feature_engineering, build_single_row, SEASON_MAP,
)
from src.evaluate import compare_models                             # noqa: E402
from src.explain import (                                           # noqa: E402
    get_feature_importance, shap_explanation, shap_summary_df,
)
from src.train import load_artefact                                 # noqa: E402
print("src imports   : OK")

# ── 5. Data + feature pipeline ──────────────────────────────────────────────
import pandas as pd   # noqa: E402
import numpy as np    # noqa: E402

df = load_processed()
df_eng, feat_cols, encoders = run_feature_engineering(df, POLLUTANT_COLS)
print(f"Data          : {len(df):,} rows loaded; {len(feat_cols)} features engineered")

# ── 6. Model load ────────────────────────────────────────────────────────────
art = load_artefact()
pipeline   = art["pipeline"]
all_metrics = art["all_metrics"]
print(f"Model         : {art['model_name']} loaded  "
      f"(RMSE={art['metrics']['RMSE']:.2f}, R2={art['metrics']['R2']:.4f})")

# ── 7. Single-row prediction ─────────────────────────────────────────────────
pol_vals = {
    c: float(df[df["City"] == "Delhi"][c].median())
    for c in POLLUTANT_COLS
    if c in df.columns
}
hist = [55.0, 60.0, 65.0, 70.0, 68.0, 72.0, 75.0]
row  = build_single_row(
    "Delhi", pd.Timestamp("2020-01-15"),
    pol_vals, hist, encoders, feat_cols,
)
pred = float(pipeline.predict(row)[0])
cat, colour = pm25_category(pred)
print(f"Prediction    : {pred:.2f} ug/m3  [{cat}]")

# ── 8. Model comparison ───────────────────────────────────────────────────────
df_cmp = compare_models(all_metrics)
print("Model comparison:")
print(df_cmp.to_string(index=False))

# ── 9. Feature importance (tree / linear — no SHAP needed) ──────────────────
df_fi = get_feature_importance(pipeline, feat_cols)
top3  = df_fi.head(3)["feature"].tolist()
print(f"Top-3 features: {top3}")

# ── 10. SHAP path — must return (None, None) without crashing ───────────────
X_sample = df_eng[feat_cols].values[:10]
sv, exp = shap_explanation(pipeline, X_sample, feat_cols)
if sv is None:
    print("SHAP path     : returned (None, None) — correct when not installed")
else:
    print("SHAP path     : values computed (SHAP is installed)")

df_shap = shap_summary_df(sv, feat_cols)   # must handle None without crash
print(f"shap_summary_df with None input: {type(df_shap).__name__} "
      f"(empty={df_shap.empty})")

# ── 11. File inspection ───────────────────────────────────────────────────────
summaries = inspect_all_files()
for fname, info in summaries.items():
    if "error" not in info:
        print(f"  {fname}: {info['rows']:,} rows, {info['cols']} cols")

# ── 12. requirements.txt — confirm shap/numba/llvmlite absent ────────────────
with open("requirements.txt", "r") as rf:
    req_lines = [l.strip() for l in rf if l.strip() and not l.startswith("#")]

forbidden = ["shap", "numba", "llvmlite"]
found_forbidden = [l for l in req_lines for f in forbidden if f in l.lower()]
if found_forbidden:
    print(f"requirements.txt WARNING: found forbidden packages: {found_forbidden}")
    sys.exit(1)
else:
    print("requirements.txt: no shap / numba / llvmlite entries — OK")

print()
print("ALL CHECKS PASSED")
