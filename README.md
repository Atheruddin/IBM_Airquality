# AI-Based Air Quality Monitoring and PM2.5 Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Live demo (once deployed):**
[https://your-username-air-quality.streamlit.app](https://share.streamlit.io)
*(Replace this URL after Streamlit Community Cloud deployment)*

---

## Problem Statement

Air pollution — particularly fine particulate matter (PM2.5) — is one of the leading
environmental health threats worldwide. In India, rapid urbanisation and industrial growth
have significantly worsened air quality in many cities. Accurate PM2.5 prediction can help
city planners, public health authorities, and citizens make informed decisions.

This project builds a complete Machine Learning pipeline to analyse historical air quality
data from 26 Indian cities (2015–2020) and predict PM2.5 concentration using data-driven
models.

> **Important:** This system uses the historical **Air Quality Data in India (2015–2020)**
> dataset. All model predictions are **estimates based on historical patterns** and should
> not be used for real-time environmental decisions without independent validation.

---

## SDG Alignment

| Goal | How this project contributes |
|------|------------------------------|
| **SDG 11** — Sustainable Cities and Communities | Monitors urban air quality trends across 26 Indian cities, identifies pollution hotspots, and supports data-informed city planning |
| **SDG 13** — Climate Action | Tracks PM2.5 and co-pollutants (NO₂, SO₂, CO) linked to fossil fuel combustion; raises awareness of air-quality / climate co-benefits |

---

## Dataset

**Source:** [Air Quality Data in India (2015–2020) — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

### Files in this repository

| File | Size | Committed | Purpose |
|------|------|-----------|---------|
| `city_day.csv` | 2.45 MB | ✅ Yes | Raw daily city-level data — source for ML |
| `stations.csv` | 14 KB | ✅ Yes | Station reference metadata |
| `data/city_day_processed.csv` | 2.1 MB | ✅ Yes | Pre-cleaned data for the app |
| `models/best_pm25_model.joblib` | 0.92 MB | ✅ Yes | Trained XGBoost model |
| `models/all_model_metrics.json` | 1 KB | ✅ Yes | Model comparison metrics |
| `city_hour.csv` | 62.6 MB | ❌ No | Too large; not needed by app |
| `station_day.csv` | 8.2 MB | ❌ No | Station-level; not needed by app |
| `station_hour.csv` | 209.5 MB | ❌ No | Too large; not needed by app |

**Total committed data: ~5.5 MB** — well within GitHub's 100 MB file limit.

### Selected Modelling Dataset: `city_day.csv`

- 29,531 daily observations, 26 Indian cities, 2015–2020
- All required pollutant features present (PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene)
- Manageable 2.45 MB — no aggregation or sampling needed
- `city_hour.csv`/`station_hour.csv` are 63–210 MB and add no modelling value for daily PM2.5 prediction

---

## Leakage Prevention

**Excluded features** (computed from PM2.5 — would leak the target):

| Column | Reason excluded |
|--------|----------------|
| `AQI` | Computed from PM2.5 via Indian CPCB sub-index formula |
| `AQI_Bucket` | Categorical label derived from AQI, therefore from PM2.5 |

Using these as model inputs would make the model trivially accurate in training but
useless in production where AQI is not known before PM2.5 is measured.

---

## Preprocessing

1. Parse `Date` as datetime, sort by City + Date
2. Drop `AQI` and `AQI_Bucket` (data leakage)
3. Drop rows with missing PM2.5 (4,598 rows removed from 29,531)
4. Cap extreme PM2.5 > 1,500 µg/m³ at 999 (0 rows affected)
5. Remove duplicate rows (0 found)
6. Standardise City names (title-case, strip whitespace)
7. Coerce non-numeric pollutant values to NaN
8. Clip negative pollutant values to 0
9. Fill missing pollutants with city-level median
10. Fill remaining NaN with global column median

---

## Feature Engineering

**Time features:** year, month, day, day_of_week, week_of_year, season, is_weekend

**Lag features** (use only past observations — no leakage):

| Feature | Description |
|---------|-------------|
| `pm25_lag_1d` | PM2.5 from 1 day prior |
| `pm25_lag_3d` | PM2.5 from 3 days prior |
| `pm25_lag_7d` | PM2.5 from 7 days prior |
| `pm25_roll_3d` | 3-day rolling mean (shifted by 1) |
| `pm25_roll_7d` | 7-day rolling mean (shifted by 1) |

**Categorical encoding:** City and season are label-encoded.

---

## Machine Learning Models

Train/test split: **chronological 80/20** — earlier dates train, later dates test.

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 10.51 | 16.50 | 0.8334 |
| Random Forest | 9.10 | 15.48 | 0.8534 |
| Gradient Boosting | 9.18 | 14.98 | 0.8628 |
| **XGBoost** ✅ | **9.06** | **14.86** | **0.8648** |

- **Train:** 2015-01-08 → 2019-12-10
- **Test:** 2019-12-10 → 2020-07-01
- **Best model:** XGBoost (lowest RMSE on unseen test data)

---

## Explainability

- **Feature importance** from XGBoost's `feature_importances_` (tree split gains)
- **SHAP** — optional; loaded only if installed (not in default requirements)

Top predictors: `pm25_lag_1d` > `pm25_roll_3d` > `pm25_roll_7d` > PM10 > NO

**Caution:** Feature importance reflects predictive power — not causal relationships.

---

## Application Pages

| Page | Description |
|------|-------------|
| Overview | KPIs, PM2.5 trend, city ranking, seasonal breakdown |
| Air Quality Analytics | Interactive EDA with city / date / pollutant filters |
| PM2.5 Prediction | Single-point prediction using leakage-free inputs |
| Next-Day Forecast | Iterative lag-based forecast up to 14 days |
| Model Performance | Metrics, actual vs predicted, residuals, feature importance |
| Data Quality | File inventory, missing values, preprocessing log |

---

## Local Setup

### Requirements

- Python 3.10 or higher
- `city_day.csv` present in the project root (download from Kaggle if needed)

### Step 1 — Create virtual environment

```powershell
python -m venv .venv
```

### Step 2 — Activate (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

If blocked by execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip --cache-dir .cache/pip
python -m pip install -r requirements.txt --cache-dir .cache/pip
```

### Step 4 — Run preprocessing and training (first time only)

```powershell
python -m src.preprocessing
python -m src.train
```

This creates:
- `data/city_day_processed.csv`
- `models/best_pm25_model.joblib`
- `models/all_model_metrics.json`

### Step 5 — Launch Streamlit

```powershell
python -m streamlit run app.py
```

Opens at: **http://localhost:8501**

### Step 6 — Run verification script

```powershell
python _verify.py
```

Expected output ends with: `ALL CHECKS PASSED`

---

## Cloudflare Tunnel (temporary public URL)

Exposes your local Streamlit app to the internet for as long as both processes run.
**The URL is temporary.** Use Streamlit Community Cloud for persistent hosting.

### Step 1 — Download cloudflared (Windows)

Download the Windows 64-bit binary from:
https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe

Save it as:
```
.tools/cloudflared.exe
```

Do **not** install it system-wide or add it to PATH permanently.

### Step 2 — Start Streamlit (Terminal 1)

```powershell
python -m streamlit run app.py
```

### Step 3 — Start the tunnel (Terminal 2)

```powershell
.\.tools\cloudflared.exe tunnel --url http://localhost:8501
```

A temporary URL like `https://xxxx-xxxx.trycloudflare.com` appears.
The tunnel closes when either process stops.

---

## Streamlit Community Cloud Deployment

### What gets committed to GitHub

```
city_day.csv                    2.45 MB  ← raw data (small enough)
stations.csv                      14 KB  ← reference
data/city_day_processed.csv      2.1 MB  ← pre-cleaned
models/best_pm25_model.joblib   0.92 MB  ← trained model
models/all_model_metrics.json      1 KB  ← metrics
src/                                     ← all source modules
app.py                                   ← Streamlit entry point
requirements.txt                         ← pinned dependencies
.streamlit/config.toml                   ← Streamlit configuration
README.md
```

**Not committed:** `.venv/`, `.cache/`, `city_hour.csv`, `station_day.csv`,
`station_hour.csv`, `*.log`, `_verify.py`

### Step 1 — Initialize Git repository

Run these commands in the project folder
(`C:\Users\Ather\Downloads\Airquality`):

```powershell
git init
git branch -M main
```

### Step 2 — Stage and commit files

```powershell
git add app.py requirements.txt README.md
git add src/
git add .streamlit/
git add data/city_day_processed.csv
git add models/best_pm25_model.joblib
git add models/all_model_metrics.json
git add notebooks/
git add city_day.csv
git add stations.csv
git commit -m "Initial commit: AI Air Quality Monitoring & PM2.5 Prediction"
```

**Verify nothing unwanted is staged:**

```powershell
git status
```

Confirm `.venv/`, `.cache/`, `station_hour.csv`, `city_hour.csv`,
`station_day.csv` are NOT listed as staged.

### Step 3 — Create GitHub repository

**Option A — GitHub CLI (recommended):**

```bash
gh repo create air-quality-ml --public --source=. --remote=origin --push
```

**Option B — GitHub website:**

1. Go to https://github.com/new
2. Repository name: `air-quality-ml`
3. Visibility: Public
4. Do **not** initialise with README (you already have one)
5. Click **Create repository**
6. Copy the remote URL shown, then run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/air-quality-ml.git
git push -u origin main
```

### Step 4 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **New app**
4. Select:
   - **Repository:** `YOUR_USERNAME/air-quality-ml`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**

Streamlit Cloud will:
- Clone your repository
- Install `requirements.txt`
- Run `app.py`

The app will be live at:
`https://YOUR_USERNAME-air-quality-ml-app-XXXX.streamlit.app`

### Step 5 — Update deployment after changes

```powershell
git add -A
git commit -m "Update: <description>"
git push origin main
```

Streamlit Community Cloud auto-redeploys on every push to `main`.

---

## Project Structure

```
Airquality/
├── .streamlit/
│   └── config.toml              # Streamlit server / theme settings
├── .venv/                       # Virtual environment (not committed)
├── .cache/pip/                  # Local pip cache (not committed)
├── .tools/                      # Local binaries e.g. cloudflared (not committed)
├── data/
│   └── city_day_processed.csv   # Pre-cleaned dataset (committed)
├── models/
│   ├── best_pm25_model.joblib   # Trained XGBoost pipeline (committed)
│   └── all_model_metrics.json   # Model comparison metrics (committed)
├── notebooks/
│   └── air_quality_analysis.ipynb
├── outputs/
│   └── plots/                   # Generated charts (not committed)
├── src/
│   ├── preprocessing.py         # Data loading & cleaning
│   ├── feature_engineering.py   # Feature creation & encoding
│   ├── train.py                 # Model training pipeline
│   ├── evaluate.py              # Metrics utilities
│   └── explain.py               # Feature importance & optional SHAP
├── app.py                       # Streamlit application (6 pages)
├── requirements.txt             # Pinned dependencies
├── README.md
├── _verify.py                   # Local verification script
├── city_day.csv                 # Raw dataset — committed (2.45 MB)
├── stations.csv                 # Station reference — committed (14 KB)
├── city_hour.csv                # Large file — NOT committed
├── station_day.csv              # Large file — NOT committed
└── station_hour.csv             # Large file (210 MB) — NOT committed
```

---

## Limitations

1. **Historical data only:** Covers 2015–2020. Extrapolation beyond this range is unreliable.
2. **26 cities only:** Predictions for cities not in the training set are not supported.
3. **No real-time data:** The system does not fetch live sensor readings.
4. **Forecast uncertainty:** Next-day forecast uses city-median pollutant proxies; uncertainty grows with horizon.
5. **No meteorological features:** Wind speed, humidity, temperature are absent from the dataset but affect PM2.5.
6. **Causality:** Feature importance reflects model-learned correlations, not causal relationships.
7. **PM2.5 ≠ AQI:** The model predicts PM2.5 concentration (µg/m³). AQI requires multi-pollutant sub-index computation.
