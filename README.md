# SmartContainerRiskManagement
An AI/ML system that predicts the risk level of shipping containers (Critical, Low Risk, or Clear) to help customs authorities detect suspicious or high-risk shipments quickly. 

# 🚢 SmartContainer Risk Engine v2

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?style=flat-square&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=flat-square&logo=flask&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.9998-00c97a?style=flat-square)
![Critical F1](https://img.shields.io/badge/Critical%20F1-1.0000-e53935?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

**AI/ML container shipment risk prediction engine built for [HackaMINeD-2026](https://www.hackamiined.nl/) · INTECH problem statement.**

Predicts whether a customs container is **Critical**, **Low Risk**, or **Clear** — and explains why.

[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#-model-architecture) · [Dashboard](#-dashboard) · [API](#-rest-api) · [Results](#-performance)

</div>

---

## 📋 Overview

Customs inspectors have limited time. SmartContainer automates the initial risk triage of container shipments — flagging those most likely to contain misdeclared goods, high-risk cargo, or suspicious trade patterns — so human inspectors can focus their efforts where it matters.

The system:
- **Classifies** every incoming container as Critical / Low Risk / Clear
- **Detects anomalies** using a hybrid ML + rule-based approach
- **Explains** each prediction in plain language
- **Visualises** the full batch on a live dashboard with country risk maps, port rankings, and more
- **Exposes** a REST API for integration with live customs systems

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎯 3-class prediction | Clear · Low Risk · Critical — always exactly 3 classes |
| 🔍 Hybrid anomaly detection | IsolationForest + rule-based composite score |
| 📊 22 engineered features | Weight, value, dwell time, country risk, HS codes, importer profile + more |
| 🗺️ Live dashboard | 5-page Flask UI with world map, charts, alert feed, anomaly watchlist |
| ⚡ REST API | FastAPI with `/predict/live` that auto-updates the dashboard |
| 📝 Human explanations | 1–2 line plain-language reason per container |
| 🌙 Dark / Light mode | Full theme toggle throughout dashboard |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/smartcontainer-risk-engine.git
cd smartcontainer-risk-engine
pip install -r requirements.txt
```

### 2. Run the full pipeline (train → predict)

```bash
python run_pipeline.py
```

This trains the model on `data/Historical_Data.csv`, then scores `data/Real_Time_Data.csv`. Outputs land in `outputs/`.

### 3. Launch the dashboard

```bash
python dashboard/app.py
```

Open **http://localhost:5000** in your browser.

### 4. Launch the REST API *(optional)*

```bash
uvicorn api.main:app --port 8000
```

Interactive docs at **http://localhost:8000/docs**.

---

## 📁 Project Structure

```
SmartContainer_v2/
│
├── data/
│   ├── Historical_Data.csv          # Training data — 54,000 rows
│   └── Real_Time_Data.csv           # Inference batch — 8,481 rows
│
├── src/
│   ├── feature_engineering.py       # 22-feature pipeline + column normalisation
│   ├── train.py                     # Model training + evaluation
│   ├── predict.py                   # Batch prediction engine
│   └── explainability.py            # Plain-language explanation generator
│
├── models/
│   └── risk_engine_v2.pkl           # Trained artifact (model + profiles + metrics)
│
├── outputs/
│   ├── predictions.csv              # ← Primary deliverable
│   ├── current.txt                  # Pointer to latest predictions file
│   ├── evaluation_report.txt        # Validation metrics
│   ├── evaluation_report_realtime.txt
│   ├── feature_importance.png
│   ├── confusion_matrix_train.png
│   ├── confusion_matrix_realtime.png
│   └── risk_distribution.png
│
├── dashboard/
│   ├── app.py                       # Flask server (5 API routes + file downloads)
│   └── templates/index.html         # Single-file dashboard UI
│
├── api/
│   └── main.py                      # FastAPI REST endpoint
│
├── requirements.txt
├── run_pipeline.py                  # One-command train + predict
└── README.md
```

---

## 🧠 Model Architecture

The model is a **two-stage hybrid** pipeline:

```
Raw CSV
   │
   ├─► feature_engineering.py
   │      22 features extracted (weight, value, dwell, country risk, HS, ...)
   │      ↓
   ├─► IsolationForest (n_estimators=300, contamination=0.05)
   │      Decision score → iso_anomaly_score  ← fed as feature into step 2
   │      Anomaly_Flag   → binary anomaly label
   │      ↓
   └─► HistGradientBoostingClassifier (max_iter=500, max_depth=6, lr=0.05)
          Class weights: Clear=1 · Low Risk=4 · Critical=50
          Output: Clear / Low Risk / Critical + probabilities
```

### Why HistGradientBoostingClassifier?

- Native support for missing values — no imputation required
- XGBoost-equivalent speed and accuracy
- Built-in class weight support for severe class imbalance (0.9% Critical)
- Permutation importances computed post-training and stored in the model artifact

### Why Isolation Forest as a feature, not just a flag?

Instead of treating the anomaly detector as a separate post-processing step, the raw **decision score** (`iso_anomaly_score`) is fed directly into the classifier as a 22nd feature. This means the main model learns how to combine the anomaly signal with all other evidence — producing better-calibrated Critical predictions than either approach alone.

---

## 🔧 Feature Engineering

22 features are computed from the raw CSV by `src/feature_engineering.py`:

| Category | Features |
|---|---|
| **Weight signals** | `weight_discrepancy_pct`, `weight_discrepancy_raw`, `weight_over_declared` |
| **Value signals** | `value_per_kg`, `log_value_per_kg`, `log_declared_value`, `log_declared_weight` |
| **Dwell time** | `dwell_time`, `dwell_time_zscore`, `high_dwell` |
| **Declaration behaviour** | `declaration_hour`, `is_night_declaration`, `is_weekend_declared`, `declaration_month` |
| **Country risk** | `is_high_risk_country`, `origin_risk_rate` |
| **Cargo** | `hs_chapter`, `is_sensitive_hs` |
| **Importer** | `importer_risk_rate` |
| **Route** | `is_transit` |
| **Anomaly** | `composite_anomaly`, `iso_anomaly_score` |

**Top features by permutation importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `weight_discrepancy_pct` | 66.4% |
| 2 | `dwell_time` | 32.8% |
| 3 | `weight_over_declared` | 0.6% |
| 4 | `log_declared_weight` | 0.1% |
| 5–8 | importer risk, HS chapter, declaration month, weight discrepancy raw | < 0.1% each |

---

## 📊 Performance

Results on the held-out real-time batch (8,481 containers):

| Metric | Validation | Real-Time |
|---|---|---|
| **Macro F1** *(primary)* | **0.9975** | **0.9998** |
| F1 — Critical class | 0.9954 | **1.0000** |
| Recall — Critical | 0.9908 | **1.0000** |
| Precision — Critical | 1.0000 | 1.0000 |
| Weighted F1 | 0.9991 | 0.9998 |

### Confusion matrix (real-time batch)

```
               Predicted
               Clear   Low Risk   Critical
Actual Clear    6646          0          0
     Low Risk      2       1760          0
     Critical      0          0         73
```

Zero missed Critical containers.

### Prediction breakdown

| Class | Count | % |
|---|---|---|
| Clear | 6,648 | 78.4% |
| Low Risk | 1,760 | 20.8% |
| Critical | 73 | 0.9% |
| Anomalies (IsoForest) | 379 | 4.5% |

---

## 📦 Output Format

`outputs/predictions.csv` columns:

| Column | Type | Description |
|---|---|---|
| `Container_ID` | str | Unique shipment identifier |
| `Risk_Score` | float (0–100) | Continuous risk score |
| `Risk_Level` | str | `Critical` / `Low Risk` / `Clear` |
| `Anomaly_Flag` | int (0/1) | 1 = IsolationForest outlier |
| `P_Clear` | float (0–1) | Model probability for Clear |
| `P_Low_Risk` | float (0–1) | Model probability for Low Risk |
| `P_Critical` | float (0–1) | Model probability for Critical |
| `Explanation_Summary` | str | Plain-language reason |
| `Origin_Country` | str | ISO 2-letter country code |
| `Destination_Port` | str | Port identifier |
| `Trade_Regime` | str | Import / Export / Transit |
| `Shipping_Line` | str | Carrier identifier |

---

## 🖥️ Dashboard

A 5-page live dashboard served by Flask at `http://localhost:5000`.

| Page | Contents |
|---|---|
| **Overview** | KPI cards · Model metrics · World risk map · Donut chart · Risk level bars · Port rankings · P(Critical) histogram |
| **Analytics** | Trade regime breakdown · Shipping line risk bars · Full country risk table |
| **🚨 Alerts** | Cards for every Critical container with score, flags, explanation |
| **⚠ Anomaly** | Hybrid anomaly breakdown (4 quadrants) · Clear+Anomaly soft watch list |
| **📋 Records** | Searchable/filterable table of all containers · Filtered CSV download |

### Dashboard features

- **World map** — Countries colour-coded by avg risk score; red glow for countries with > 10 Critical containers; hover tooltip with full stats
- **P(Critical) chart** — Adaptive bins + log scale to handle the bimodal distribution (99.1% of containers have P=0, 0.9% have P=1)
- **Live filter** — Records page fetches per-level from the server so all 6,648 Clear containers are always visible
- **Dark / Light mode** — Full CSS custom property theme toggle
- **CSV export** — Download filtered subset (All / Critical / Low Risk / Clear) directly from the table

---

## ⚡ REST API

FastAPI server at `http://localhost:8000`. Interactive docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Model status + version |
| `POST` | `/predict/batch` | Upload CSV → download `predictions.csv` |
| `POST` | `/predict/json` | Upload CSV → JSON `{summary, predictions[]}` |
| `POST` | `/predict/live` | Upload CSV → saves to `outputs/` + updates dashboard |
| `POST` | `/predict/live?format=csv` | Same as above + CSV download response |
| `GET` | `/model/info` | Model metadata, training metrics, top features |

### Example — score a CSV via curl

```bash
curl -X POST http://localhost:8000/predict/json \
  -F "file=@data/Real_Time_Data.csv" | python3 -m json.tool
```

### Example — score and push to dashboard

```bash
curl -X POST http://localhost:8000/predict/live \
  -F "file=@your_new_batch.csv"
# Dashboard at localhost:5000 will show new data on next refresh
```

---

## 🔬 Anomaly Detection — 4 Cases

The hybrid system produces four distinct container states, each with a tailored explanation:

| State | Flag | Risk Level | Meaning |
|---|---|---|---|
| **Critical + Anomaly** | 1 | Critical | High-risk AND statistically unusual — top priority for physical inspection |
| **Critical, no Anomaly** | 0 | Critical | High-risk via known fraud patterns (weight mismatch, sanctions country, etc.) |
| **Clear + Anomaly** | 1 | Clear | Model says clear but IsoForest flags as unusual — soft watch list |
| **Clear, no Anomaly** | 0 | Clear | Genuinely normal shipment |

---

## ⚙️ Requirements

```
scikit-learn >= 1.3.0
pandas       >= 1.5.0
numpy        >= 1.24.0
matplotlib   >= 3.6.0
seaborn      >= 0.12.0
flask        >= 2.3.0
fastapi      >= 0.104.0
uvicorn      >= 0.24.0
python-multipart >= 0.0.6
```

Python **3.10+** recommended.

---

## 🗺️ Roadmap

- [ ] Add SHAP explainability waterfall plots per container
- [ ] PostgreSQL backend for multi-user dashboard
- [ ] Docker + `docker-compose` for one-command deploy
- [ ] Streaming inference via Kafka topic
- [ ] Threshold calibration UI for inspector sensitivity tuning

---

## 👥 Team

Built for **HackaMINeD-2026** · INTECH customs risk problem statement.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>SmartContainer Risk Engine v2 · HistGradientBoosting + IsolationForest · Macro F1 = 0.9998</sub>
</div>
