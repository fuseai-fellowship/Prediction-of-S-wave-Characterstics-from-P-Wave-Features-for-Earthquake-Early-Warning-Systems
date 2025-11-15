```markdown
# 🌍 Earthquake Early Warning — PGA Predictor

Pipeline to train an ML model that predicts Peak Ground Acceleration (PGA) from P‑wave features, fetch seismograms from IRIS, extract P‑wave features, and serve predictions via an interactive Streamlit dashboard.

---

## 🚀 Table of Contents

- [Project Overview](#project-overview)
- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Requirements & Installation](#requirements--installation)
- [Quickstart](#quickstart)
  - [Train the Model](#train-the-model)
  - [Run the Streamlit App](#run-the-streamlit-app)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Artifacts & Outputs](#artifacts--outputs)
- [Development & Testing](#development--testing)
- [Docker](#docker)
- [Security & Secrets](#security--secrets)
- [Design Decisions & Notes](#design-decisions--notes)
- [Contributing](#contributing)

---

## 🧭 Project Overview

This project implements an Earthquake Early Warning (EEW) PGA predictor:

- Preprocesses a labeled dataset of P‑wave‑derived features.
- Trains an XGBoost regressor predicting PGA (model trained on log1p(PGA)).
- Saves model + preprocessing artifacts for inference.
- Fetches live waveforms from IRIS, detects P‑waves, extracts features, and predicts PGA in real‑time.
- Displays results in a Streamlit dashboard (waveform plots, P‑window zoom, predicted PGA, station map).

The codebase separates concerns: data IO, feature extraction, preprocessing, training, prediction, visualization, and UI.

---

## ✨ Highlights

- Reproducible training script (`scripts/run_train.py`) with default hyperparameters.
- Consistent feature extraction used in training and inference.
- Streamlit dashboard with waveform plots, PGA gauge, and station map.
- Dockerfile for containerized deployment.
- Clean package structure under `src/eew_pga` with unit test skeleton.

---

## 📁 Repository Structure

```
eew-pga-repo/
├── README.md
├── requirements.txt
├── Dockerfile
├── scripts/
│   └── run_train.py
├── app/
│   └── app.py
├── src/
│   └── eew_pga/
│       ├── config.py
│       ├── data_io.py
│       ├── features.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── predictor.py
│       ├── iris_client.py
│       └── viz.py
├── data/
└── artifacts/
```

---

## 🛠 Requirements & Installation

**Python:** 3.9+ recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies include:
`numpy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `obspy`, `streamlit`, `matplotlib`, `seaborn`, `tqdm`, `pytest`.

> Note: obspy and xgboost may require OS-level build tools or use of conda for simpler installation in some environments.

---

## ⚡ Quickstart

### Train the Model

Place your dataset CSV in `data/` and run:

```bash
python scripts/run_train.py --data data/EEW_features.csv --out artifacts
```

This will:
- Load and clean the dataset.
- Split into train/validation/test (log-transformed target).
- Fit preprocessing (RobustScaler, SimpleImputer, SelectKBest).
- Train XGBoost model.
- Save artifacts to `artifacts/`:
  - `xgb_eew_final.joblib`
  - `preproc_objects.joblib`

### Run the Streamlit App

```bash
streamlit run app/app.py
```

Open the URL printed by Streamlit (default: `http://localhost:8501`).

> Ensure model + preprocessing artifacts exist in `artifacts/` or set via env vars.

---

## ⚙️ Configuration & Environment Variables

- `EEW_ARTIFACTS_DIR` — directory to store artifacts (default: `artifacts/`)
- `EEW_MODEL_PATH` — path to model artifact (optional override)
- `EEW_PREPROC_PATH` — path to preprocessing artifact (optional override)

Example:

```bash
export EEW_MODEL_PATH="artifacts/xgb_eew_final.joblib"
export EEW_PREPROC_PATH="artifacts/preproc_objects.joblib"
```

---

## 📦 Artifacts & Outputs

- `xgb_eew_final.joblib` — trained XGBoost model
- `preproc_objects.joblib` — preprocessing objects (`scaler`, `imputer`, `selector`)
- Optional CSVs / figures saved under `artifacts/`

---

## 🧪 Development & Testing

Run tests:

```bash
pytest -q
```

Suggested tests:
- Feature extraction (`p_wave_features_calc`)
- Predictor loading & inference
- Data loading and preprocessing

---

## 🐳 Docker

Build:

```bash
docker build -t eew-pga .
```

Run (mount artifacts and set env vars):

```bash
docker run -p 8501:8501 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e EEW_MODEL_PATH=/app/artifacts/xgb_eew_final.joblib \
  -e EEW_PREPROC_PATH=/app/artifacts/preproc_objects.joblib \
  eew-pga
```

---

## 🔒 Security & Secrets

- Do not commit API keys/tokens to the repo.
- Configure secrets in environment variables at runtime.
- For public deployments, use a secrets manager.

---

## 📝 Design Decisions & Notes

- Using log1p on target and features reduces heteroscedasticity and stabilizes training.
- Preprocessing objects are persisted and reused for inference to avoid feature drift.
- The same P‑wave feature extraction functions are used in both training and inference for consistency.
- Modular structure allows easy replacement of model, hyperparameters, or UI.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feat/awesome`.
3. Add tests for new functionality.
4. Submit a PR with a clear description.
