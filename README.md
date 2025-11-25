# 🌍 Earthquake Early Warning — PGA Predictor

## Domain Knowledge 🌎

Earthquake Early Warning (EEW) systems aim to detect the initial seismic P-waves, which travel faster but are less destructive, to provide advance notice before the arrival of more damaging S-waves and surface waves. Peak Ground Acceleration (PGA) is a key metric representing the maximum ground motion during an earthquake, directly related to potential structural damage. By extracting features from P-wave segments—such as amplitude, frequency content, and energy—machine learning models can rapidly estimate PGA, enabling timely alerts for critical infrastructure, transportation systems, and public safety interventions.

## Abstract 🔍

Pipeline to train an ML model that predicts Peak Ground Acceleration (PGA) from P‑wave features, fetch seismograms from IRIS, use STA/LTA algorithm to detect P Wave onset, extract P‑wave features, and serve real‑time predictions via an interactive Streamlit dashboard.


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
- Trains an XGBoost regressor predicting PGA.
- Saves model + preprocessing artifacts for inference.
- Fetches live waveforms from IRIS, detects P‑waves, extracts features, and predicts PGA in (near) real time.
- Displays results in a Streamlit dashboard (waveform plots, P‑window zoom, predicted PGA, station map).

## Model Performance 🚀

<img width="1157" height="378" alt="image" src="https://github.com/user-attachments/assets/d2fba919-9535-475e-96ea-893b3a34e08b" />


---

---

## 📁 Repository Structure

Top-level

```
.
├── .gitignore
├── artifacts/   
├── data/                 # dataset files (optional)
├── Dockerfile
├── notebooks/            # analysis notebooks
├── app.py                # original top-level Streamlit app
├── app/                  # app package / assets
├── requirements.txt
├── scripts/              # convenience scripts
├── src/                  # source code
└── tests/                # unit tests
```

---

## 🛠 Requirements & Installation

**Python:** 3.9+ recommended.

Create and activate virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OS notes:
- obspy and xgboost can require system packages (build tools, libxml2, etc.) or be installed more easily via conda/miniforge on some platforms. If you run into install errors, prefer a conda environment.

---

### Run the Streamlit App

Ensure artifacts exist (from training) then:

```bash
streamlit run app/app.py
```

Open the address printed by Streamlit (default: http://localhost:8501).

The app will:
- Fetch waveform(s) from IRIS (public FDSN).
- Detect P-wave with STA/LTA trigger.
- Extract P-window features and run the saved model to predict PGA.
- Show waveform plots, P-window zoom, features table and a simple PGA gauge.

---

## ⚙️ Configuration & Environment Variables

- `EEW_ARTIFACTS_DIR` — directory for artifacts (default: `artifacts/`)
- `EEW_MODEL_PATH` — override full model path (optional)
- `EEW_PREPROC_PATH` — override preproc path (optional)

Example:

```bash
export EEW_MODEL_PATH="artifacts/xgb_eew_final.joblib"
export EEW_PREPROC_PATH="artifacts/preproc_objects.joblib"
```

---

## 📦 Artifacts & Outputs

- `xgb_eew_final.joblib` — trained XGBoost model (predicts log1p(PGA))
- `preproc_objects.joblib` — dict containing scaler, imputer, selector and feature list
- Optional CSVs / PNGs saved under `artifacts/` for single-sample predictions and visualizations

---

## 🧪 Development & Testing

Run the test suite:

```bash
pytest -q
```

Suggested tests to add/maintain:
- Feature extraction (consistency between training and inference).
- Preprocessing serialization/deserialization.
- Predictor end-to-end on a small synthetic example.
- IRIS fetcher — mocked tests for network calls.

---

## 🐳 Docker

Simple build and run (mount artifacts so container uses trained models):

```bash
docker build -t eew-pga .
docker run -p 8501:8501 \
  -v $(pwd)/artifacts:/app/artifacts \
  -e EEW_MODEL_PATH=/app/artifacts/xgb_eew_final.joblib \
  -e EEW_PREPROC_PATH=/app/artifacts/preproc_objects.joblib \
  eew-pga
```

Tip: If obspy or xgboost fail to install in the slim image, use a base image with required system libs or use conda-based images.

---

## 🔒 Security & Secrets

- Do NOT commit API keys, ngrok tokens, or other secrets to the repository.
- Configure any tokens via environment variables or a secret manager at runtime.

---

## 📝 Design Decisions & Notes

- Using log1p on both features and target stabilizes training and better handles heavy tails.
- Preprocessing objects (scaler, imputer, selector) are saved and reused to avoid feature-drift between train and inference.
- Inline P‑wave feature extractor is the canonical implementation — keep it identical between training and app.
- The Streamlit app currently uses public IRIS access; handle network failures gracefully in production.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b feat/your-feature`.
3. Add tests for new functionality.
4. Submit a PR with a clear description and rationale.

---

