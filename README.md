# Prediction-of-S-wave-Characterstics-from-P-Wave-Features-for-Earthquake-Early-Warning-Systems

Draft Readme:

# 🌍 Real-Time P-Wave Feature Extraction & Earthquake PGA Prediction

This repository implements a complete pipeline for **real-time earthquake early warning (EEW)** using seismogram data. It extracts 17 P-wave features from real seismograms and predicts **Peak Ground Acceleration (PGA)** using trained **XGBoost** and **ANN** models.

---

## 🔹 Features

- **Real-time P-wave detection** using STA/LTA algorithm.
- **17 P-wave feature extraction** (amplitude, derivatives, energy proxies, duration metrics).
- **Pre-trained models**: XGBoost & ANN regression for PGA prediction.
- **Batch and real-time seismogram support**:
  - Upload seismogram files (`.mseed`, `.SAC`).
  - Fetch latest waveforms from **IRIS FDSN network** stations.
- **REST API** for seamless integration.
- Frontend for visualization and predictions.
- Handles errors, missing P-wave, and preprocessing automatically.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/eew-pwave-pga.git
cd eew-pwave-pga
