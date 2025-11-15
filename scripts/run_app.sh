#!/usr/bin/env bash
# Simple launcher for the streamlit app
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
# ensure streamlit runs from repo root so relative imports work
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

streamlit run app/app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
