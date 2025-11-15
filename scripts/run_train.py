#!/usr/bin/env python3
"""
Convenience wrapper to run training from CLI.

Example:
  python scripts/run_train.py --data-path data/EEW_features_YYYY-MM-DD.csv --out-dir artifacts
"""
import argparse
import os
from src.eew_pga.train import main as train_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to training CSV")
    parser.add_argument("--out-dir", default="artifacts", help="Directory to save artifacts")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_main(args.data_path, args.out_dir)