"""
Training entrypoint that reproduces the notebook's training flow.
The function `main` is intentionally minimal so scripts/run_train.py can call it.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from .data_io import load_and_clean
from .preprocessing import fit_preproc, transform_features, save_preproc
from .model import train_xgb, evaluate_model, save_model
from .config import P_WAVE_FEATURES as p_wave_features
from .utils import set_seed

def main(data_path: str, out_dir: str):
    set_seed(4)
    df = load_and_clean(data_path, p_wave_features)
    X = df[p_wave_features].copy()
    y_raw = df['PGA'].copy()
    X = np.log1p(X)
    y_log = np.log1p(y_raw)

    # stratified splits using qcut bins
    y_bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y_bins))
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X.iloc[temp_idx], y_bins.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train_log, y_val_log, y_test_log = y_log.iloc[train_idx], y_log.iloc[val_idx], y_log.iloc[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y_raw.iloc[train_idx], y_raw.iloc[val_idx], y_raw.iloc[test_idx]

    preproc = fit_preproc(X_train, y_train_log, k='all')
    X_train_sel = transform_features(preproc, X_train)
    X_val_sel = transform_features(preproc, X_val)
    X_test_sel = transform_features(preproc, X_test)

    best_params = {
        'n_estimators': 776,
        'learning_rate': 0.010590433420511285,
        'max_depth': 6,
        'subsample': 0.666852461341688,
        'colsample_bytree': 0.8724127328229327
    }

    model = train_xgb(X_train_sel, y_train_log, best_params, random_state=42)

    val_metrics = evaluate_model(model, X_val_sel, y_val_log, y_val_raw)
    test_metrics = evaluate_model(model, X_test_sel, y_test_log, y_test_raw)

    print("Validation metrics (log):", val_metrics['log'])
    print("Validation metrics (raw):", val_metrics['raw'])
    print("Test metrics (log):", test_metrics['log'])
    print("Test metrics (raw):", test_metrics['raw'])

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "xgb_eew_final.joblib")
    preproc_path = os.path.join(out_dir, "preproc_objects.joblib")
    save_model(model, model_path)
    save_preproc(preproc, p_wave_features, preproc_path)
    print("Saved model ->", model_path)
    print("Saved preproc ->", preproc_path)