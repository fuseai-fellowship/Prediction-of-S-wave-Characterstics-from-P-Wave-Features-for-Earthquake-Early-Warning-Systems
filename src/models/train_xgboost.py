import os
import argparse
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

FEATURES = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

DEFAULT_XGB_PARAMS = {
    "n_estimators": 776,
    "learning_rate": 0.010590433420511285,
    "max_depth": 6,
    "subsample": 0.666852461341688,
    "colsample_bytree": 0.8724127328229327
}

def load_params(json_path):
    if json_path is None:
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

def train_xgb(csv_path, out_dir, xgb_params=None, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    y_raw = df['PGA']
    y = np.log1p(y_raw)

    # Preprocessing: impute -> scale -> selector (k='all' preserves features order)
    imputer = SimpleImputer(strategy='mean')
    scaler = RobustScaler()
    selector = SelectKBest(score_func=f_regression, k='all')

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    X_sel = selector.fit_transform(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=test_size, random_state=random_state)

    params = DEFAULT_XGB_PARAMS.copy()
    if xgb_params:
        params.update(xgb_params)
    # Ensure required params
    params.update({"objective": "reg:squarederror", "random_state": random_state})

    model = xgb.XGBRegressor(**params)

    # Try to use early stopping if supported by the installed xgboost version
    try:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
    except TypeError:
        model.fit(X_train, y_train)

    # Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    joblib.dump(selector, os.path.join(out_dir, 'selector.joblib'))
    with open(os.path.join(out_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    preds = model.predict(X_test)
    print("✅ XGBoost: test MAE (log):", mean_absolute_error(y_test, preds))
    print("Saved XGBoost artifacts to:", out_dir)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to features CSV (with PGA column).")
    parser.add_argument("--out", default="models", help="Output directory to save artifacts.")
    parser.add_argument("--params", default=None, help="Optional JSON path with xgboost params to override defaults.")
    args = parser.parse_args()

    xgb_params = load_params(args.params) if args.params else None
    train_xgb(args.csv, args.out, xgb_params)
