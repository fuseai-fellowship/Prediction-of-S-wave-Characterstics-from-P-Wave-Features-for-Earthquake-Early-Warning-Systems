import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any

def train_xgb(X_train_sel, y_train_log, best_params=None, random_state: int = 42):
    """Train XGBoost regressor with best_params dict (or defaults)."""
    if best_params is None:
        best_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8}
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=int(best_params['n_estimators']),
        learning_rate=float(best_params['learning_rate']),
        max_depth=int(best_params['max_depth']),
        subsample=float(best_params['subsample']),
        colsample_bytree=float(best_params['colsample_bytree']),
        random_state=random_state,
        verbosity=0,
        tree_method='auto'
    )
    model.fit(X_train_sel, y_train_log)
    return model

def evaluate_model(model, X_sel, y_log, y_raw) -> Dict[str, Dict[str, float]]:
    preds_log = model.predict(X_sel)
    preds_raw = np.expm1(preds_log)
    return {
        'log': {
            'R2': float(r2_score(y_log, preds_log)),
            'MAE': float(mean_absolute_error(y_log, preds_log)),
            'RMSE': float(mean_squared_error(y_log, preds_log, squared=False))
        },
        'raw': {
            'R2': float(r2_score(y_raw, preds_raw)),
            'MAE': float(mean_absolute_error(y_raw, preds_raw)),
            'RMSE': float(mean_squared_error(y_raw, preds_raw, squared=False))
        }
    }

def save_model(model, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    return joblib.load(path)

def predict_from_features(model, preproc, feats_df):
    """feats_df: Pandas DataFrame with columns matching preproc['p_wave_features']"""
    X = np.log1p(feats_df)
    X_sel = preproc["selector"].transform(preproc["imputer"].transform(preproc["scaler"].transform(X)))
    pred_log = model.predict(X_sel)
    pred_raw = np.expm1(pred_log)
    return pred_log, pred_raw