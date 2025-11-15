import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from typing import Tuple, Dict, Any

def fit_preproc(X_train: pd.DataFrame, y_train_log: pd.Series, k='all'):
    """Fit scaler, imputer, selector on training data."""
    scaler = RobustScaler().fit(X_train)
    imputer = SimpleImputer(strategy='mean').fit(X_train)
    selector = SelectKBest(score_func=f_regression, k=k).fit(X_train, y_train_log)
    return {"scaler": scaler, "imputer": imputer, "selector": selector}

def transform_features(preproc: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Apply scaler -> imputer -> selector to a DataFrame X."""
    scaler = preproc["scaler"]
    imputer = preproc["imputer"]
    selector = preproc["selector"]
    arr = selector.transform(imputer.transform(scaler.transform(X)))
    return arr

def save_preproc(preproc: Dict[str, Any], p_wave_features: list, path: str):
    payload = {"scaler": preproc["scaler"], "imputer": preproc["imputer"], "selector": preproc["selector"], "p_wave_features": p_wave_features}
    joblib.dump(payload, path)

def load_preproc(path: str) -> Dict[str, Any]:
    return joblib.load(path)