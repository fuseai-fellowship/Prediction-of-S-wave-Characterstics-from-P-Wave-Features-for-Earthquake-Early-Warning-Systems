import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def fit_preproc(X_train, y_train_log, k='all'):
    scaler = RobustScaler().fit(X_train)
    imputer = SimpleImputer(strategy='mean').fit(X_train)
    selector = SelectKBest(score_func=f_regression, k=k).fit(X_train, y_train_log)
    return {"scaler": scaler, "imputer": imputer, "selector": selector}

def transform_features(preproc, X):
    scaler = preproc["scaler"]
    imputer = preproc["imputer"]
    selector = preproc["selector"]
    return selector.transform(imputer.transform(scaler.transform(X)))

def save_preproc(preproc, p_wave_features, path):
    payload = {"scaler": preproc["scaler"], "imputer": preproc["imputer"], "selector": preproc["selector"], "p_wave_features": p_wave_features}
    joblib.dump(payload, path)

def load_preproc(path):
    return joblib.load(path)