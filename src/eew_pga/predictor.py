from .model import load_model, predict_from_features
from .preprocessing import load_preproc
from .config import ARTIFACT_DIR, MODEL_NAME, PREPROC_NAME
import os
import pandas as pd

def load_artifacts(artifact_dir=ARTIFACT_DIR):
    model_path = os.path.join(artifact_dir, MODEL_NAME)
    preproc_path = os.path.join(artifact_dir, PREPROC_NAME)
    model = load_model(model_path)
    preproc = load_preproc(preproc_path)
    return model, preproc

def predict_from_window(window, dt, model, preproc):
    from .features import p_wave_features_calc
    feats = p_wave_features_calc(window, dt)
    feats_df = pd.DataFrame([feats])[preproc.get("p_wave_features")]
    _, pred_raw = predict_from_features(model, preproc, feats_df)
    return float(pred_raw[0]), feats