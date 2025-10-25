"""
Load models and provide prediction helpers used by FastAPI endpoints.
"""
import os, joblib, pickle, numpy as np, torch
from src.features.extract_p_wave_features import p_wave_features_calc, window_from_trace, P_WAVE_FEATURE_ORDER
from src.data.preprocess import detrend_and_filter
from src.data.fetch_seismogram import fetch_trace
from obspy import Trace

MODELS_DIR = os.environ.get('MODELS_DIR', 'models')

def load_artifacts(models_dir=MODELS_DIR):
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    selector = joblib.load(os.path.join(models_dir, 'selector.joblib'))
    with open(os.path.join(models_dir, 'xgb_model.pkl'), 'rb') as f:
        import pickle
        xgb_model = pickle.load(f)
    art = torch.load(os.path.join(models_dir, 'torch_ann.pt'), map_location='cpu')
    # build ANN model architecture to match saved state
    from torch import nn
    layers = []
    in_size = art['input_size']
    for h in art['hidden_sizes']:
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(art.get('dropout', 0.2)))
        in_size = h
    layers.append(nn.Linear(in_size, 1))
    ann_model = nn.Sequential(*layers)
    ann_model.load_state_dict(art['model_state_dict'])
    ann_model.eval()
    return scaler, selector, xgb_model, ann_model

def features_to_input(feature_dict, scaler, selector, order=P_WAVE_FEATURE_ORDER):
    x = np.array([feature_dict.get(k, 0.0) for k in order], dtype=float).reshape(1,-1)
    x = np.nan_to_num(x, nan=0.0)
    x_scaled = scaler.transform(x)
    x_sel = selector.transform(x_scaled)
    return x_sel

def predict_from_features(feature_dict, scaler, selector, xgb_model, ann_model):
    X = features_to_input(feature_dict, scaler, selector)
    xgb_log = float(xgb_model.predict(X)[0])
    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    ann_log = float(ann_model(Xt).detach().cpu().numpy().flatten()[0])
    ensemble_log = float(np.mean([xgb_log, ann_log]))
    return {
        "xgb_log": xgb_log,
        "ann_log": ann_log,
        "ensemble_log": ensemble_log,
        "xgb_raw": float(np.expm1(xgb_log)),
        "ann_raw": float(np.expm1(ann_log)),
        "ensemble_raw": float(np.expm1(ensemble_log))
    }

def predict_from_trace(trace, scaler, selector, xgb_model, ann_model, win_seconds=2.0):
    # preprocess trace
    trace = detrend_and_filter(trace)
    # detect p-index using STA/LTA
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    data = trace.data
    dt = trace.stats.delta
    try:
        cft = classic_sta_lta(data, int(1.0/dt), int(10.0/dt))
        trig = trigger_onset(cft, 2.5, 1.0)
        if len(trig) == 0:
            p_index = None
        else:
            p_index = int(trig[0][0])
    except Exception:
        p_index = int(np.argmax(np.abs(data)))

    if p_index is None:
        return None

    p_window, dt = window_from_trace(trace, p_index, win_seconds)
    if len(p_window) < 5:
        return None

    features = p_wave_features_calc(p_window, dt)
    preds = predict_from_features(features, scaler, selector, xgb_model, ann_model)
    preds['features'] = features
    preds['p_index'] = int(p_index)
    preds['sampling_rate'] = float(trace.stats.sampling_rate)
    # include p_window for plotting (convert to python floats)
    preds['p_window'] = [float(x) for x in p_window.tolist()]
    return preds
