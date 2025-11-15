"""
Streamlit app entrypoint.

Requires artifacts/xgb_eew_final.joblib and artifacts/preproc_objects.joblib
(Or update the paths below.)
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import traceback
import matplotlib.pyplot as plt
import pydeck as pdk
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from src.eew.features import p_wave_features_calc, p_wave_features
from src.eew.fetcher import fetch_waveform, fetch_single_seismogram, DEFAULT_CLIENT
from src.eew.preproc import load_preproc, transform_features
from src.eew.model import load_model, predict_from_features

# artifact paths (update if needed)
MODEL_PATH = "artifacts/xgb_eew_final.joblib"
PREPROC_PATH = "artifacts/preproc_objects.joblib"

st.set_page_config(page_title="EEW PGA Dashboard", layout="wide", page_icon="🌍")

# load model & preproc
try:
    model = load_model(MODEL_PATH)
    preproc = load_preproc(PREPROC_PATH)
    scaler = preproc["scaler"]
    imputer = preproc["imputer"]
    selector = preproc["selector"]
    p_wave_features_list = preproc.get("p_wave_features", p_wave_features)
except Exception as e:
    st.error(f"Could not load model/preproc: {e}")
    st.stop()

# helper visuals & small util functions (kept compact)
def draw_gauge(ax, value_norm, color="#FF8000"):
    theta = np.linspace(-np.pi/2, np.pi/2, 200)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=6, color="#333")
    ax.plot(np.cos(theta), np.sin(theta), linewidth=4, color=color, alpha=0.9)
    angle = -90 + value_norm * 180
    ang_rad = angle * np.pi / 180
    ax.plot([0, 0.85 * np.cos(ang_rad)], [0, 0.85 * np.sin(ang_rad)], color="#222", lw=3)
    ax.add_patch(plt.Circle((0, 0), 0.05, color="#222"))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.2); ax.axis("off")

def pga_intensity_level(pga_g):
    if pga_g < 0.001: return ("Micro / No Shake", "#00b050")
    if pga_g < 0.2: return ("Light", "#ffc000")
    if pga_g < 0.25: return ("Moderate", "#ff8000")
    if pga_g < 0.3: return ("Strong", "#ff3333")
    if pga_g < 0.4: return ("Severe", "#8000ff")
    return ("Extreme", "#6600cc")

st.title("🌍 Earthquake Early Warning — PGA Predictor")
st.write("Fetch waveform from IRIS, extract P-wave features, and predict PGA with a trained XGBoost model.")

with st.sidebar:
    st.header("Station & Time")
    network = st.text_input("Network", "IU")
    station = st.text_input("Station", "ANMO")
    date_selected = st.date_input("Date", pd.to_datetime("2024-10-01"))
    hour = st.number_input("Hour (UTC)", 0, 23, 0)

if st.button("Fetch waveform & predict"):
    try:
        client = DEFAULT_CLIENT
        start = UTCDateTime(f"{date_selected.year}-{date_selected.month:02d}-{date_selected.day:02d}T{hour:02d}:00:00")
        end = start + 2 * 3600
        tr = fetch_waveform(network, station, start, end, client=client)
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=0.5, freqmax=19.9)
        dt = tr.stats.delta

        cft = classic_sta_lta(tr.data, max(1, int(round(1.0 / dt))), max(1, int(round(10.0 / dt))))
        trig = trigger_onset(cft, 2.5, 1.0)
        if len(trig) == 0:
            st.warning("No P-wave detected in waveform.")
        else:
            p_index = int(trig[0][0])
            win = max(1, int(round(2.0 / dt)))
            p_window = tr.data[p_index:p_index + win]
            feats = p_wave_features_calc(p_window, dt)
            feats_df = pd.DataFrame([feats])[p_wave_features_list]
            pred_log, pred_raw = predict_from_features(model, {"scaler": scaler, "imputer": imputer, "selector": selector}, feats_df)
            pred_g = float(pred_raw[0] / 980.0)
            level_label, level_color = pga_intensity_level(pred_g)

            st.metric("Predicted PGA (cm/s²)", f"{float(pred_raw[0]):.3f}")
            st.metric("Predicted PGA (g)", f"{pred_g:.5f}")
            st.markdown(f"<div style='padding:8px;border-radius:6px;background:{level_color};color:#fff;width:220px'>{level_label}</div>", unsafe_allow_html=True)

            # waveform preview
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            t = np.arange(0, tr.stats.npts) * dt
            axs[0].plot(t, tr.data, lw=0.6)
            axs[0].axvline(p_index * dt, color='red', ls='--')
            axs[1].plot(np.arange(len(p_window)) * dt, p_window, lw=0.8)
            axs[1].set_title("P-window (zoom)")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Extracted features (subset)")
            feats_short = {k: feats[k] for k in list(feats.keys())[:10]}
            st.table(pd.DataFrame.from_dict(feats_short, orient='index', columns=['value']).round(6))

    except Exception as e:
        st.error("Runtime error. See details below.")
        st.text(traceback.format_exc())