"""
Streamlit app that uses artifacts in artifacts/ by default.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from src.eew_pga.predictor import load_artifacts, predict_from_window
from src.eew_pga.iris_client import fetch_waveform

st.set_page_config(page_title="EEW PGA Predictor", layout="wide", page_icon="🌍")
st.title("EEW PGA Predictor")

with st.sidebar:
    st.header("Station & Time")
    network = st.text_input("Network", "IU")
    station = st.text_input("Station", "ANMO")
    date_selected = st.date_input("Date", pd.to_datetime("2024-10-01"))
    hour = st.number_input("Hour (UTC)", 0, 23, 0)
    artifacts_dir = st.text_input("Artifacts dir", "artifacts")

try:
    model, preproc = load_artifacts(artifacts_dir)
except Exception as e:
    st.error(f"Could not load artifacts: {e}")
    st.stop()

if st.button("Fetch & Predict"):
    try:
        start = UTCDateTime(f"{date_selected.year}-{date_selected.month:02d}-{date_selected.day:02d}T{hour:02d}:00:00")
        end = start + 2*3600
        tr = fetch_waveform(network, station, start, end)
        dt = tr.stats.delta
        cft = classic_sta_lta(tr.data, max(1, int(round(1.0/dt))), max(1, int(round(10.0/dt))))
        trig = trigger_onset(cft, 2.5, 1.0)
        if len(trig) == 0:
            st.warning("No P-wave detected")
        else:
            p_index = int(trig[0][0])
            win = max(1, int(round(2.0/dt)))
            p_window = tr.data[p_index:p_index+win]
            pred_raw, feats = predict_from_window(p_window, dt, model, preproc)
            pred_g = pred_raw / 980.0
            st.metric("Predicted PGA (cm/s²)", f"{pred_raw:.3f}")
            st.metric("Predicted PGA (g)", f"{pred_g:.5f}")

            # quick plots
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6))
            t = np.arange(len(tr.data)) * dt
            ax1.plot(t, tr.data, lw=0.6); ax1.axvline(p_index*dt, color='r', ls='--')
            ax2.plot(np.arange(len(p_window))*dt, p_window, lw=0.8); ax2.set_title("P-window")
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Extracted features (first 12)")
            feats_df = pd.DataFrame.from_dict(feats, orient='index', columns=['value'])
            st.table(feats_df.head(12))
    except Exception as e:
        st.error(f"Runtime error: {e}")
        st.text(str(e))
