import streamlit as st
import numpy as np
import pandas as pd
import joblib
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import matplotlib.pyplot as plt
import pydeck as pdk
import warnings, traceback, time

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Selected high corner frequency")

# ---- Load model & preprocessing ----
try:
    model = joblib.load("artifacts/xgb_eew_final.joblib")
    preproc = joblib.load("artifacts/preproc_objects.joblib")
    scaler = preproc["scaler"]
    imputer = preproc["imputer"]
    selector = preproc["selector"]
    p_wave_features = preproc["p_wave_features"]
except Exception as e:
    st.error(f"❌ Could not load model/preprocessing: {e}")
    st.stop()

client = None

# ---- Page Config & Style ----
st.set_page_config(page_title="EEW PGA Dashboard", layout="wide", page_icon="🌍")
st.markdown("""
<style>
/* Dark mode background */
body, .stApp {
    background-color: #121212;
    color: #ffffff;
}
/* Header */
.big-title { font-size: 32px; font-weight:700; margin-bottom:4px; color:#fff; }
.subheader { color:#ccc; font-size:15px; margin-bottom:10px; }
/* Divider */
.divider { border-bottom: 1px solid rgba(255,255,255,0.06); margin: 12px 0px; }
/* Badge / metric */
.badge { display:inline-block; padding:6px 12px; border-radius:18px; font-size:13px; font-weight:600; color:white; margin-top:6px; }
/* Group card */
.group-card { border-radius:10px; padding:12px; margin:6px 0px; color:#ffffff !important; box-shadow: 0 6px 20px rgba(2,6,23,0.18); border: 1px solid rgba(0,0,0,0.08); background: linear-gradient(120deg,#2ecc71,#3498db); }
.group-title { font-weight:700; font-size:14px; margin-bottom:6px; color:#ffffff !important; }
.feature-row { display:flex; justify-content:space-between; align-items:center; padding:4px 0px; font-size:13px; }
.feature-name { color:#ffffff !important; font-weight:600; }
.feature-val { font-weight:800; color:#ffffff !important; margin-left:8px; font-size:14px; }
.small-muted { color:#cfcfcf; font-size:13px; }

/* Reduce gap above gauge placeholder to pull it up slightly */
.gauge-container { margin-top: -8px; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="big-title">🌍 Earthquake Early Warning — PGA Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Fetch seismograms from IRIS, extract P-wave features, and predict Peak Ground Acceleration (PGA) using a trained XGBoost model.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---- Sidebar Inputs ----
st.sidebar.header("📡 Station Parameters")
network = st.sidebar.text_input("Network", "IU")
station = st.sidebar.text_input("Station Code", "ANMO")
date_selected = st.sidebar.date_input("Date", pd.to_datetime("2024-10-01"))
hour = st.sidebar.number_input("Hour (UTC)", 0, 23, 0)
st.sidebar.markdown("---")
year, month, day = date_selected.year, date_selected.month, date_selected.day

# ---- Historical examples ----
historical_examples = [
    {"name":"2011 Great Tōhoku", "year":2011, "pga_g":2.75},
    {"name":"1994 Northridge", "year":1994, "pga_g":1.82},
    {"name":"1999 Chi-Chi (TW)", "year":1999, "pga_g":1.00},
    {"name":"1995 Kobe", "year":1995, "pga_g":0.91},
    {"name":"2010 El Mayor–Cucapah","year":2010,"pga_g":0.58},
    {"name":"2008 Wenchuan", "year":2008, "pga_g":0.40},
    {"name":"2004 Sumatra-Andaman","year":2004,"pga_g":0.50},
    {"name":"2004 Parkfield","year":2004,"pga_g":0.05},
    {"name":"Example moderate event","year":2004,"pga_g":0.20},
    {"name":"Local light quake","year":2019,"pga_g":0.008},
]

# ---- Group definitions ----
groups = {
    "Time-based": {"keys":["durP","tauPd","tauPt"]},
    "Amplitude-based": {"keys":["PDd","PVd","PAd","PDt","PVt","PAt"]},
    "Derivative-based": {"keys":["ddt_PDd","ddt_PVd","ddt_PAd","ddt_PDt","ddt_PVt","ddt_PAt"]},
    "Energy-based": {"keys":["pkev12","pkev23"]}
}
group_colors = {
    "Time-based":"#2b8cff",
    "Amplitude-based":"#ff8c42",
    "Derivative-based":"#2ecc71",
    "Energy-based":"#9b59b6"
}

# ---- Helper functions ----
def round_to_bin(val, bin_size=0.05): return round(val / bin_size) * bin_size
def find_examples_near_bin(rounded_bin, tol=0.05): return [ev for ev in historical_examples if abs(ev["pga_g"] - rounded_bin) <= tol]
def pga_intensity_level(pga_g):
    if pga_g < 0.001: return ("Micro / No Shake", "#00b050")
    if pga_g < 0.2: return ("Light", "#ffc000")
    if pga_g < 0.25: return ("Moderate", "#ff8000")
    if pga_g < 0.3: return ("Strong", "#ff3333")
    if pga_g < 0.4: return ("Severe", "#8000ff")
    return ("Extreme", "#6600cc")

def p_wave_features_calc(window, dt):
    durP=len(window)*dt
    PDd = np.max(window)-np.min(window) if len(window)>0 else 0.0
    grad = np.gradient(window)/dt if len(window)>1 else np.array([0.0])
    PVd = np.max(np.abs(grad)) if len(grad)>0 else 0.0
    PAd = np.mean(np.abs(window)) if len(window)>0 else 0.0
    PDt = np.max(window) if len(window)>0 else 0.0
    PVt = np.max(grad) if len(grad)>0 else 0.0
    PAt = np.sqrt(np.mean(window**2)) if len(window)>0 else 0.0
    tauPd = durP/PDd if PDd!=0 else 0.0
    tauPt = durP/PDt if PDt!=0 else 0.0
    ddt = lambda x: np.mean(np.abs(np.gradient(x))) if len(x)>1 else 0.0
    return {
        "pkev12": np.sum(window**2)/len(window) if len(window)>0 else 0.0,
        "pkev23": np.sum(np.abs(window))/len(window) if len(window)>0 else 0.0,
        "durP": durP, "tauPd": tauPd, "tauPt": tauPt,
        "PDd": PDd, "PVd": PVd, "PAd": PAd, "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "ddt_PDd": ddt(window), "ddt_PVd": ddt(grad), "ddt_PAd": ddt(np.abs(window)),
        "ddt_PDt": ddt(np.maximum(window,0)), "ddt_PVt": ddt(grad), "ddt_PAt": ddt(window**2)
    }

def draw_gauge(ax, value_norm, color="#FF8000"):
    theta=np.linspace(-np.pi/2,np.pi/2,200)
    ax.plot(np.cos(theta),np.sin(theta),linewidth=6,color="#333")
    ax.plot(np.cos(theta),np.sin(theta),linewidth=4,color=color,alpha=0.9)
    for v in np.linspace(0,1,5):
        ang=(-90+v*180)*np.pi/180
        x0,y0=0.92*np.cos(ang),0.92*np.sin(ang)
        x1,y1=1.05*np.cos(ang),1.05*np.sin(ang)
        ax.plot([x0,x1],[y0,y1],color="#666",lw=1)
    angle=-90+value_norm*180
    ang_rad=angle*np.pi/180
    ax.plot([0,0.85*np.cos(ang_rad)],[0,0.85*np.sin(ang_rad)],color="#222",lw=3)
    ax.add_patch(plt.Circle((0,0),0.05,color="#222"))
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.2,1.2); ax.axis("off")

def detect_spikes_simple(data, dt, min_distance_s=0.5, threshold_factor=3.0, max_peaks=8):
    absd=np.abs(data)
    meanv=np.mean(absd)
    stdv=np.std(absd)
    thresh=meanv+threshold_factor*stdv
    if np.isnan(thresh) or thresh<=0: return []
    dist=max(1,int(round(min_distance_s/dt)))
    peaks=[]
    i=1; N=len(absd)
    while i<N-1:
        if absd[i]>thresh and absd[i]>absd[i-1] and absd[i]>=absd[i+1]:
            if len(peaks)==0 or (i-peaks[-1][0])>=dist:
                peaks.append((i,float(absd[i])))
                i+=dist
                continue
        i+=1
    peaks=sorted(peaks,key=lambda x:x[1],reverse=True)[:max_peaks]
    return peaks

def classify_delay(delta_s):
    if delta_s<=0: return "At/Before P"
    if delta_s<60: return "Likely S (local/regional)"
    if delta_s<600: return "Regional / long-period S"
    return "Teleseismic / separate event"

def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

# ---- Main workflow ----
if st.sidebar.button("🚀 Fetch & Predict"):
    try:
        st.toast("Initializing IRIS client...", icon="🔄")
        from obspy.clients.fdsn import Client
        if client is None: client = Client("IRIS")

        st.toast("Fetching waveform...", icon="🌐")
        start = UTCDateTime(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00:00")
        end = start + 2*3600
        stn = client.get_waveforms(network, station, "*", "BHZ", start, end)[0]

        st.toast("Preprocessing waveform...", icon="⚙️")
        stn.detrend("demean")
        stn.filter("bandpass", freqmin=0.5, freqmax=19.9)
        dt = stn.stats.delta

        # ---- Get station coordinates (robust) ----
        lat = None; lon = None
        try:
            inventory = client.get_stations(network=network, station=station, level="station")
            # robust extraction
            if hasattr(inventory, "networks") and len(inventory.networks) > 0:
                net = inventory.networks[0]
                if hasattr(net, "stations") and len(net.stations) > 0:
                    sta = net.stations[0]
                    lat = getattr(sta, "latitude", None)
                    lon = getattr(sta, "longitude", None)
            # fallback older indexing
            if lat is None or lon is None:
                try:
                    lat = inventory[0][0].latitude
                    lon = inventory[0][0].longitude
                except Exception:
                    pass
        except Exception as e:
            # don't fail the whole app for map issues
            st.sidebar.warning(f"Could not fetch station metadata: {e}")

        cft = classic_sta_lta(stn.data, max(1,int(round(1.0/dt))), max(1,int(round(10.0/dt))))
        trig = trigger_onset(cft,2.5,1.0)
        if len(trig)==0:
            st.warning("⚠️ No P-wave detected.")
        else:
            p_index = int(trig[0][0])
            win = max(1,int(round(2.0/dt)))
            p_window = stn.data[p_index:p_index+win]
            feats = p_wave_features_calc(p_window,dt)

            st.toast("Predicting PGA...", icon="🧠")
            X = np.log1p(pd.DataFrame([feats])[p_wave_features])
            X_sel = selector.transform(imputer.transform(scaler.transform(X)))
            pred_log = model.predict(X_sel)[0]
            pred_pga = np.expm1(pred_log)
            pred_g = pred_pga / 980.0
            rounded_bin = round_to_bin(pred_g,0.05)
            examples = find_examples_near_bin(rounded_bin,0.05)
            level_label, level_color = pga_intensity_level(pred_g)

            # Tabs: add a Map tab in the main dashboard
            overview, waveforms, features_tab, map_tab = st.tabs(["📊 Overview","📈 Waveforms","🧮 P-Wave Features","📍 Station Map"])

            # ---- Overview ----
            with overview:
                st.subheader("Predicted PGA Results")
                c1,c2=st.columns([2,1])
                with c1:
                    st.metric("Predicted PGA (cm/s²)",f"{pred_pga:.3f}")
                    st.metric("Predicted PGA (g)",f"{pred_g:.5f}")
                    st.markdown(f'<div class="badge" style="background-color:{level_color}">{level_label}</div>',unsafe_allow_html=True)
                with c2:
                    # put gauge in container with reduced top margin (CSS class)
                    placeholder=st.empty()
                    placeholder.markdown('<div class="gauge-container"></div>', unsafe_allow_html=True)
                    cap=0.25
                    value_norm=min(pred_g/cap,1.0)
                    steps=16
                    # animate needle quickly
                    for i in range(1,steps+1):
                        interp=(i/steps)*value_norm
                        figg,axg=plt.subplots(figsize=(3.2,2.4))
                        if interp<0.05: color="#00b050"
                        elif interp<0.2: color="#ffc000"
                        elif interp<0.4: color="#ff8000"
                        elif interp<0.6: color="#ff3333"
                        else: color="#8000ff"
                        draw_gauge(axg,interp,color=color)
                        # pull title up a bit using pad
                        axg.set_title("PGA Intensity", fontsize=10, pad=-8)
                        placeholder.pyplot(figg)
                        plt.close(figg)
                        time.sleep(0.008)
                    figg_fin,axg_fin=plt.subplots(figsize=(3.2,2.4))
                    draw_gauge(axg_fin,value_norm,color=level_color)
                    axg_fin.set_title("PGA Intensity", fontsize=10, pad=-8)
                    placeholder.pyplot(figg_fin)
                    plt.close(figg_fin)

            # ---- Waveforms ----
            with waveforms:
                st.subheader("Waveform Visualization")
                data = stn.data
                spikes = detect_spikes_simple(data,dt)
                spike_idx = None
                for idx,h in sorted(spikes,key=lambda x:x[0]):
                    if idx*dt > p_index*dt + 1.0:
                        spike_idx = idx
                        break
                col1,col2,col3 = st.columns(3)

                # col1: Spike Zoom (leftmost)
                with col1:
                    st.markdown("**Spike zoom**")
                    if spike_idx is None: st.info("No significant later spike detected.")
                    else:
                        spike_half=20.0
                        z_start_s=max(0.0,spike_idx*dt-spike_half)
                        z_end_s=min(len(data)*dt,spike_idx*dt+spike_half)
                        zs=int(z_start_s/dt)
                        ze=int(z_end_s/dt)
                        t_z_rel=(np.arange(zs,ze)*dt)-(spike_idx*dt)
                        fig_c,ax_c=plt.subplots(figsize=(5,3))
                        ax_c.plot(t_z_rel,data[zs:ze],lw=0.6,color="#1f77b4")
                        ax_c.axvline(0.0,color="r",ls="--")
                        p_rel=p_index*dt-(spike_idx*dt)
                        if z_start_s<=p_index*dt<=z_end_s:
                            ax_c.axvline(p_rel,color="orange",ls=":",alpha=0.7)
                        ax_c.set_xlabel("Seconds relative to spike")
                        ax_c.set_ylabel("Amplitude")
                        st.pyplot(fig_c)
                        plt.close(fig_c)

                # col2: Pre-P baseline (middle)
                with col2:
                    st.markdown("**Pre-P baseline**")
                    pre_sec=5.0; post_sec=2.0
                    p_time_s=p_index*dt
                    s_time=max(0.0,p_time_s-pre_sec)
                    e_time=min(len(data)*dt,p_time_s+post_sec)
                    s_idx=int(s_time/dt)
                    e_idx=int(e_time/dt)
                    t_rel=(np.arange(s_idx,e_idx)*dt)-p_time_s
                    fig_a,ax_a=plt.subplots(figsize=(5,3))
                    ax_a.plot(t_rel,data[s_idx:e_idx],lw=0.6,color="#1f77b4")
                    ax_a.axvline(0.0,color="r",ls="--")
                    ax_a.set_xlabel("Seconds relative to P pick")
                    ax_a.set_ylabel("Amplitude")
                    st.pyplot(fig_a)
                    plt.close(fig_a)

                # col3: Detected P-window (rightmost)
                with col3:
                    st.markdown("**Detected P-Window**")
                    t_pw=np.arange(len(p_window))*dt
                    fig_b,ax_b=plt.subplots(figsize=(5,3))
                    ax_b.plot(t_pw,p_window,lw=0.8,color="#2ca02c")
                    ax_b.set_xlabel("Seconds from P pick")
                    ax_b.set_ylabel("Amplitude")
                    ax_b.text(0.02,0.9,f"dur={feats['durP']:.3g}s PAt={feats['PAt']:.3g}", transform=ax_b.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
                    st.pyplot(fig_b)
                    plt.close(fig_b)

            # ---- Features tab ----
            with features_tab:
                st.subheader("P-Wave Features")
                try:
                    feats_df=pd.DataFrame.from_dict(feats,orient="index",columns=["Value"])
                    feats_df.index.name="Feature"
                except Exception:
                    feats_df=pd.DataFrame(columns=["Value"])
                group_names=list(groups.keys())
                cols=st.columns(len(group_names),gap="small")
                for i,gname in enumerate(group_names):
                    with cols[i]:
                        bg=group_colors.get(gname,"#2b8cff")
                        st.markdown(f"<div class='group-card' style='background:{bg};'>",unsafe_allow_html=True)
                        st.markdown(f"<div class='group-title'>{gname}</div>",unsafe_allow_html=True)
                        keys=[k for k in groups[gname]["keys"] if k in feats_df.index]
                        if not keys: st.markdown("<div style='color:#ffffff'>_No features_</div>",unsafe_allow_html=True)
                        else:
                            for k in keys:
                                v=float(feats_df.loc[k,"Value"])
                                st.markdown(f"<div class='feature-row'><div class='feature-name'>{k}</div><div class='feature-val'>{v:.5g}</div></div>",unsafe_allow_html=True)
                        st.markdown("</div>",unsafe_allow_html=True)

            # ---- Station Map tab (main dashboard) ----
            with map_tab:
                st.subheader("Station Location & PGA Intensity")
                if lat is None or lon is None:
                    st.warning("Station coordinates unavailable — cannot render map.")
                else:
                    # prepare a marker whose color/intensity is based on pred_g
                    color_rgb = hex_to_rgb(level_color)
                    # radius: scale with pred_g (clamped)
                    radius_m = max(5000, min(100000, int(20000 * (pred_g / 0.25) + 5000))) if pred_g>0 else 5000
                    map_df = pd.DataFrame([{"lat": lat, "lon": lon, "pga_g": pred_g, "name": station}])
                    # center view
                    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=5, pitch=0)
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position='[lon, lat]',
                        get_radius=radius_m,
                        radius_scale=1,
                        get_fill_color=color_rgb + [200],  # add alpha
                        pickable=True,
                        auto_highlight=True
                    )
                    tooltip = {"html": "<b>Station:</b> {name} <br/> <b>Pred PGA (g):</b> {pga_g}", "style": {"color": "white"}}
                    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
                    st.pydeck_chart(deck)

    except Exception as e:
        st.error(f"❌ Runtime error: {e}")
        st.text(traceback.format_exc())
        st.info("App recovered — adjust parameters and retry.")
