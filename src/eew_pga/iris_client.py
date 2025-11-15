import random, datetime
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from typing import Optional, Dict, Any

DEFAULT_CLIENT = Client("IRIS")

def fetch_waveform(network: str, station: str, starttime: UTCDateTime, endtime: UTCDateTime, client: Client = DEFAULT_CLIENT, channel: str = "BHZ"):
    st = client.get_waveforms(network, station, "*", channel, starttime, endtime)
    tr = st[0].copy()
    tr.detrend("demean")
    tr.filter("bandpass", freqmin=0.5, freqmax=19.9)
    return tr

def fetch_single_seismogram(client: Client = DEFAULT_CLIENT, stations=None, net='IU', year_choices=(2022,2023,2024), max_attempts=8):
    """Try to fetch a random example and locate a P-window (used in notebook)."""
    if stations is None:
        stations = ["ANMO","COR","MAJO","KBL"]
    for attempt in range(max_attempts):
        try:
            yr = random.choice(year_choices)
            starttime = UTCDateTime(datetime.datetime(
                yr, random.randint(1,12), random.randint(1,25),
                random.randint(0,21), 0, 0
            ))
            endtime = starttime + 2*3600
            st = None
            for station in stations:
                try:
                    st = client.get_waveforms(net, station, "*", "BHZ", starttime, endtime)
                    if st and len(st) > 0:
                        break
                except Exception:
                    continue
            if st is None or len(st) == 0:
                continue
            tr = st[0].copy()
            tr.detrend("demean")
            tr.filter("bandpass", freqmin=0.5, freqmax=20.0)
            dt = tr.stats.delta
            cft = classic_sta_lta(tr.data, max(1,int(round(1/dt))), max(1,int(round(10/dt))))
            trig = trigger_onset(cft, 2.5, 1.0)
            if len(trig) == 0:
                continue
            p_index = int(trig[0][0])
            win = int(round(2.0 / dt))
            p_window = tr.data[p_index : p_index + win]
            if len(p_window) < 10:
                continue
            from .features import p_wave_features_calc
            feats = p_wave_features_calc(p_window, dt)
            meta = {
                "station": tr.stats.station,
                "network": tr.stats.network,
                "starttime": str(starttime),
                "sampling_rate": tr.stats.sampling_rate,
                "p_index": p_index,
                "win_samples": len(p_window)
            }
            return {"trace": tr, "p_window": p_window, "p_index": p_index, "feats": feats, "meta": meta}
        except Exception:
            continue
    return None