import random, datetime
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from typing import Optional, Dict, Any

# default IRIS client (can be replaced)
DEFAULT_CLIENT = Client("IRIS")

def fetch_single_seismogram(
    client: Client = DEFAULT_CLIENT,
    stations: Optional[list] = None,
    net: str = "IU",
    year_choices: list = (2022, 2023, 2024),
    max_attempts: int = 8,
    channel: str = "BHZ",
    sta_timeout: float = 3600
) -> Optional[Dict[str, Any]]:
    """
    Attempt to fetch a single waveform with an identified P-window.
    Returns a dict with keys: trace, p_window, p_index, feats, meta
    or None if unsuccessful.
    """
    if stations is None:
        stations = ["ANMO", "COR", "MAJO", "KBL"]

    for attempt in range(max_attempts):
        try:
            yr = random.choice(year_choices)
            starttime = UTCDateTime(datetime.datetime(
                yr, random.randint(1, 12), random.randint(1, 25),
                random.randint(0, 21), 0, 0
            ))
            endtime = starttime + 2 * 3600
            st = None
            for station in stations:
                try:
                    st = client.get_waveforms(net, station, "*", channel, starttime, endtime)
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
            cft = classic_sta_lta(tr.data, max(1, int(round(1 / dt))), max(1, int(round(10 / dt))))
            trig = trigger_onset(cft, 2.5, 1.0)
            if len(trig) == 0:
                continue
            p_index = int(trig[0][0])
            win = int(round(2.0 / dt))
            p_window = tr.data[p_index: p_index + win]
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


def fetch_waveform(
    network: str,
    station: str,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    client: Client = DEFAULT_CLIENT,
    channel: str = "BHZ"
):
    """Fetch a waveform and return a processed Trace (detrended, filtered)."""
    st = client.get_waveforms(network, station, "*", channel, starttime, endtime)
    tr = st[0].copy()
    tr.detrend("demean")
    tr.filter("bandpass", freqmin=0.5, freqmax=19.9)
    return tr