import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client("IRIS")

def fetch_trace(network="IU", station="ANMO", channel="BHZ", starttime: UTCDateTime = None, endtime: UTCDateTime = None):
    """
    Fetch a trace from IRIS FDSN. Returns an ObsPy Trace object.
    If starttime is None, it fetches the most recent 2 hours.
    """
    if starttime is None:
        endtime = UTCDateTime()
        starttime = endtime - 2*3600
    try:
        st = client.get_waveforms(network, station, "*", channel, starttime, endtime)
        if len(st) == 0:
            raise RuntimeError("No trace returned from IRIS")
        tr = st[0]
        return tr
    except Exception as e:
        raise RuntimeError(f"Error fetching trace: {e}")
