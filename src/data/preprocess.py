from obspy import Trace
import numpy as np

def detrend_and_filter(trace, freqmin=0.5, freqmax=20.0):
    """
    Detrend and bandpass filter an ObsPy Trace in-place and return it.
    """
    trace.detrend("demean")
    # safe-guard: ensure enough samples for filter
    try:
        trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    except Exception:
        # keep trace as-is if filtering fails
        pass
    return trace

def trace_to_array(trace):
    """Return numpy array of trace data and dt (sampling interval)."""
    return trace.data.astype(float), float(trace.stats.delta)
