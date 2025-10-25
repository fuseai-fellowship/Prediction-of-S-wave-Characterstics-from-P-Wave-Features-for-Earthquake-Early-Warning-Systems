import numpy as np

P_WAVE_FEATURE_ORDER = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

def ddt(x):
    x = np.asarray(x)
    if x.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.gradient(x))))

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    """
    window : 1D numpy window (P-wave)
    dt : sampling interval in seconds
    Returns dict of 17 features in P_WAVE_FEATURE_ORDER.
    """
    if window is None or len(window) == 0:
        return {k: np.nan for k in P_WAVE_FEATURE_ORDER}

    durP = float(len(window) * dt)
    PDd = float(np.max(window) - np.min(window))
    grad = np.gradient(window) / dt
    PVd = float(np.max(np.abs(grad)))
    PAd = float(np.mean(np.abs(window)))
    PDt = float(np.max(window))
    PVt = float(np.max(grad))
    PAt = float(np.sqrt(np.mean(window ** 2)))
    tauPd = float(durP / PDd) if PDd != 0 else 0.0
    tauPt = float(durP / PDt) if PDt != 0 else 0.0

    ddt_PDd = ddt(window)
    ddt_PVd = ddt(grad)
    ddt_PAd = ddt(np.abs(window))
    ddt_PDt = ddt(np.maximum(window, 0))
    ddt_PVt = ddt(grad)
    ddt_PAt = ddt(window ** 2)

    pkev12 = float(np.sum(window ** 2) / len(window))
    pkev23 = float(np.sum(np.abs(window)) / len(window))

    return {
        "pkev12": pkev12, "pkev23": pkev23,
        "durP": durP, "tauPd": tauPd, "tauPt": tauPt,
        "PDd": PDd, "PVd": PVd, "PAd": PAd,
        "PDt": PDt, "PVt": PVt, "PAt": PAt,
        "ddt_PDd": ddt_PDd, "ddt_PVd": ddt_PVd,
        "ddt_PAd": ddt_PAd, "ddt_PDt": ddt_PDt,
        "ddt_PVt": ddt_PVt, "ddt_PAt": ddt_PAt
    }

def window_from_trace(trace, p_index: int, win_seconds: float = 2.0):
    """
    Extract P-window from ObsPy trace by sample index p_index.
    Returns (window_numpy, dt)
    """
    dt = trace.stats.delta
    win = int(win_seconds / dt)
    start = int(p_index)
    end = start + win
    data = trace.data
    if start < 0: start = 0
    if end > len(data): end = len(data)
    return data[start:end].astype(float), float(dt)
