import numpy as np

# canonical feature list used during training
p_wave_features = [
    'pkev12','pkev23','durP','tauPd','tauPt',
    'PDd','PVd','PAd','PDt','PVt','PAt',
    'ddt_PDd','ddt_PVd','ddt_PAd','ddt_PDt','ddt_PVt','ddt_PAt'
]

def p_wave_features_calc(window: np.ndarray, dt: float) -> dict:
    """Extract the same P-wave features used in training.

    Args:
        window: 1D numpy array with P-window amplitude samples
        dt: sampling interval (seconds)

    Returns:
        dict: feature_name -> numeric value
    """
    if window is None or len(window) == 0:
        return {k: 0.0 for k in p_wave_features}

    window = np.asarray(window, dtype=float)
    durP = len(window) * dt
    PDd = float(np.max(window) - np.min(window))
    grad = np.gradient(window) / dt if len(window) > 1 else np.array([0.0])
    PVd = float(np.max(np.abs(grad))) if len(grad) > 0 else 0.0
    PAd = float(np.mean(np.abs(window)))
    PDt = float(np.max(window))
    PVt = float(np.max(grad)) if len(grad) > 0 else 0.0
    PAt = float(np.sqrt(np.mean(window ** 2)))
    tauPd = durP / PDd if PDd != 0 else 0.0
    tauPt = durP / PDt if PDt != 0 else 0.0

    def ddt(x):
        x = np.asarray(x, dtype=float)
        return float(np.mean(np.abs(np.gradient(x)))) if len(x) > 1 else 0.0

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