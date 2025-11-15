import numpy as np
from src.eew_pga.features import p_wave_features_calc, p_wave_features

def test_p_wave_features_calc_basic():
    # synthetic sinusoid window
    dt = 0.01
    t = np.arange(0, 1.0, dt)
    window = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
    feats = p_wave_features_calc(window, dt)
    # ensure all expected keys exist
    for k in p_wave_features:
        assert k in feats
        assert isinstance(feats[k], float) or isinstance(feats[k], (int, float))
    # simple sanity checks
    assert feats["durP"] == pytest.approx(len(window) * dt, rel=1e-6)
    assert feats["pkev12"] > 0
