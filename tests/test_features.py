import numpy as np
from src.features.extract_p_wave_features import p_wave_features_calc

def test_features_count():
    sig = np.hanning(200)
    feats = p_wave_features_calc(sig, dt=0.01)
    assert isinstance(feats, dict)
    assert len(feats) == 17
    assert 'PAt' in feats
