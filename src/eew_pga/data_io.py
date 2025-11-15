import os
import pandas as pd

def load_and_clean(path, p_wave_features):
    """Load CSV and perform minimal cleaning used by training script."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        df = pd.read_csv(path, skiprows=[1])
    except Exception:
        df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if col not in ['filename', 'date', 'time']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fill numeric NaNs with median
    df = df.fillna(df.median(numeric_only=True))
    # keep rows where all p-wave features > 0
    df = df[(df[p_wave_features] > 0).all(axis=1)]
    return df