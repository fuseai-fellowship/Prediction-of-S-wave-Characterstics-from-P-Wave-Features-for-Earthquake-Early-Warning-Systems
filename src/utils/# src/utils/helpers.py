import os, random, numpy as np
import joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_joblib(obj, path):
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)
