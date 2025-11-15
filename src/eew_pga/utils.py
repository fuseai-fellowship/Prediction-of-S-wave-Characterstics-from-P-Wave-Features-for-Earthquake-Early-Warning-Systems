import os
import random
import numpy as np

def set_seed(seed=4):
    """Set seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        # torch optional
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)