import matplotlib.pyplot as plt
import numpy as np

def plot_pred_vs_true(y_true, y_pred, title=None, figsize=(6,4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=8, alpha=0.6)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    ax.plot([mn, mx], [mn, mx], linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title)
    return fig