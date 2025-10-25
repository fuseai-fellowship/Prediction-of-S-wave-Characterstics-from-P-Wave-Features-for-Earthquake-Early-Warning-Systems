import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }
