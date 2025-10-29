from typing import Dict, Tuple
import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def quantile_score(y_true: np.ndarray, q_preds: np.ndarray, quantiles) -> float:
    Q = len(quantiles)
    y = np.expand_dims(y_true, axis=-1)
    e = y - q_preds
    loss = 0.0
    for i, q in enumerate(quantiles):
        loss += np.mean(np.maximum((q - 1) * e[..., i], q * e[..., i]))
    return float(loss / Q)

def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))
