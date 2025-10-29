from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

def stl_zscore_anomalies(series: pd.Series, seasonal: int = 24, z: float = 3.0) -> pd.DataFrame:
    """Detect anomalies via STL residual z-scores."""
    try:
        from statsmodels.tsa.seasonal import STL
    except Exception as e:
        raise RuntimeError("statsmodels is required for STL-based anomaly detection.") from e

    stl = STL(series, period=seasonal, robust=True).fit()
    resid = stl.resid
    mu, sigma = resid.mean(), resid.std(ddof=1)
    zscores = (resid - mu) / (sigma + 1e-8)
    is_anom = np.abs(zscores) >= z
    return pd.DataFrame({"residual": resid, "z": zscores, "anomaly": is_anom}, index=series.index)

def isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.Series:
    """Multivariate anomaly detection using IsolationForest."""
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        raise RuntimeError("scikit-learn is required for IsolationForest anomalies.") from e
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(df.values)  # -1 anomalies, 1 normal
    return pd.Series(labels == -1, index=df.index, name="anomaly")
