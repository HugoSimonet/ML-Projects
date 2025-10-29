from typing import Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class WindowConfig:
    input_len: int = 96
    pred_len: int = 24
    stride: int = 1

class TimeSeriesDataset:
    """Sliding-window dataset for (uni/multi)variate series.
    Expects a pandas DataFrame with a DateTimeIndex and columns as variables.
    """
    def __init__(self, df: pd.DataFrame, config: WindowConfig):
        assert isinstance(df.index, pd.DatetimeIndex)
        self.df = df.sort_index()
        self.values = self.df.values.astype(np.float32)
        self.config = config
        self.indices = self._make_indices()

    def _make_indices(self):
        n = len(self.values)
        idx = []
        L, P, S = self.config.input_len, self.config.pred_len, self.config.stride
        for start in range(0, n - L - P + 1, S):
            idx.append((start, start + L, start + L + P))
        return idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        s, m, e = self.indices[i]
        x = self.values[s:m]
        y = self.values[m:e]
        return x, y

def make_toy_series(n: int = 500, k: int = 2, freq: str = "h", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = {}
    for i in range(k):
        data[f"x{i+1}"] = (
            10 * np.sin(2 * np.pi * t / 24) +
            3 * np.sin(2 * np.pi * t / 168) +
            0.05 * t +
            rng.normal(0, 1, size=n)
        )
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    return pd.DataFrame(data, index=idx)
