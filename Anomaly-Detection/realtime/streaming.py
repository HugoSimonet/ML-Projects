from collections import deque
from typing import Deque, Tuple
import numpy as np

class SlidingWindowBuffer:
    def __init__(self, window_size: int, feature_dim: int):
        self.window_size = window_size
        self.buf: Deque[np.ndarray] = deque(maxlen=window_size)
        self.feature_dim = feature_dim

    def push(self, x_t: np.ndarray):
        assert x_t.shape[-1] == self.feature_dim
        self.buf.append(x_t)

    def get_window(self) -> np.ndarray:
        if len(self.buf) < self.window_size:
            raise ValueError("Insufficient history for a full window")
        return np.stack(self.buf, axis=0)

def greedy_stream_predict(model, window: np.ndarray, horizon: int) -> np.ndarray:
    """Convenience wrapper to produce horizon-step forecast from a history window."""
    import torch
    model.eval()
    x = torch.tensor(window[None, ...], dtype=torch.float32)
    last = x[:, -1:, :].repeat(1, horizon, 1)
    with torch.no_grad():
        out = model(x, last)
        if isinstance(out, tuple):  # gaussian
            out = out[0]
    return out.squeeze(0).cpu().numpy()
