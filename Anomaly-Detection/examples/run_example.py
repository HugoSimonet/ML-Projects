import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.time_series_data import make_toy_series, TimeSeriesDataset, WindowConfig
from models.time_series_models import TransformerForecaster
from training.forecasting_trainer import ForecastingTrainer
from evaluation.forecasting_metrics import mae, rmse, mape, smape

def main():
    # 1) Data
    df = make_toy_series(n=1000, k=3, freq="H")
    cfg = WindowConfig(input_len=96, pred_len=24, stride=4)
    ds = TimeSeriesDataset(df, cfg)

    # train/val split
    n = len(ds)
    tr_n = int(n * 0.8)
    tr_idx = list(range(tr_n))
    va_idx = list(range(tr_n, n))

    class _Wrap(Dataset):
        def __init__(self, base, idxs):
            self.base = base; self.idxs = idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]

    tr_ds, va_ds = _Wrap(ds, tr_idx), _Wrap(ds, va_idx)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=32, shuffle=False)

    # 2) Model: point forecast head (switch to 'quantile' or 'gaussian' as desired)
    model = TransformerForecaster(in_dim=df.shape[1], out_dim=df.shape[1], d_model=32, head_type="point")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # 3) Train
    trainer = ForecastingTrainer(model, opt, device="cpu", head_type="point", patience=3)
    hist = trainer.fit(tr_dl, va_dl, epochs=10)
    print("history:", hist)

    # 4) Evaluate on final validation batch
    x, y = next(iter(va_dl))
    with torch.no_grad():
        last = x[:, -1:, :].repeat(1, y.shape[1], 1)
        pred = model(x, last)

    y_np, p_np = y.numpy(), pred.numpy()
    print("MAE:", mae(y_np, p_np))
    print("RMSE:", rmse(y_np, p_np))
    print("MAPE:", mape(y_np, p_np))
    print("SMAPE:", smape(y_np, p_np))

if __name__ == "__main__":
    main()
