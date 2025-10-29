from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class ForecastingTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str = "cpu", head_type: str = "point", patience: int = 5):
        self.model = model.to(device)
        self.optim = optimizer
        self.device = device
        self.head_type = head_type
        self.patience = patience

    def _batch_to_device(self, batch):
        x, y = batch
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        return x, y

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 20) -> Dict[str, Any]:
        best_val = float("inf")
        wait = 0
        history = {"train_loss": [], "val_loss": []}

        for ep in range(1, epochs + 1):
            self.model.train()
            tr_losses = []
            for batch in train_loader:
                x, y = self._batch_to_device(batch)
                B, Tin, D = x.shape
                Tout = y.shape[1]
                tgt_init = x[:, -1:, :].repeat(1, Tout, 1)

                self.optim.zero_grad()
                if self.head_type == "gaussian":
                    mu, log_std = self.model(x, tgt_init)
                    loss = self.model.head.nll(mu, log_std, y)
                elif self.head_type == "quantile":
                    qpred = self.model(x, tgt_init)
                    loss = self.model.head.pinball_loss(qpred, y)
                else:
                    pred = self.model(x, tgt_init)
                    loss = nn.MSELoss()(pred, y)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optim.step()
                tr_losses.append(loss.item())

            tr_mean = float(np.mean(tr_losses))
            history["train_loss"].append(tr_mean)

            val_mean = None
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = self._batch_to_device(batch)
                        B, Tin, D = x.shape
                        Tout = y.shape[1]
                        tgt_init = x[:, -1:, :].repeat(1, Tout, 1)
                        if self.head_type == "gaussian":
                            mu, log_std = self.model(x, tgt_init)
                            vloss = self.model.head.nll(mu, log_std, y)
                        elif self.head_type == "quantile":
                            qpred = self.model(x, tgt_init)
                            vloss = self.model.head.pinball_loss(qpred, y)
                        else:
                            pred = self.model(x, tgt_init)
                            vloss = nn.MSELoss()(pred, y)
                        val_losses.append(vloss.item())
                val_mean = float(np.mean(val_losses))
                history["val_loss"].append(val_mean)

                if val_mean < best_val:
                    best_val = val_mean
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

        return history

    @torch.no_grad()
    def predict(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        self.model.eval()
        x = x.to(self.device)
        last = x[:, -1:, :]
        if self.head_type == "gaussian":
            outs = []
            cur = last.clone()
            for _ in range(horizon):
                mu, _ = self.model(x, cur)
                step = mu[:, -1:, :]
                outs.append(step)
                cur = torch.cat([cur, step], dim=1)
            return torch.cat(outs, dim=1)
        else:
            cur = last.repeat(1, horizon, 1)
            out = self.model(x, cur)
            return out
