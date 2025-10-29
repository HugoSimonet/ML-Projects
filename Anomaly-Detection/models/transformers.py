import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VanillaTransformer(nn.Module):
    """Encoder-Decoder Transformer for sequence-to-sequence forecasting.
    Inputs are expected already embedded to d_model.
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.pos_dec = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src, tgt: (B, T, D)
        src = self.pos_enc(src)
        tgt = self.pos_dec(tgt)
        out = self.transformer(
            src, tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return out  # (B, T_tgt, D)

class Informer(nn.Module):
    """Interface stub for Informer (ProbSparse attention, distilling, etc.).
    Plug in your implementation here or swap with a library/third-party module.
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # TODO: Implement ProbSparse & encoder distilling
        raise NotImplementedError("Informer is a placeholder in this baseline.")

class Autoformer(nn.Module):
    """Interface stub for Autoformer (decomposition-based attention).
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # TODO: Implement auto-correlation mechanism & decomposition blocks
        raise NotImplementedError("Autoformer is a placeholder in this baseline.")
