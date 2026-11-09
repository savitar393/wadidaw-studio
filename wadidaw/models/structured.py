"""M3: Structured Two-Branch Multimodal Gain Staging Model (`L_pred` + `r_pred` -> `g_pred`).

Implements Section 6.4 of dokumentasi.md:
- Shared per-stem CNN audio encoder predicts per-stem loudness `L_pred` (4 values in dBFS).
- Frozen text encoder + MLP predicts relative target balance `r_pred` (4 values in dB, sum = 0).
- Physics-guided gain staging formula:
      L_ref  = mean(L_pred)
      g_pred = L_ref + r_pred - L_pred
- Multi-task loss with auxiliary supervision on `L_pred` and `r_pred`:
      loss = MSE(g_pred, g_true) + 0.1 * MSE(L_pred, L_dsp) + 0.1 * MSE(r_pred, r_label)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wadidaw.data.features import GAIN_MAX_DB, GAIN_MIN_DB
from wadidaw.models.audio_cnn import SharedStemAudioCNN
from wadidaw.models.text_encoder import TEXT_EMBED_DIM, TextEncoderWrapper


class StructuredTwoBranchModel(nn.Module):
    """M3 Structured Two-Branch model decomposing gain prediction into `L` (audio) and `r` (text)."""

    model_code = "m3"

    def __init__(
        self,
        audio_embed_dim: int = 128,
        text_embed_dim: int = TEXT_EMBED_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        text_backend: str = "subword",
    ) -> None:
        super().__init__()
        self.audio_branch = SharedStemAudioCNN(embed_dim=audio_embed_dim, dropout=dropout)
        self.text_encoder = TextEncoderWrapper(backend=text_backend, embed_dim=text_embed_dim)

        self.text_mlp = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        audio: torch.Tensor,
        prompts: Sequence[str] | torch.Tensor,
        active_mask: torch.Tensor | None = None,
        clamp_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute `gains_pred`, `loudness_pred`, and `balance_pred` from `(audio, prompts)`."""
        device = audio.device
        _, l_pred = self.audio_branch(audio)  # (B, 4)

        text_emb = self.text_encoder.encode_sentences(prompts, device=device)  # (B, 384)
        raw_r = 6.0 * torch.tanh(self.text_mlp(text_emb) / 6.0)  # bounded in (-6, 6) dB

        if active_mask is not None:
            mask_f = active_mask.to(device=device, dtype=torch.float32)
            active_count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
            r_mean = (raw_r * mask_f).sum(dim=1, keepdim=True) / active_count
            r_pred = (raw_r - r_mean) * mask_f
            l_ref = (l_pred * mask_f).sum(dim=1, keepdim=True) / active_count
            g_pred = (l_ref + r_pred - l_pred) * mask_f
        else:
            r_pred = raw_r - raw_r.mean(dim=1, keepdim=True)
            l_ref = l_pred.mean(dim=1, keepdim=True)
            g_pred = l_ref + r_pred - l_pred

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        return {
            "gains_pred": g_pred,
            "loudness_pred": l_pred,
            "balance_pred": r_pred,
            "l_ref_pred": l_ref.squeeze(-1),
        }


def compute_multimodal_loss(
    outputs: dict[str, torch.Tensor],
    gains_true: torch.Tensor,
    loudness_true: torch.Tensor | None = None,
    balance_true: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    lambda_loudness: float = 0.1,
    lambda_balance: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute primary gain MSE loss plus optional auxiliary loudness/balance MSE losses."""
    g_pred = outputs["gains_pred"]

    if active_mask is not None:
        mask_f = active_mask.to(device=g_pred.device, dtype=torch.float32)
        denom = mask_f.sum().clamp(min=1.0)
        gain_mse = ((g_pred - gains_true).square() * mask_f).sum() / denom
    else:
        mask_f = None
        denom = None
        gain_mse = F.mse_loss(g_pred, gains_true)

    total_loss = gain_mse
    loss_dict: dict[str, torch.Tensor] = {"gain_mse": gain_mse}

    if "loudness_pred" in outputs and loudness_true is not None and lambda_loudness > 0.0:
        l_pred = outputs["loudness_pred"]
        if mask_f is not None and denom is not None:
            l_mse = ((l_pred - loudness_true).square() * mask_f).sum() / denom
        else:
            l_mse = F.mse_loss(l_pred, loudness_true)
        total_loss = total_loss + lambda_loudness * l_mse
        loss_dict["loudness_mse"] = l_mse

    if "balance_pred" in outputs and balance_true is not None and lambda_balance > 0.0:
        r_pred = outputs["balance_pred"]
        if mask_f is not None and denom is not None:
            r_mse = ((r_pred - balance_true).square() * mask_f).sum() / denom
        else:
            r_mse = F.mse_loss(r_pred, balance_true)
        total_loss = total_loss + lambda_balance * r_mse
        loss_dict["balance_mse"] = r_mse

    loss_dict["loss"] = total_loss
    return loss_dict
