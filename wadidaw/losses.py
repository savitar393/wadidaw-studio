"""Loss functions for the M1/M4 gain-staging models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_multimodal_loss(
    outputs: dict[str, torch.Tensor],
    gains_true: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    **_ignored: object,
) -> dict[str, torch.Tensor]:
    """Masked MSE between predicted and target stem gains (dB)."""
    g_pred = outputs["gains_pred"]

    if active_mask is not None:
        mask_f = active_mask.to(device=g_pred.device, dtype=torch.float32)
        denom = mask_f.sum().clamp(min=1.0)
        gain_mse = ((g_pred - gains_true).square() * mask_f).sum() / denom
    else:
        gain_mse = F.mse_loss(g_pred, gains_true)

    return {"gain_mse": gain_mse, "loss": gain_mse}
