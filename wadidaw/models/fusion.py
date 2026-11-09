"""Multimodal fusion models M1, M2, M4, M5 and ablation variants (`audio_only`, `text_only`).

Implements Section 6.3, 6.5, 6.6, 6.7, and 6.8 of dokumentasi.md:
- `LateFusionScratchModel` (M1): Scratch shared-stem CNN + Text Encoder -> Concat + MLP -> g* (4)
  Supports `ablation_mode` in `("multimodal", "audio_only", "text_only")`.
- `LateFusionPretrainedModel` (M2): Frozen/Multi-scale CNN14-style backbone + Text Encoder -> Concat + MLP -> g* (4).
- `CrossAttentionGainModel` (M4): Audio stem-temporal tokens (Keys/Values) + Text clause tokens (Queries)
  via Multihead Cross-Attention -> Stem-Query Pooling -> MLP -> g* (4).
- `FiLMConditionedModel` (M5): Feature-wise Linear Modulation (gamma * feature_map + beta) conditioned
  on the text prompt -> Pooling -> MLP -> g* (4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wadidaw.data.features import GAIN_MAX_DB, GAIN_MIN_DB
from wadidaw.models.audio_cnn import (
    MultiScaleSpectralBackbone,
    SharedStemAudioCNN,
    StemFrameTokenizer,
)
from wadidaw.models.structured import StructuredTwoBranchModel
from wadidaw.models.text_encoder import TEXT_EMBED_DIM, TextEncoderWrapper


class LateFusionScratchModel(nn.Module):
    """M1: Late Fusion Scratch baseline (and `audio_only` / `text_only` ablations)."""

    def __init__(
        self,
        audio_embed_dim: int = 64,
        text_embed_dim: int = TEXT_EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        ablation_mode: str = "multimodal",
        text_backend: str = "subword",
    ) -> None:
        super().__init__()
        if ablation_mode not in ("multimodal", "audio_only", "text_only"):
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}")

        self.ablation_mode = ablation_mode
        self.model_code = "m1" if ablation_mode == "multimodal" else f"m1_{ablation_mode}"

        self.audio_encoder = SharedStemAudioCNN(embed_dim=audio_embed_dim, dropout=dropout)
        self.text_encoder = TextEncoderWrapper(backend=text_backend, embed_dim=text_embed_dim)

        audio_flat_dim = 4 * audio_embed_dim  # 256
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_flat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + 256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(
        self,
        audio: torch.Tensor,
        prompts: Sequence[str] | torch.Tensor,
        active_mask: torch.Tensor | None = None,
        clamp_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = audio.device
        bsz = audio.shape[0]

        if self.ablation_mode == "text_only":
            a_feat = torch.zeros((bsz, 256), device=device, dtype=torch.float32)
        else:
            stem_emb, _ = self.audio_encoder(audio)
            a_feat = self.audio_proj(stem_emb.reshape(bsz, -1))

        if self.ablation_mode == "audio_only":
            t_feat = torch.zeros((bsz, 256), device=device, dtype=torch.float32)
        else:
            t_raw = self.text_encoder.encode_sentences(prompts, device=device)
            t_feat = self.text_proj(t_raw)

        fused = torch.cat((a_feat, t_feat), dim=-1)
        g_pred = self.fusion_mlp(fused)

        if active_mask is not None:
            g_pred = g_pred * active_mask.to(device=device, dtype=torch.float32)

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        return {"gains_pred": g_pred}


class LateFusionPretrainedModel(nn.Module):
    """M2: Late Fusion with frozen multi-scale spectral backbone (CNN14-style)."""

    model_code = "m2"

    def __init__(
        self,
        audio_embed_dim: int = 128,
        text_embed_dim: int = TEXT_EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        freeze_audio_backbone: bool = True,
        pretrained_checkpoint: Path | str | None = None,
        text_backend: str = "subword",
    ) -> None:
        super().__init__()
        self.audio_encoder = MultiScaleSpectralBackbone(
            embed_dim=audio_embed_dim,
            freeze_backbone=freeze_audio_backbone,
            checkpoint_path=pretrained_checkpoint,
        )
        self.text_encoder = TextEncoderWrapper(backend=text_backend, embed_dim=text_embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * audio_embed_dim + text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(
        self,
        audio: torch.Tensor,
        prompts: Sequence[str] | torch.Tensor,
        active_mask: torch.Tensor | None = None,
        clamp_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = audio.device
        a_flat = self.audio_encoder(audio)  # (B, 4 * audio_embed_dim)
        t_emb = self.text_encoder.encode_sentences(prompts, device=device)  # (B, 384)

        g_pred = self.fusion_mlp(torch.cat((a_flat, t_emb), dim=-1))

        if active_mask is not None:
            g_pred = g_pred * active_mask.to(device=device, dtype=torch.float32)

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        return {"gains_pred": g_pred}


class CrossAttentionGainModel(nn.Module):
    """M4: Cross-Attention multimodal architecture (Section 6.7).

    Text clause tokens act as Queries over temporal-stem audio Keys/Values, followed by
    per-stem cross-modal pooling and gain prediction.
    """

    model_code = "m4"

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_tokens: int = 25,
        text_embed_dim: int = TEXT_EMBED_DIM,
        dropout: float = 0.15,
        text_backend: str = "subword",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.audio_tokenizer = StemFrameTokenizer(
            d_model=d_model,
            n_temporal_tokens=n_temporal_tokens,
        )
        self.text_encoder = TextEncoderWrapper(backend=text_backend, embed_dim=text_embed_dim)

        self.text_token_proj = nn.Sequential(
            nn.Linear(text_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.text_sent_proj = nn.Sequential(
            nn.Linear(text_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Text queries attend to audio frame keys/values (Section 6.7)
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Stem summaries attend to text-enriched context tokens
        self.stem_to_context_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(4 * d_model + d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        audio: torch.Tensor,
        prompts: Sequence[str] | torch.Tensor,
        active_mask: torch.Tensor | None = None,
        clamp_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = audio.device
        bsz = audio.shape[0]

        # Audio tokens: (B, 4, T', D) -> (B, 4*T', D)
        stem_tokens = self.audio_tokenizer(audio)
        audio_kv = stem_tokens.reshape(bsz, -1, self.d_model)
        stem_pooled = stem_tokens.mean(dim=2)  # (B, 4, D)

        # Text clause tokens: (B, L_text, 384) -> (B, L_text, D)
        text_tok_raw = self.text_encoder.encode_tokens(prompts, device=device)
        text_queries = self.text_token_proj(text_tok_raw)
        text_sent = self.text_sent_proj(self.text_encoder.encode_sentences(prompts, device=device))

        # Step 1: Text queries attend over audio frame keys/values
        attended_text, _ = self.text_to_audio_attn(
            query=text_queries,
            key=audio_kv,
            value=audio_kv,
            need_weights=False,
        )
        context_tokens = self.norm1(text_queries + attended_text)

        # Step 2: Each of the 4 stems attends over the cross-modal context tokens
        attended_stems, _ = self.stem_to_context_attn(
            query=stem_pooled,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        fused_stems = self.norm2(stem_pooled + attended_stems)  # (B, 4, D)

        combined = torch.cat((fused_stems.reshape(bsz, -1), text_sent), dim=-1)
        g_pred = self.head(combined)

        if active_mask is not None:
            g_pred = g_pred * active_mask.to(device=device, dtype=torch.float32)

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        return {"gains_pred": g_pred}


class FiLMConditionedModel(nn.Module):
    """M5: Feature-wise Linear Modulation (FiLM) conditioning model (Section 6.8)."""

    model_code = "m5"

    def __init__(
        self,
        text_embed_dim: int = TEXT_EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.15,
        text_backend: str = "subword",
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoderWrapper(backend=text_backend, embed_dim=text_embed_dim)

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 4), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=(2, 4), padding=1)

        # FiLM generators producing (scale, shift) for each conv stage
        self.film1 = nn.Linear(text_embed_dim, 32 * 2)
        self.film2 = nn.Linear(text_embed_dim, 64 * 2)
        self.film3 = nn.Linear(text_embed_dim, 128 * 2)

        # Direct stem energy skip + pooled FiLM features + text embedding -> g*
        self.head = nn.Sequential(
            nn.Linear(128 * 2 + 4 * 64 + text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),
        )

    @staticmethod
    def _apply_film(feat: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = 1.0 + 0.5 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return F.gelu(feat * gamma + beta)

    def forward(
        self,
        audio: torch.Tensor,
        prompts: Sequence[str] | torch.Tensor,
        active_mask: torch.Tensor | None = None,
        clamp_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        device = audio.device
        bsz = audio.shape[0]
        t_emb = self.text_encoder.encode_sentences(prompts, device=device)

        audio_conv = F.avg_pool2d(audio, kernel_size=(1, 4)) if audio.shape[-1] >= 16 else audio
        h = self._apply_film(self.conv1(audio_conv), self.film1(t_emb))
        h = self._apply_film(self.conv2(h), self.film2(t_emb))
        h = self._apply_film(self.conv3(h), self.film3(t_emb))

        pooled_mean = h.mean(dim=(2, 3))
        pooled_std = h.std(dim=(2, 3), unbiased=False)
        stem_mel_means = audio.mean(dim=-1).reshape(bsz, -1)  # (B, 256)

        combined = torch.cat((pooled_mean, pooled_std, stem_mel_means, t_emb), dim=-1)
        g_pred = self.head(combined)

        if active_mask is not None:
            g_pred = g_pred * active_mask.to(device=device, dtype=torch.float32)

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        return {"gains_pred": g_pred}


MODEL_REGISTRY: dict[str, str] = {
    "m1": "M1 Late Fusion Scratch",
    "m1_audio_only": "M1 Ablation (Audio-Only)",
    "m1_text_only": "M1 Ablation (Text-Only)",
    "m2": "M2 Late Fusion Pretrained Backbone",
    "m3": "M3 Structured Two-Branch (Main)",
    "m4": "M4 Cross-Attention Multimodal",
    "m5": "M5 FiLM Conditioning",
}


def build_model(model_name: str, text_backend: str = "subword", **kwargs: object) -> nn.Module:
    """Instantiate one of the multimodal gain staging models (`m1`..`m5` and ablations)."""
    key = model_name.strip().lower()
    if key == "m1":
        return LateFusionScratchModel(ablation_mode="multimodal", text_backend=text_backend, **kwargs)
    if key == "m1_audio_only":
        return LateFusionScratchModel(ablation_mode="audio_only", text_backend=text_backend, **kwargs)
    if key == "m1_text_only":
        return LateFusionScratchModel(ablation_mode="text_only", text_backend=text_backend, **kwargs)
    if key == "m2":
        return LateFusionPretrainedModel(text_backend=text_backend, **kwargs)
    if key == "m3":
        return StructuredTwoBranchModel(text_backend=text_backend, **kwargs)
    if key == "m4":
        return CrossAttentionGainModel(text_backend=text_backend, **kwargs)
    if key == "m5":
        return FiLMConditionedModel(text_backend=text_backend, **kwargs)
    raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
