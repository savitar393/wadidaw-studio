"""Multimodal fusion models: M1 baseline and M4 cross-attention variants.

- ``LateFusionScratchModel`` (M1): scratch shared-stem CNN + text encoder,
  concatenation, MLP -> 4 gains. Supports audio-only/text-only ablations.
- ``CrossAttentionGainModel`` (M4): audio stem-temporal tokens + text clause
  tokens via multi-head cross-attention -> 4 gains. Supports scratch log-mel
  and cached OpenL3 embeddings.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wadidaw.data.features import GAIN_MAX_DB, GAIN_MIN_DB
from wadidaw.models.audio_cnn import (
    OpenL3TokenProjector,
    SharedStemAudioCNN,
    StemFrameTokenizer,
)
from wadidaw.models.text_encoder import (
    TEXT_EMBED_DIM,
    TextEncoderWrapper,
    get_text_embed_dim,
)


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


class CrossAttentionGainModel(nn.Module):
    """M4: Cross-Attention multimodal architecture (Section 6.7).

    Text clause tokens act as Queries over temporal-stem audio Keys/Values, followed by
    per-stem cross-modal pooling and gain prediction.

    ``audio_backend="openl3"`` expects cached OpenL3 embeddings shaped
    ``(B, 4, T, 512)`` instead of log-mel audio ``(B, 4, 64, 800)``.
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
        audio_backend: str = "scratch",
        openl3_dim: int = 512,
        predict_pan: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.audio_backend = str(audio_backend).lower()
        self.predict_pan = bool(predict_pan)
        self.audio_input_key = "openl3" if self.audio_backend == "openl3" else "audio"

        if self.audio_backend == "openl3":
            self.audio_tokenizer = OpenL3TokenProjector(
                openl3_dim=openl3_dim,
                d_model=d_model,
                n_temporal_tokens=n_temporal_tokens,
            )
        elif self.audio_backend == "scratch":
            self.audio_tokenizer = StemFrameTokenizer(
                d_model=d_model,
                n_temporal_tokens=n_temporal_tokens,
            )
        else:
            raise ValueError(
                f"Unknown audio_backend '{audio_backend}'. Use 'scratch' or 'openl3'."
            )

        self.text_encoder = TextEncoderWrapper(
            backend=text_backend,
            embed_dim=get_text_embed_dim(text_backend),
        )

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
        self.pan_head = (
            nn.Sequential(
                nn.Linear(4 * d_model + d_model, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 4),
                nn.Tanh(),
            )
            if self.predict_pan
            else None
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
        n_tokens = stem_tokens.shape[2]
        audio_kv = stem_tokens.reshape(bsz, -1, self.d_model)
        stem_pooled = stem_tokens.mean(dim=2)  # (B, 4, D)

        if active_mask is not None:
            mask_f = active_mask.to(device=device, dtype=torch.float32)
            audio_token_mask = (
                active_mask.to(device=device, dtype=torch.bool)
                .unsqueeze(-1)
                .expand(-1, -1, n_tokens)
                .reshape(bsz, -1)
            )
            stem_pooled = stem_pooled * mask_f.unsqueeze(-1)
        else:
            mask_f = None
            audio_token_mask = torch.ones(
                (bsz, 4 * n_tokens), dtype=torch.bool, device=device
            )

        # Text tokens and padding mask. IndoBERT returns variable-length tokens.
        text_tok_raw, text_mask = self.text_encoder.encode_tokens_with_mask(
            prompts, device=device
        )
        text_queries = self.text_token_proj(text_tok_raw)
        text_sent = self.text_sent_proj(
            self.text_encoder.encode_sentences(prompts, device=device)
        )

        # Step 1: Text queries attend over audio frame keys/values.
        attended_text, _ = self.text_to_audio_attn(
            query=text_queries,
            key=audio_kv,
            value=audio_kv,
            key_padding_mask=~audio_token_mask,
            need_weights=False,
        )
        context_tokens = self.norm1(text_queries + attended_text)
        context_tokens = context_tokens * text_mask.unsqueeze(-1).to(context_tokens.dtype)

        # Step 2: Each of the 4 stems attends over the cross-modal context tokens.
        attended_stems, _ = self.stem_to_context_attn(
            query=stem_pooled,
            key=context_tokens,
            value=context_tokens,
            key_padding_mask=~text_mask,
            need_weights=False,
        )
        fused_stems = self.norm2(stem_pooled + attended_stems)  # (B, 4, D)
        if mask_f is not None:
            fused_stems = fused_stems * mask_f.unsqueeze(-1)

        combined = torch.cat((fused_stems.reshape(bsz, -1), text_sent), dim=-1)
        g_pred = self.head(combined)
        pans_pred = self.pan_head(combined) if self.pan_head is not None else None

        if active_mask is not None:
            g_pred = g_pred * active_mask.to(device=device, dtype=torch.float32)
            if pans_pred is not None:
                pans_pred = pans_pred * active_mask.to(device=device, dtype=torch.float32)

        if clamp_output or not self.training:
            g_pred = g_pred.clamp(GAIN_MIN_DB, GAIN_MAX_DB)

        outputs = {"gains_pred": g_pred}
        if pans_pred is not None:
            outputs["pans_pred"] = pans_pred
        return outputs


MODEL_REGISTRY: dict[str, str] = {
    "m1": "M1 Late Fusion Scratch",
    "m1_audio_only": "M1 Ablation (Audio-Only)",
    "m1_text_only": "M1 Ablation (Text-Only)",
    "m4": "M4 Cross-Attention Multimodal (Scratch)",
    "m4_openl3": "M4 Cross-Attention + OpenL3",
    "m4_openl3_indobert": "M4 Cross-Attention + OpenL3 + IndoBERT",
}


def build_model(model_name: str, text_backend: str = "subword", **kwargs: object) -> nn.Module:
    """Instantiate an M1 or M4 model variant."""
    key = model_name.strip().lower()
    kwargs.setdefault("text_embed_dim", get_text_embed_dim(text_backend))

    if key == "m1":
        return LateFusionScratchModel(ablation_mode="multimodal", text_backend=text_backend, **kwargs)
    if key == "m1_audio_only":
        return LateFusionScratchModel(ablation_mode="audio_only", text_backend=text_backend, **kwargs)
    if key == "m1_text_only":
        return LateFusionScratchModel(ablation_mode="text_only", text_backend=text_backend, **kwargs)
    if key == "m4":
        return CrossAttentionGainModel(text_backend=text_backend, **kwargs)
    if key == "m4_openl3":
        kwargs.setdefault("audio_backend", "openl3")
        kwargs.setdefault("n_temporal_tokens", 80)
        return CrossAttentionGainModel(text_backend=text_backend, **kwargs)
    if key == "m4_openl3_indobert":
        kwargs.setdefault("audio_backend", "openl3")
        kwargs.setdefault("n_temporal_tokens", 80)
        kwargs["text_embed_dim"] = get_text_embed_dim("indobert")
        return CrossAttentionGainModel(text_backend="indobert", **kwargs)
    raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
