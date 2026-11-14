"""Audio encoders for 4-stem `(B, 4, 64, 800)` log-mel spectrograms.

Implements Section 6.1 and Section 6.4 of dokumentasi.md:
- `SharedStemAudioCNN`: shared Conv2D encoder per stem (`32 -> 64 -> 128`) with
  calibrated RMS log-mean-power pooling for M1, M3, and M5.
- `MultiScaleSpectralBackbone`: deeper CNN14-inspired residual spectral-temporal
  feature extractor for M2.
- `StemFrameTokenizer`: extracts per-stem temporal token sequences `(B, 4, T', D)`
  for M4 cross-attention.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_LN10_OVER_10 = math.log(10.0) / 10.0
_TEN_OVER_LN10 = 10.0 / math.log(10.0)


def compute_calibrated_mel_stats(audio_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract calibrated dBFS log-mean-power, mean dB, and temporal std per mel band.

    Args:
        audio_norm: `(N, 1, 64, T)` normalized log-mel in `[-1, 1]` where `dB = audio_norm * 50 - 30`.

    Returns:
        - `band_lmp_norm`: `(N, 64)` log-mean-power per mel bin (normalized to `[-1, 1]`).
        - `broadband_lmp_db`: `(N, 1)` broadband stem RMS estimate in dBFS.
        - `band_std_norm`: `(N, 64)` temporal standard deviation across frames.
    """
    x_squeezed = audio_norm.squeeze(1)  # (N, 64, T)
    frames = x_squeezed.shape[-1]

    # Convert from [-1, 1] back to dBFS: dB = x * 50.0 - 30.0
    # Compute log-mean-power across time via logsumexp on a 4x pooled grid for speed
    x_pool = F.avg_pool1d(x_squeezed, kernel_size=4, stride=4) if frames >= 16 else x_squeezed
    db_pool = x_pool * 50.0 - 30.0
    t_len = db_pool.shape[-1]

    # 10 * log10(mean_t(10**(dB / 10)))
    band_lmp_db = _TEN_OVER_LN10 * (
        torch.logsumexp(db_pool * _LN10_OVER_10, dim=-1) - math.log(max(t_len, 1))
    )  # (N, 64)
    broadband_lmp_db = _TEN_OVER_LN10 * (
        torch.logsumexp(band_lmp_db * _LN10_OVER_10, dim=-1, keepdim=True) - math.log(64.0)
    )  # (N, 1)

    band_lmp_norm = (band_lmp_db + 30.0) / 50.0
    band_std_norm = db_pool.std(dim=-1, unbiased=False) / 25.0
    return band_lmp_norm, broadband_lmp_db, band_std_norm


class SharedStemAudioCNN(nn.Module):
    """Shared 2D CNN applied independently to each of the 4 stems.

    Input shape:  `(B, 4, 64, T)` normalized dBFS log-mel spectrogram.
    Outputs:
        - `stem_embeddings`: `(B, 4, embed_dim)` per-stem feature vectors.
        - `loudness_pred`:   `(B, 4)` predicted stem loudness in dBFS (used in M3).
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 3-stage Conv2D hierarchy from Section 6.4: Conv2D 32 -> 64 -> 128
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
        )

        # Combine 2D conv pooled features (128 mean + 128 std) + calibrated mel stats (64 lmp + 64 std + 1 broadband)
        self.proj = nn.Sequential(
            nn.Linear(128 * 2 + 64 * 2 + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        # Residual loudness head anchored on broadband log-mean-power + learned spectral correction
        self.loudness_head = nn.Linear(embed_dim + 1, 1)
        nn.init.zeros_(self.loudness_head.weight)
        nn.init.constant_(self.loudness_head.bias, 0.0)
        with torch.no_grad():
            self.loudness_head.weight[0, -1] = 1.0

    def encode_stems_and_loudness(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 4 or audio.shape[1] != 4 or audio.shape[2] != 64:
            raise ValueError(f"Expected audio shape (B, 4, 64, T), got {tuple(audio.shape)}.")

        bsz, n_stems, n_mels, frames = audio.shape
        x = audio.reshape(bsz * n_stems, 1, n_mels, frames)

        # Downsample (64, 800) -> (32, 100) before 2D Conv stack for 8x faster execution
        x_conv = F.avg_pool2d(x, kernel_size=(2, 8)) if frames >= 16 else x
        feat_map = self.conv_blocks(x_conv)  # (B*4, 128, 4, 13)
        conv_mean = feat_map.mean(dim=(2, 3))
        conv_std = feat_map.std(dim=(2, 3), unbiased=False)

        band_lmp_norm, broadband_lmp_db, band_std_norm = compute_calibrated_mel_stats(x)

        combined = torch.cat((conv_mean, conv_std, band_lmp_norm, band_std_norm, broadband_lmp_db / 20.0), dim=-1)
        emb = self.proj(combined)
        l_pred = self.loudness_head(torch.cat((emb, broadband_lmp_db), dim=-1)).squeeze(-1)

        return (
            emb.reshape(bsz, n_stems, self.embed_dim),
            l_pred.reshape(bsz, n_stems),
        )

    def encode_stems(self, audio: torch.Tensor) -> torch.Tensor:
        emb, _ = self.encode_stems_and_loudness(audio)
        return emb

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(stem_embeddings, loudness_pred)` with shapes `((B, 4, D), (B, 4))`."""
        return self.encode_stems_and_loudness(audio)


class _ResSpectralBlock(nn.Module):
    """Residual 2D convolutional block inspired by PANNs CNN14."""

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int] = (2, 2)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.GroupNorm(8, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + self.shortcut(x))


class MultiScaleSpectralBackbone(nn.Module):
    """PANNs-CNN14-style multi-resolution audio feature extractor for M2."""

    def __init__(
        self,
        embed_dim: int = 128,
        freeze_backbone: bool = True,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(202602)
        try:
            self.backbone = nn.Sequential(
                _ResSpectralBlock(1, 32, stride=(2, 2)),
                _ResSpectralBlock(32, 64, stride=(2, 2)),
                _ResSpectralBlock(64, 128, stride=(2, 2)),
                _ResSpectralBlock(128, 256, stride=(2, 2)),
            )
        finally:
            torch.random.set_rng_state(rng_state)

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(256 * 2 + 64 * 2 + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def encode_stems(self, audio: torch.Tensor) -> torch.Tensor:
        bsz, n_stems, n_mels, frames = audio.shape
        x = audio.reshape(bsz * n_stems, 1, n_mels, frames)

        x_conv = F.avg_pool2d(x, kernel_size=(2, 8)) if frames >= 16 else x
        feat = self.backbone(x_conv)
        avg_pool = feat.mean(dim=(2, 3))
        max_pool = feat.amax(dim=(2, 3))

        band_lmp_norm, broadband_lmp_db, band_std_norm = compute_calibrated_mel_stats(x)
        concat = torch.cat((avg_pool, max_pool, band_lmp_norm, band_std_norm, broadband_lmp_db / 20.0), dim=-1)
        return self.head(concat).reshape(bsz, n_stems, self.embed_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode_stems(audio).reshape(audio.shape[0], -1)


class OpenL3TokenProjector(nn.Module):
    """Project cached OpenL3 embeddings into M4 audio tokens.

    Input:  `(B, 4, T, openl3_dim)` embeddings, e.g. `(B, 4, 80, 512)`.
    Output: `(B, 4, n_temporal_tokens, d_model)` tokens with stem/time identity.
    """

    def __init__(
        self,
        openl3_dim: int = 512,
        d_model: int = 128,
        n_temporal_tokens: int = 80,
    ) -> None:
        super().__init__()
        self.openl3_dim = openl3_dim
        self.d_model = d_model
        self.n_temporal_tokens = n_temporal_tokens
        self.proj = nn.Sequential(
            nn.Linear(openl3_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.stem_type_embedding = nn.Parameter(torch.randn(1, 4, 1, d_model) * 0.02)
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, 1, n_temporal_tokens, d_model) * 0.02
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 4 or embeddings.shape[1] != 4:
            raise ValueError(
                "Expected OpenL3 embeddings with shape (B, 4, T, D), "
                f"got {tuple(embeddings.shape)}."
            )
        bsz, n_stems, frames, dim = embeddings.shape
        if dim != self.openl3_dim:
            raise ValueError(
                f"Expected OpenL3 dimension {self.openl3_dim}, got {dim}."
            )

        x = embeddings.reshape(bsz * n_stems, frames, dim).transpose(1, 2)  # (B*4, D, T)
        if frames != self.n_temporal_tokens:
            x = F.interpolate(
                x,
                size=self.n_temporal_tokens,
                mode="linear",
                align_corners=False,
            )
        x = x.transpose(1, 2)  # (B*4, T, D)
        tokens = self.proj(x).reshape(bsz, n_stems, self.n_temporal_tokens, self.d_model)
        return tokens + self.stem_type_embedding + self.temporal_pos_embedding


class StemFrameTokenizer(nn.Module):
    """Tokenize `(B, 4, 64, 800)` into temporal-stem tokens `(B, 4, T', d_model)` for M4 Cross-Attention."""

    def __init__(self, d_model: int = 128, n_temporal_tokens: int = 25) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_temporal_tokens = n_temporal_tokens

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=(2, 1), padding=1),
            nn.GELU(),
        )
        self.freq_proj = nn.Linear(d_model * 4 + 5, d_model)
        self.stem_type_embedding = nn.Parameter(torch.randn(1, 4, 1, d_model) * 0.02)
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 1, n_temporal_tokens, d_model) * 0.02)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        bsz, n_stems, n_mels, frames = audio.shape
        x = audio.reshape(bsz * n_stems, 1, n_mels, frames)

        x_conv = F.avg_pool2d(x, kernel_size=(2, 8)) if frames >= 16 else x
        feat = self.conv(x_conv)
        if feat.shape[-1] != self.n_temporal_tokens or feat.shape[-2] != 4:
            feat = F.adaptive_avg_pool2d(feat, (4, self.n_temporal_tokens))

        _, broadband_lmp_db, _ = compute_calibrated_mel_stats(x)
        broadband_rep = (broadband_lmp_db / 20.0).unsqueeze(1).expand(-1, self.n_temporal_tokens, 1)

        raw_pooled = F.adaptive_avg_pool2d(x, (4, self.n_temporal_tokens)).squeeze(1)
        feat_flat = feat.permute(0, 3, 1, 2).reshape(bsz * n_stems, self.n_temporal_tokens, -1)
        raw_flat = raw_pooled.permute(0, 2, 1)

        tokens = self.freq_proj(torch.cat((feat_flat, raw_flat, broadband_rep), dim=-1))
        tokens = tokens.reshape(bsz, n_stems, self.n_temporal_tokens, self.d_model)
        tokens = tokens + self.stem_type_embedding + self.temporal_pos_embedding
        return tokens
