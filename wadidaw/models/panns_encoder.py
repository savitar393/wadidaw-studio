"""Frozen PANNs CNN14 audio encoder for M2.

This is a *vendored*, inference-only subset of the official PANNs `Cnn14`
network (Kong et al., 2020).  Only the log-mel -> embedding path is kept:

    log-mel dB (64 x T) -> bn0 -> conv_block1..6 -> temporal pooling -> fc1 -> 2048-d

The pretrained AudioSet checkpoint (``Cnn14_mAP=0.431.pth``, ~320 MB) is
downloaded once from Zenodo and loaded with ``strict=False``, keeping only the
``bn0`` / ``conv_block*`` / ``fc1`` tensors.  All parameters are frozen and the
module always runs in evaluation mode so BatchNorm uses its running statistics.

The project's log-mel features are 16 kHz / 64 bins / 8 s; PANNs was trained on
32 kHz / 64 bins / 10 s.  Feeding the project log-mel directly is therefore a
documented domain shift, but it keeps the absolute stem level (`g0`) visible to
the model, unlike gain-invariant embedding caches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PANNS_EMBED_DIM: int = 2048
PANNS_CHECKPOINT_FILENAME: str = "Cnn14_mAP=0.431.pth"
PANNS_CHECKPOINT_URL: str = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
)


def default_checkpoint_path() -> Path:
    """Return the canonical PANNs checkpoint location (``~/panns_data``)."""
    env_path = os.environ.get("PANNS_CHECKPOINT")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / "panns_data" / PANNS_CHECKPOINT_FILENAME


def download_panns_checkpoint(path: Path | str | None = None) -> Path:
    """Download the official PANNs Cnn14 checkpoint if it is not present."""
    target = Path(path).expanduser() if path is not None else default_checkpoint_path()
    if target.is_file() and target.stat().st_size >= 3e8:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[panns] downloading Cnn14 checkpoint -> {target} (~320 MB)")

    try:
        from urllib.request import urlretrieve

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            pct = min(100.0, 100.0 * block_num * block_size / total_size)
            sys.stdout.write(f"\r[panns] {pct:5.1f}%")
            sys.stdout.flush()

        urlretrieve(PANNS_CHECKPOINT_URL, target, reporthook=_progress)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "Failed to download the PANNs Cnn14 checkpoint from "
            f"{PANNS_CHECKPOINT_URL}. Download it manually and pass "
            f"`checkpoint_path=...` (expected at {target})."
        ) from exc

    if target.stat().st_size < 3e8:
        raise RuntimeError(
            f"PANNs checkpoint at {target} looks incomplete "
            f"({target.stat().st_size} bytes)."
        )
    return target


def _init_layer(layer: nn.Linear | nn.Conv2d) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(layer: nn.BatchNorm2d) -> None:
    layer.bias.data.fill_(0.0)
    layer.weight.data.fill_(1.0)


class _ConvBlock(nn.Module):
    """PANNs ConvBlock: conv(3x3) -> BN -> ReLU -> conv(3x3) -> BN -> ReLU -> pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1), bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
            padding=(1, 1), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(
        self,
        x: torch.Tensor,
        pool_size: tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg+max":
            return F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(
                x, kernel_size=pool_size
            )
        raise ValueError(f"Unsupported pool_type: {pool_type}")


class PANNsCnn14Encoder(nn.Module):
    """Frozen PANNs CNN14 feature extractor operating on 64-bin log-mel input.

    Input to :meth:`encode_stems` is ``(B, 4, 64, T)`` log-mel in dBFS (the same
    tensor the dataset already produces as ``logmel_db``).  Output is
    ``(B, 4, 2048)``, the AudioSet ``fc1`` embedding per stem.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        download: bool = True,
        embed_dim: int = PANNS_EMBED_DIM,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        _init_bn(self.bn0)
        _init_layer(self.fc1)

        path = Path(checkpoint_path).expanduser() if checkpoint_path else default_checkpoint_path()
        if not path.is_file() or path.stat().st_size < 3e8:
            if not download:
                raise FileNotFoundError(
                    f"PANNs checkpoint not found at {path} and download=False."
                )
            path = download_panns_checkpoint(path)

        self.checkpoint_path = path
        self._load_pretrained(path)

        if embed_dim != PANNS_EMBED_DIM:
            self.proj = nn.Linear(PANNS_EMBED_DIM, embed_dim)
        else:
            self.proj = nn.Identity()

        if freeze:
            for param in self.parameters():
                param.requires_grad_(False)

        # The encoder is always inference-only: keep BatchNorm/dropout deterministic.
        self.eval()

    def _load_pretrained(self, path: Path) -> None:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected PANNs checkpoint format at {path}.")

        prefix = ("bn0.", "conv_block", "fc1.")
        filtered = {k: v for k, v in state.items() if k.startswith(prefix)}
        if "fc1.weight" not in filtered:
            raise RuntimeError(
                f"PANNs checkpoint at {path} does not contain the expected CNN14 "
                "weights (fc1.weight missing)."
            )

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        # Only the (skipped) classifier / spec-augmenter keys may be missing.
        real_missing = [k for k in missing if k.startswith(prefix)]
        if real_missing:
            raise RuntimeError(
                f"Could not load {len(real_missing)} PANNs tensors, e.g. {real_missing[:5]}."
            )
        self.load_report = {
            "loaded": len(filtered),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }

    def train(self, mode: bool = True) -> "PANNsCnn14Encoder":
        """Keep the frozen encoder in eval mode regardless of the parent model."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward_logmel(self, logmel_db: torch.Tensor) -> torch.Tensor:
        """Encode ``(N, 64, T)`` log-mel dBFS into ``(N, embed_dim)`` embeddings."""
        if logmel_db.ndim == 3:
            x = logmel_db.transpose(1, 2)  # (N, T, mel)
        elif logmel_db.ndim == 2:
            x = logmel_db.transpose(0, 1).unsqueeze(0)
        else:
            raise ValueError(
                f"Expected log-mel with shape (N, 64, T), got {tuple(logmel_db.shape)}."
            )

        x = x.unsqueeze(1).transpose(1, 3)  # (N, mel, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (N, 1, T, mel)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")

        x = torch.mean(x, dim=3)  # (N, 2048, T')
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        x = F.relu_(self.fc1(x))
        return self.proj(x)

    @torch.no_grad()
    def encode_stems(self, logmel_db: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 4, 64, T)`` log-mel dBFS into ``(B, 4, embed_dim)``."""
        if logmel_db.ndim != 4 or logmel_db.shape[1] != 4:
            raise ValueError(
                f"Expected log-mel with shape (B, 4, 64, T), got {tuple(logmel_db.shape)}."
            )
        bsz, n_stems, n_mels, frames = logmel_db.shape
        flat = logmel_db.reshape(bsz * n_stems, n_mels, frames)
        emb = self.forward_logmel(flat)
        return emb.reshape(bsz, n_stems, self.embed_dim)


__all__ = [
    "PANNS_EMBED_DIM",
    "PANNS_CHECKPOINT_FILENAME",
    "PANNS_CHECKPOINT_URL",
    "PANNsCnn14Encoder",
    "default_checkpoint_path",
    "download_panns_checkpoint",
]