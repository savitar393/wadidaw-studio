"""OpenL3 embedding extraction for the M4-OpenL3 audio branch.

OpenL3 is a *frozen* pretrained audio encoder.  This module is intentionally
kept separate from the model so embeddings can be extracted once and cached to
disk.  Training then only needs the cached ``(4, T, 512)`` tensors instead of
running TensorFlow/OpenL3 inside the PyTorch training loop.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import resample_poly

from wadidaw.data.features import SAMPLE_RATE_RAW, STEM_NAMES
from wadidaw.data.musdb import load_excerpt_stems_44k


OPENL3_SAMPLE_RATE: int = 48_000
OPENL3_EMBED_DIM: int = 512
OPENL3_FRAMES: int = 80  # ~8 s at OpenL3's default 0.1 s hop


def _resample_mono_to_openl3(
    stereo: np.ndarray,
    orig_sr: int = SAMPLE_RATE_RAW,
) -> np.ndarray:
    """Downmix a stereo stem to mono and resample it to 48 kHz."""
    arr = np.asarray(stereo, dtype=np.float32)
    if arr.ndim == 2:
        mono = arr.mean(axis=1, dtype=np.float32)
    elif arr.ndim == 1:
        mono = arr
    else:
        raise ValueError(f"Expected mono/stereo audio, got shape {arr.shape}.")

    if int(orig_sr) == OPENL3_SAMPLE_RATE:
        return mono.astype(np.float32, copy=False)

    factor = gcd(int(orig_sr), OPENL3_SAMPLE_RATE)
    up = OPENL3_SAMPLE_RATE // factor
    down = int(orig_sr) // factor
    return resample_poly(mono, up, down).astype(np.float32, copy=False)


def _fit_time_frames(embedding: np.ndarray, target_frames: int = OPENL3_FRAMES) -> np.ndarray:
    """Pad or truncate an OpenL3 frame sequence to a fixed length."""
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected OpenL3 embedding shape (frames, dim), got {arr.shape}.")
    frames, dim = arr.shape
    if frames == target_frames:
        return arr
    if frames > target_frames:
        return arr[:target_frames]
    if frames == 0:
        return np.zeros((target_frames, dim), dtype=np.float32)
    pad = np.repeat(arr[-1:, :], target_frames - frames, axis=0)
    return np.concatenate((arr, pad), axis=0).astype(np.float32, copy=False)


def extract_openl3_embeddings(
    stems_stereo: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    orig_sr: int = SAMPLE_RATE_RAW,
    content_type: str = "music",
    input_repr: str = "mel256",
    embedding_size: int = OPENL3_EMBED_DIM,
    target_frames: int = OPENL3_FRAMES,
    verbose: bool = False,
) -> np.ndarray:
    """Extract ``(4, target_frames, embedding_size)`` OpenL3 embeddings.

    The OpenL3 dependency is imported lazily so the rest of the project can be
    used without TensorFlow/OpenL3 installed.
    """
    try:
        import openl3  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency is optional
        raise RuntimeError(
            "OpenL3 is not installed. Run `pip install openl3 tensorflow` "
            "and ensure libsndfile is available on Linux."
        ) from exc

    if isinstance(stems_stereo, Mapping):
        stem_items = [stems_stereo[name] for name in STEM_NAMES]
    else:
        stem_items = list(stems_stereo)
    if len(stem_items) != 4:
        raise ValueError(f"Expected 4 stems, got {len(stem_items)}.")

    embeddings: list[np.ndarray] = []
    for stem in stem_items:
        mono_48k = _resample_mono_to_openl3(stem, orig_sr=orig_sr)
        emb, _timestamps = openl3.get_audio_embedding_from_data(
            x=mono_48k,
            sr=OPENL3_SAMPLE_RATE,
            content_type=content_type,
            input_repr=input_repr,
            embedding_size=int(embedding_size),
            verbose=verbose,
        )
        embeddings.append(_fit_time_frames(emb, target_frames=target_frames))

    return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


def extract_openl3_for_record(
    record: Mapping[str, object],
    song_dir: Path | str,
    content_type: str = "music",
    input_repr: str = "mel256",
    embedding_size: int = OPENL3_EMBED_DIM,
    target_frames: int = OPENL3_FRAMES,
    verbose: bool = False,
) -> np.ndarray:
    """Load an excerpt's original stems and extract its OpenL3 embedding."""
    stems = load_excerpt_stems_44k(
        song_dir=song_dir,
        start_sec=float(record["excerpt_start_sec"]),
        duration_sec=float(record.get("duration_sec", 8.0)),
    )
    return extract_openl3_embeddings(
        stems_stereo=stems,
        orig_sr=SAMPLE_RATE_RAW,
        content_type=content_type,
        input_repr=input_repr,
        embedding_size=embedding_size,
        target_frames=target_frames,
        verbose=verbose,
    )
