"""YAMNet embedding extraction for the M2 pretrained late-fusion model.

YAMNet (Howard et al., 2019; TF Hub ``google/yamnet``) is a *frozen* AudioSet
classifier whose penultimate layer yields 1024-d frame embeddings.  This module
is intentionally kept separate from the PyTorch model: embeddings are extracted
once with TensorFlow and cached to disk, so the training loop only needs the
cached ``(4, 2048)`` tensors instead of running TensorFlow.

YAMNet consumes a **16 kHz mono waveform** (unlike PANNs, which reads log-mel).
The per-stem frame embeddings are pooled over time (mean or mean+max) and a
per-stem broadband loudness column is appended later by the cache builder so the
input gain ``g0`` stays observable to the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from wadidaw.data.features import SAMPLE_RATE_MODEL, SAMPLE_RATE_RAW, resample_to_16k_mono
from wadidaw.data.musdb import load_excerpt_stems_44k
from wadidaw.data.prompts import STEM_NAMES


YAMNET_SAMPLE_RATE: int = SAMPLE_RATE_MODEL
YAMNET_EMBED_DIM: int = 1024
YAMNET_POOL_DIM: dict[str, int] = {"mean": YAMNET_EMBED_DIM, "mean_max": 2 * YAMNET_EMBED_DIM}

YAMNET_HANDLE: str = "https://tfhub.dev/google/yamnet/1"
YAMNET_KAGGLE_HANDLE: str = "https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1"


def resolve_yamnet_handle(model: str | Path | None = None) -> str:
    """Resolve the YAMNet model location (local path or TF Hub handle)."""
    if model is not None:
        return str(model)
    env = os.environ.get("YAMNET_MODEL")
    if env:
        return env
    return YAMNET_HANDLE


def load_yamnet_model(model: str | Path | None = None):
    """Load the frozen YAMNet TF Hub module.

    Tries the configured location, then the Kaggle mirror.  TensorFlow/TF-Hub
    are imported lazily so the rest of the project works without them.
    """
    try:
        import tensorflow_hub as hub  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency is optional
        raise RuntimeError(
            "tensorflow-hub is not installed. Run `pip install tensorflow tensorflow-hub`."
        ) from exc

    candidates = [resolve_yamnet_handle(model)]
    if model is None and os.environ.get("YAMNET_MODEL") is None:
        candidates.append(YAMNET_KAGGLE_HANDLE)

    last_error: Exception | None = None
    for handle in candidates:
        try:
            print(f"[yamnet] loading {handle}")
            return hub.load(handle)
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            print(f"[yamnet] failed to load {handle}: {exc}")

    raise RuntimeError(
        "Could not load YAMNet from "
        + ", ".join(candidates)
        + ". Download the model manually and pass `--yamnet-model <path-or-handle>` "
        "or set the YAMNET_MODEL environment variable."
    ) from last_error


def _pool_frames(embedding: np.ndarray, pool: str = "mean_max") -> np.ndarray:
    """Reduce YAMNet ``(frames, 1024)`` embeddings to a fixed-length vector."""
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected YAMNet embedding shape (frames, dim), got {arr.shape}.")
    if arr.shape[0] == 0:
        arr = np.zeros((1, arr.shape[1]), dtype=np.float32)

    mean = arr.mean(axis=0)
    if pool == "mean":
        return mean.astype(np.float32, copy=False)
    if pool == "mean_max":
        return np.concatenate((mean, arr.max(axis=0)), axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported pool '{pool}'. Expected 'mean' or 'mean_max'.")


def extract_yamnet_embeddings(
    stems_stereo: Mapping[str, np.ndarray] | Sequence[np.ndarray],
    model,
    orig_sr: int = SAMPLE_RATE_RAW,
    pool: str = "mean_max",
) -> np.ndarray:
    """Extract ``(4, D)`` pooled YAMNet embeddings from four stereo stems.

    ``D`` is 1024 for ``pool="mean"`` and 2048 for ``pool="mean_max"``.
    """
    if isinstance(stems_stereo, Mapping):
        stem_items = [stems_stereo[name] for name in STEM_NAMES]
    else:
        stem_items = list(stems_stereo)
    if len(stem_items) != 4:
        raise ValueError(f"Expected 4 stems, got {len(stem_items)}.")

    embeddings: list[np.ndarray] = []
    for stem in stem_items:
        mono16 = resample_to_16k_mono(stem, orig_sr=orig_sr)
        outputs = model(mono16.astype(np.float32))
        # TF Hub YAMNet returns (scores, embeddings, spectrogram).
        frame_embeddings = np.asarray(outputs[1].numpy() if hasattr(outputs[1], "numpy") else outputs[1])
        embeddings.append(_pool_frames(frame_embeddings, pool=pool))

    return np.stack(embeddings, axis=0).astype(np.float32, copy=False)


def extract_yamnet_for_record(
    record: Mapping[str, object],
    song_dir: Path | str,
    model,
    pool: str = "mean_max",
) -> np.ndarray:
    """Load an excerpt's original stems and extract its pooled YAMNet embedding."""
    stems = load_excerpt_stems_44k(
        song_dir=song_dir,
        start_sec=float(record["excerpt_start_sec"]),
        duration_sec=float(record.get("duration_sec", 8.0)),
    )
    return extract_yamnet_embeddings(
        stems_stereo=stems,
        model=model,
        orig_sr=SAMPLE_RATE_RAW,
        pool=pool,
    )


__all__ = [
    "YAMNET_EMBED_DIM",
    "YAMNET_HANDLE",
    "YAMNET_KAGGLE_HANDLE",
    "YAMNET_POOL_DIM",
    "YAMNET_SAMPLE_RATE",
    "extract_yamnet_embeddings",
    "extract_yamnet_for_record",
    "load_yamnet_model",
    "resolve_yamnet_handle",
]
