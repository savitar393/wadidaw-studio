"""Build a frozen PANNs CNN14 feature cache for M2.

Runs the pretrained PANNs CNN14 encoder once over every multimodal sample and
stores the per-stem embeddings next to the per-stem broadband loudness:

    features[i] = [ emb_0 .. emb_(D-1), broadband_lmp_db ]   shape (4, D + 1)

Because the dataset applies the sample-specific initial gain ``g0`` to the
log-mel before the encoder runs, the cached embedding keeps the input level
information that a gain-invariant embedding (e.g. raw OpenL3) would lose.  A
dedicated loudness column is appended as an explicit level feature.

Outputs
-------
- ``data/cache/panns/{split}.npy``            ``(N, 4, 2048 + 1)`` float32
- ``data/cache/panns/metadata.json``
- ``data/prompts/{split}_panns_samples.json``  manifests with ``panns_path``
  and ``panns_index`` so ``MusdbMultimodalDataset`` can load the right row.

Example:
    python -m scripts.build_panns_cache --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wadidaw.data.dataset import MusdbMultimodalDataset
from wadidaw.data.musdb import ROOT
from wadidaw.models.audio_cnn import compute_calibrated_mel_stats
from wadidaw.models.panns_encoder import (
    PANNS_EMBED_DIM,
    PANNsCnn14Encoder,
    default_checkpoint_path,
)


SAMPLES_ROOT = ROOT / "data" / "prompts"
CACHE_ROOT = ROOT / "data" / "cache" / "panns"

MANIFEST_SPECS = (
    ("train", "train_samples.json", "train_panns_samples.json"),
    ("validation", "validation_samples.json", "validation_panns_samples.json"),
    ("test_synthetic", "test_synthetic_samples.json", "test_synthetic_panns_samples.json"),
    ("test_human", "test_human_samples.json", "test_human_panns_samples.json"),
)


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_samples(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "samples" in data:
        return list(data["samples"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unrecognized sample manifest format: {path}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def _extract_split(
    records: list[dict[str, object]],
    encoder: PANNsCnn14Encoder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    progress_every: int = 20,
) -> np.ndarray:
    dataset = MusdbMultimodalDataset(records, cache_in_memory=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    features: list[np.ndarray] = []
    n_seen = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(loader, start=1):
        logmel = batch["logmel_db"].to(device=device, dtype=torch.float32)
        emb = encoder.encode_stems(logmel)  # (B, 4, 2048)

        bsz, n_stems, n_mels, frames = logmel.shape
        flat = logmel.reshape(bsz * n_stems, 1, n_mels, frames)
        normalised = (flat + 30.0) * 0.02  # matches dataset feature scaling
        _, broadband_db, _ = compute_calibrated_mel_stats(normalised)
        level = broadband_db.reshape(bsz, n_stems, 1)

        combined = torch.cat((emb, level), dim=-1).cpu().numpy().astype(np.float32)
        features.append(combined)
        n_seen += bsz

        if batch_idx % progress_every == 0 or batch_idx == len(loader):
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(
                f"    [{n_seen}/{len(records)}] {rate:5.1f} samples/s "
                f"({elapsed:5.1f}s)"
            )

        if max_samples is not None and n_seen >= max_samples:
            break

    if not features:
        raise RuntimeError("No samples were processed; check the manifest.")

    arr = np.concatenate(features, axis=0)
    if max_samples is not None:
        arr = arr[:max_samples]
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache frozen PANNs CNN14 features for M2."
    )
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"PANNs checkpoint (default: {default_checkpoint_path()}).")
    parser.add_argument("--no-download", action="store_true",
                        help="Never download the checkpoint; fail if missing.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Samples per forward pass (each sample = 4 stems).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", nargs="+",
                        choices=[spec[0] for spec in MANIFEST_SPECS],
                        default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per split (smoke test).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    encoder = PANNsCnn14Encoder(
        checkpoint_path=args.checkpoint,
        download=not args.no_download,
        freeze=True,
    ).to(device)
    print(
        f"[panns] checkpoint={encoder.checkpoint_path} "
        f"device={device} embed_dim={encoder.embed_dim}"
    )

    selected = set(args.splits) if args.splits else {spec[0] for spec in MANIFEST_SPECS}
    metadata: dict[str, object] = {
        "encoder": "PANNs Cnn14",
        "checkpoint": str(encoder.checkpoint_path),
        "embed_dim": PANNS_EMBED_DIM,
        "feature_layout": "[:2048]=embedding, [2048]=broadband_lmp_db",
        "splits": {},
    }

    for split_name, input_name, output_name in MANIFEST_SPECS:
        if split_name not in selected:
            continue
        input_path = args.samples_root / input_name
        if not input_path.is_file():
            print(f"[skip] manifest not found: {input_path}")
            continue

        cache_path = args.cache_root / f"{split_name}.npy"
        output_path = args.samples_root / output_name
        if cache_path.is_file() and not args.overwrite:
            print(f"[skip] cache exists: {cache_path} (use --overwrite)")
            continue

        records = _load_samples(input_path)
        print(f"[{split_name}] extracting {len(records)} samples...")
        features = _extract_split(
            records=records,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        )
        np.save(cache_path, features)
        print(
            f"[{split_name}] wrote {cache_path} shape={features.shape} "
            f"({cache_path.stat().st_size / 1e6:.1f} MB)"
        )

        rel_cache = _relative_to_root(cache_path)
        output_samples: list[dict[str, object]] = []
        for idx, record in enumerate(records):
            if idx >= len(features):
                break
            updated = dict(record)
            updated["panns_path"] = rel_cache
            updated["panns_index"] = int(idx)
            updated["panns_embed_dim"] = int(PANNS_EMBED_DIM)
            updated["panns_level_features"] = 1
            output_samples.append(updated)

        output_path.write_text(
            json.dumps(output_samples, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[{split_name}] wrote {len(output_samples)} samples -> {output_path}")

        metadata["splits"][split_name] = {
            "cache": rel_cache,
            "manifest": _relative_to_root(output_path),
            "n_samples": int(len(output_samples)),
            "shape": list(features.shape),
        }

    metadata_path = args.cache_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nDone. Metadata -> {metadata_path}")
    print("Train M2 with: --panns --text-backend huggingface")


if __name__ == "__main__":
    main()