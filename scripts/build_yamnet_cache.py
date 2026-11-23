"""Build a frozen YAMNet embedding cache for M2.

YAMNet (TF Hub ``google/yamnet``) is run once over every unique audio excerpt.
For each excerpt and each of the 4 stems we store the time-pooled YAMNet
embedding (``mean`` -> 1024-d or ``mean+max`` -> 2048-d).  Per-sample rows are
then assembled by appending the per-stem broadband loudness (dBFS) computed from
the g0-perturbed log-mel, so the input gain stays observable to the model:

    features[i] = [ pooled_embedding (D), broadband_lmp_db ]   shape (4, D + 1)

Because the same excerpt is reused by several prompts, YAMNet is evaluated once
per unique excerpt and the embeddings are reused across prompts.

Outputs
-------
- ``data/cache/yamnet/{split}.npy``            ``(N, 4, D + 1)`` float32
- ``data/cache/yamnet/metadata.json``
- ``data/prompts/{split}_yamnet_samples.json`` manifests with ``yamnet_path``
  and ``yamnet_index`` so ``MusdbMultimodalDataset`` can load the right row.

Example:
    python -m scripts.build_yamnet_cache --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wadidaw.data.features import apply_gain_to_logmel_db
from wadidaw.data.musdb import MUSDB_ROOT, ROOT, SPLIT_MANIFEST_PATH, load_split_manifest
from wadidaw.data.prompts import STEM_NAMES
from wadidaw.data.yamnet_features import (
    YAMNET_EMBED_DIM,
    YAMNET_POOL_DIM,
    extract_yamnet_for_record,
    load_yamnet_model,
    resolve_yamnet_handle,
)
from wadidaw.models.audio_cnn import compute_calibrated_mel_stats


SAMPLES_ROOT = ROOT / "data" / "prompts"
CACHE_ROOT = ROOT / "data" / "cache" / "yamnet"

MANIFEST_SPECS = (
    ("train", "train_samples.json", "train_yamnet_samples.json"),
    ("validation", "validation_samples.json", "validation_yamnet_samples.json"),
    ("test_synthetic", "test_synthetic_samples.json", "test_synthetic_yamnet_samples.json"),
    ("test_human", "test_human_samples.json", "test_human_yamnet_samples.json"),
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


def _stem_level_db(record: dict[str, object], root: Path = ROOT) -> np.ndarray:
    """Per-stem broadband loudness (dBFS) from the g0-perturbed log-mel cache."""
    raw = np.load(root / str(record["audio_path"])).astype(np.float32)
    g0 = np.asarray(record["g0_db"], dtype=np.float32)
    mask = np.asarray(record.get("active_mask", [True, True, True, True]), dtype=bool)
    effective_g0 = np.where(mask, g0, 0.0).astype(np.float32)
    logmel_perturbed_db = apply_gain_to_logmel_db(raw, effective_g0)
    normalised = (logmel_perturbed_db + 30.0) * 0.02
    tensor = torch.from_numpy(normalised).reshape(4, 1, normalised.shape[1], normalised.shape[2])
    _, broadband_db, _ = compute_calibrated_mel_stats(tensor)
    return broadband_db.reshape(4).numpy().astype(np.float32)


def _excerpt_id(record: dict[str, object]) -> str:
    return Path(str(record["audio_path"])).stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen YAMNet embeddings for M2.")
    parser.add_argument("--musdb-root", type=Path, default=MUSDB_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--yamnet-model", default=None,
                        help=f"YAMNet TF Hub handle or local path (default: {resolve_yamnet_handle()}).")
    parser.add_argument("--pool", choices=sorted(YAMNET_POOL_DIM), default="mean_max")
    parser.add_argument("--splits", nargs="+",
                        choices=[spec[0] for spec in MANIFEST_SPECS],
                        default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-excerpts", type=int, default=None,
                        help="Cap unique excerpts per split (smoke test).")
    args = parser.parse_args()

    args.cache_root.mkdir(parents=True, exist_ok=True)

    split_manifest = load_split_manifest(args.split_manifest)
    song_dirs: dict[str, Path] = {}
    for members in split_manifest["splits"].values():
        for member in members:
            song_dirs[str(member["song"])] = args.musdb_root / str(member["relative_dir"])

    model = load_yamnet_model(args.yamnet_model)
    embed_dim = YAMNET_POOL_DIM[args.pool]
    print(f"[yamnet] model={resolve_yamnet_handle(args.yamnet_model)} pool={args.pool} dim={embed_dim}")

    selected = set(args.splits) if args.splits else {spec[0] for spec in MANIFEST_SPECS}
    embedding_cache: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "encoder": "YAMNet",
        "handle": resolve_yamnet_handle(args.yamnet_model),
        "pool": args.pool,
        "embed_dim": embed_dim,
        "feature_layout": f"[:{embed_dim}]=pooled_embedding, [{embed_dim}]=broadband_lmp_db",
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
        unique_ids: list[str] = []
        seen: set[str] = set()
        for record in records:
            excerpt_id = _excerpt_id(record)
            if excerpt_id not in seen:
                seen.add(excerpt_id)
                unique_ids.append(excerpt_id)
        if args.max_excerpts is not None:
            unique_ids = unique_ids[: args.max_excerpts]
            allowed = set(unique_ids)
        else:
            allowed = None

        print(f"[{split_name}] extracting {len(unique_ids)} unique excerpts for {len(records)} records...")
        t0 = time.time()
        by_id = {_excerpt_id(r): r for r in records}
        for idx, excerpt_id in enumerate(unique_ids, start=1):
            if excerpt_id in embedding_cache:
                continue
            record = by_id[excerpt_id]
            song = str(record["song"])
            if song not in song_dirs:
                raise KeyError(f"Song not found in split manifest: {song}")
            embedding_cache[excerpt_id] = extract_yamnet_for_record(
                record=record,
                song_dir=song_dirs[song],
                model=model,
                pool=args.pool,
            )
            if idx % 20 == 0 or idx == len(unique_ids):
                rate = idx / max(time.time() - t0, 1e-6)
                print(f"  [{idx}/{len(unique_ids)}] {rate:5.1f} excerpts/s")

        rows: list[np.ndarray] = []
        kept: list[dict[str, object]] = []
        for record in records:
            excerpt_id = _excerpt_id(record)
            if allowed is not None and excerpt_id not in allowed:
                continue
            embedding = embedding_cache[excerpt_id]
            level = _stem_level_db(record)[:, None]
            rows.append(np.concatenate((embedding, level), axis=1))
            kept.append(record)

        if not rows:
            print(f"[skip] no rows for {split_name}")
            continue

        features = np.stack(rows, axis=0).astype(np.float32)
        np.save(cache_path, features)
        print(
            f"[{split_name}] wrote {cache_path} shape={features.shape} "
            f"({cache_path.stat().st_size / 1e6:.1f} MB)"
        )

        rel_cache = _relative_to_root(cache_path)
        output_samples: list[dict[str, object]] = []
        for idx, record in enumerate(kept):
            updated = dict(record)
            updated["yamnet_path"] = rel_cache
            updated["yamnet_index"] = int(idx)
            updated["yamnet_embed_dim"] = int(embed_dim)
            updated["yamnet_pool"] = args.pool
            updated["yamnet_level_features"] = 1
            output_samples.append(updated)

        output_path.write_text(json.dumps(output_samples, indent=2, ensure_ascii=False) + "\n")
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
    print("Train M2 with: --yamnet --text-backend huggingface")


if __name__ == "__main__":
    main()
