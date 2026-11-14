"""Extract and cache OpenL3 embeddings for M4-OpenL3.

This script reads the existing multimodal sample manifests, extracts one
OpenL3 embedding per unique audio excerpt, and writes new manifests that
contain an additional ``openl3_path`` field.  The original log-mel manifests
are left untouched.

Example:
    python -m scripts.build_openl3_cache --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wadidaw.data.musdb import MUSDB_ROOT, ROOT, load_split_manifest
from wadidaw.data.openl3_features import (
    OPENL3_EMBED_DIM,
    OPENL3_FRAMES,
    extract_openl3_for_record,
)


SAMPLES_ROOT = ROOT / "data" / "prompts"
OPENL3_CACHE_ROOT = ROOT / "data" / "cache" / "openl3"


MANIFEST_SPECS = (
    ("train", "train_samples.json", "train_openl3_samples.json", "train"),
    ("validation", "validation_samples.json", "validation_openl3_samples.json", "validation"),
    ("test", "test_synthetic_samples.json", "test_synthetic_openl3_samples.json", "test"),
    ("test_human", "test_human_samples.json", "test_human_openl3_samples.json", "test"),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenL3 embeddings and write M4-OpenL3 sample manifests."
    )
    parser.add_argument("--musdb-root", type=Path, default=MUSDB_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "data" / "splits" / "musdb18hq_v1.json")
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--cache-root", type=Path, default=OPENL3_CACHE_ROOT)
    parser.add_argument("--content-type", choices=("music", "env"), default="music")
    parser.add_argument("--input-repr", choices=("linear", "mel128", "mel256"), default="mel256")
    parser.add_argument("--embedding-size", type=int, choices=(512, 6144), default=OPENL3_EMBED_DIM)
    parser.add_argument("--frames", type=int, default=OPENL3_FRAMES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-excerpts", type=int, default=None, help="Optional cap for a smoke test.")
    parser.add_argument("--verbose-openl3", action="store_true")
    args = parser.parse_args()

    split_manifest = load_split_manifest(args.split_manifest)
    song_dirs: dict[str, Path] = {}
    for split_name, members in split_manifest["splits"].items():
        for member in members:
            song_dirs[str(member["song"])] = args.musdb_root / str(member["relative_dir"])

    for split_name, input_name, output_name, cache_split in MANIFEST_SPECS:
        input_path = args.samples_root / input_name
        if not input_path.is_file():
            print(f"[skip] manifest not found: {input_path}")
            continue

        samples = _load_samples(input_path)
        unique: dict[str, dict[str, object]] = {}
        for record in samples:
            excerpt_id = Path(str(record["audio_path"])).stem
            unique.setdefault(excerpt_id, record)

        excerpt_items = list(unique.items())
        if args.max_excerpts is not None:
            excerpt_items = excerpt_items[: args.max_excerpts]
        selected_ids = {excerpt_id for excerpt_id, _ in excerpt_items}

        print(f"[{split_name}] extracting {len(excerpt_items)} unique excerpts...")
        cache_dir = args.cache_root / cache_split
        cache_dir.mkdir(parents=True, exist_ok=True)

        for idx, (excerpt_id, record) in enumerate(excerpt_items, start=1):
            out_path = cache_dir / f"{excerpt_id}.npy"
            if out_path.is_file() and not args.overwrite:
                print(f"  [{idx}/{len(excerpt_items)}] cached {out_path.name}")
                continue

            song = str(record["song"])
            if song not in song_dirs:
                raise KeyError(f"Song not found in split manifest: {song}")
            embedding = extract_openl3_for_record(
                record=record,
                song_dir=song_dirs[song],
                content_type=args.content_type,
                input_repr=args.input_repr,
                embedding_size=args.embedding_size,
                target_frames=args.frames,
                verbose=args.verbose_openl3,
            )
            np.save(out_path, embedding.astype(np.float16))
            print(
                f"  [{idx}/{len(excerpt_items)}] {out_path.name} "
                f"{tuple(embedding.shape)}"
            )

        output_samples: list[dict[str, object]] = []
        for record in samples:
            excerpt_id = Path(str(record["audio_path"])).stem
            if excerpt_id not in selected_ids:
                continue
            openl3_path = args.cache_root / cache_split / f"{excerpt_id}.npy"
            updated = dict(record)
            updated["openl3_path"] = _relative_to_root(openl3_path)
            updated["openl3_frames"] = int(args.frames)
            updated["openl3_embed_dim"] = int(args.embedding_size)
            output_samples.append(updated)

        output_path = args.samples_root / output_name
        output_path.write_text(
            json.dumps(output_samples, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"[{split_name}] wrote {len(output_samples)} samples -> {output_path}")

    print("\nDone. Use the *_openl3_samples.json manifests with an M4-OpenL3 config.")


if __name__ == "__main__":
    main()
