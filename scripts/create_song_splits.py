"""Create the MUSDB18-HQ split manifest (80/20/50) and Indonesian prompt banks."""

from __future__ import annotations

import argparse
from pathlib import Path

from wadidaw.data.musdb import (
    MUSDB_ROOT,
    SPLIT_MANIFEST_PATH,
    create_musdb_split_manifest,
)
from wadidaw.data.prompts import save_prompt_banks


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "data" / "prompts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MUSDB18-HQ song splits (80/20/50) and Indonesian prompt banks."
    )
    parser.add_argument("--musdb-root", type=Path, default=MUSDB_ROOT)
    parser.add_argument("--output", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = create_musdb_split_manifest(
        musdb_root=args.musdb_root,
        output_path=args.output,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    for split, members in manifest["splits"].items():
        artists = {item["artist"] for item in members}
        print(f"{split.upper():10s}: {len(members):3d} songs ({len(artists):2d} unique artists)")

    prompt_paths = save_prompt_banks(args.prompts_dir)
    print("\nSaved split manifest:", args.output)
    for key, path in prompt_paths.items():
        print(f"Saved prompt bank [{key}]: {path}")


if __name__ == "__main__":
    main()
