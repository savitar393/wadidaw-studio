import json
from pathlib import Path

import numpy as np

from wadidaw.music_env import DATASET


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "splits" / "babyslakh_v1.json"


def main():
    song_dirs = sorted(
        path
        for path in DATASET.rglob("stems")
        if path.is_dir()
        and path.parent.name.startswith("Track")
    )

    if len(song_dirs) != 20:
        raise RuntimeError(
            f"Expected 20 BabySlakh songs, found {len(song_dirs)}"
        )

    # Fixed split seed: reproducible across runs.
    rng = np.random.default_rng(2026)
    indices = rng.permutation(len(song_dirs))

    groups = {
        "train": indices[:12],
        "validation": indices[12:16],
        "test": indices[16:20],
    }

    manifest = {
        "dataset": "BabySlakh 16k",
        "seed": 2026,
        "splits": {
            split: [
                {
                    "index": int(i),
                    "song": song_dirs[int(i)].parent.name,
                }
                for i in members
            ]
            for split, members in groups.items()
        },
    }

    DESTINATION.parent.mkdir(
        parents=True, exist_ok=True
    )

    if DESTINATION.exists():
        existing = json.loads(
            DESTINATION.read_text()
        )
        if existing != manifest:
            raise RuntimeError(
                "An incompatible split manifest already exists."
            )
    else:
        DESTINATION.write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    for split, members in manifest["splits"].items():
        names = [item["song"] for item in members]
        print(f"{split.upper()} ({len(names)} songs)")
        print(" ", ", ".join(names))

    print("\nSplit manifest:", DESTINATION)


if __name__ == "__main__":
    main()
