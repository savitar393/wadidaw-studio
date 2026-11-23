"""Merge per-model result rows into the team-wide ``outputs/main_results.csv``.

``scripts/train_models.py`` only writes rows for the models it actually trains,
so running M2 alone would otherwise drop M1/M3/M4/M5.  This script replaces the
rows of selected models in the target table and leaves every other row intact.

Example:
    python -m scripts.merge_main_results \
        --source outputs/m2_final/main_results.csv --models m2 \
        --target outputs/main_results.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from wadidaw.data.musdb import ROOT


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge model rows into main_results.csv.")
    parser.add_argument("--source", type=Path, required=True,
                        help="New results CSV (e.g. outputs/m2_final/main_results.csv).")
    parser.add_argument("--target", type=Path, default=ROOT / "outputs" / "main_results.csv")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model keys to replace, e.g. m2 or m1 m1_audio_only.")
    parser.add_argument("--backup", action="store_true",
                        help="Write a .bak copy of the target before overwriting.")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Source CSV not found: {args.source}")
    if not args.target.is_file():
        raise SystemExit(f"Target CSV not found: {args.target}")

    target_fields, target_rows = _read_rows(args.target)
    source_fields, source_rows = _read_rows(args.source)

    missing = [field for field in target_fields if field not in source_fields]
    if missing:
        raise SystemExit(
            f"Source {args.source} is missing target columns: {missing}"
        )

    models = set(args.models)
    replacements = {str(row["model"]) for row in source_rows} & models
    absent = models - replacements
    if absent:
        raise SystemExit(
            f"Source does not contain rows for: {sorted(absent)}. "
            f"Available: {sorted({str(row['model']) for row in source_rows})}"
        )

    new_rows = [row for row in target_rows if str(row["model"]) not in models]
    new_rows.extend(
        {field: row.get(field, "") for field in target_fields}
        for row in source_rows
        if str(row["model"]) in models
    )

    if args.backup:
        backup = args.target.with_suffix(args.target.suffix + ".bak")
        shutil.copy2(args.target, backup)
        print(f"Backup -> {backup}")

    with args.target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(new_rows)

    print(
        f"Updated {args.target}: replaced {len(replacements)} model(s) "
        f"({', '.join(sorted(models))}), {len(new_rows)} rows total."
    )


if __name__ == "__main__":
    main()