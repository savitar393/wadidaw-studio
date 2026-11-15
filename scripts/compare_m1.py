"""Compare baseline vs candidate M1 runs from ``outputs/training/*_results.json``.

Reports per-seed deltas as well as mean/std summaries so that improvements can be
attributed consistently instead of relying on a single aggregate number.

Example:
    python -m scripts.compare_m1 \
        --baseline "m1_seed{seed}" \
        --candidate "m1e2_m1_seed{seed}" \
        --out outputs/m1_e2_vs_e0.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS: tuple[str, ...] = (
    "mae_db",
    "rmse_db",
    "bias_db",
    "band_mae_db",
    "band_mse_norm",
    "log_spectral_distance_db",
    "success_rate_1db",
    "balance_mae_db",
    "instruction_adherence",
)

STD_METRICS: tuple[str, ...] = (
    "mae_db",
    "band_mae_db",
    "success_rate_1db",
    "instruction_adherence",
)


def load_split_metrics(
    training_dir: Path,
    pattern: str,
    seed: int,
    split: str,
) -> dict[str, object]:
    """Load the metrics dictionary for one run/split from its results JSON."""
    path = training_dir / f"{pattern.format(seed=seed)}_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run results not found: {path}")
    data = json.loads(path.read_text())
    metrics = data.get(split)
    if not isinstance(metrics, dict):
        raise KeyError(f"{path} has no metrics for split '{split}'")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate M1 runs per seed."
    )
    parser.add_argument("--training-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument(
        "--baseline",
        default="m1_seed{seed}",
        help="Baseline run name pattern; use {seed} as placeholder.",
    )
    parser.add_argument(
        "--candidate",
        default="m1e2_m1_seed{seed}",
        help="Candidate run name pattern; use {seed} as placeholder.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test_human", "test_synthetic"],
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/m1_comparison.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []

    for split in args.splits:
        base_vals: dict[str, list[float]] = {m: [] for m in METRICS}
        cand_vals: dict[str, list[float]] = {m: [] for m in METRICS}

        for seed in args.seeds:
            base = load_split_metrics(args.training_dir, args.baseline, seed, split)
            cand = load_split_metrics(args.training_dir, args.candidate, seed, split)

            row: dict[str, object] = {"split": split, "seed": seed}
            for metric in METRICS:
                b = float(base[metric])
                c = float(cand[metric])
                row[f"base_{metric}"] = round(b, 4)
                row[f"cand_{metric}"] = round(c, 4)
                row[f"delta_{metric}"] = round(c - b, 4)
                base_vals[metric].append(b)
                cand_vals[metric].append(c)
            rows.append(row)

        summary: dict[str, object] = {"split": split, "seed": "mean"}
        for metric in METRICS:
            summary[f"base_{metric}"] = round(mean(base_vals[metric]), 4)
            summary[f"cand_{metric}"] = round(mean(cand_vals[metric]), 4)
            summary[f"delta_{metric}"] = round(
                mean(cand_vals[metric]) - mean(base_vals[metric]), 4
            )
        for metric in STD_METRICS:
            summary[f"base_{metric}_std"] = round(pstdev(base_vals[metric]), 4)
            summary[f"cand_{metric}_std"] = round(pstdev(cand_vals[metric]), 4)
        rows.append(summary)

        print(
            f"{split:14s} | "
            f"MAE {summary['base_mae_db']:.4f} -> {summary['cand_mae_db']:.4f} "
            f"({summary['delta_mae_db']:+.4f}) | "
            f"Succ<1dB {summary['base_success_rate_1db']:.1f}% -> "
            f"{summary['cand_success_rate_1db']:.1f}% | "
            f"Adherence {summary['base_instruction_adherence']:.1f}% -> "
            f"{summary['cand_instruction_adherence']:.1f}%"
        )

    fieldnames = ["split", "seed"]
    for metric in METRICS:
        fieldnames += [f"base_{metric}", f"cand_{metric}", f"delta_{metric}"]
    for metric in STD_METRICS:
        fieldnames += [f"base_{metric}_std", f"cand_{metric}_std"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison table -> {args.out}")


if __name__ == "__main__":
    main()
