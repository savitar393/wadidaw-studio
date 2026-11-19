"""Compare M1 runs from ``outputs/training/*_results.json``.

Two modes are supported:

1. **Pairwise** (default): baseline vs. one candidate, per seed, with mean/std
   summaries. This is what E1/E2 vs. the old E0 baseline used.
2. **Leaderboard** (``--runs``): many candidate patterns at once, aggregated per
   split over whichever seeds are present. Handy for the hyper-parameter tuning
   runs, which are usually executed with a single seed.

Examples:
    python -m scripts.compare_m1 \
        --baseline "m1_seed{seed}" \
        --candidate "m1e2_m1_seed{seed}" \
        --out outputs/m1_e2_vs_e0.csv

    python -m scripts.compare_m1 \
        --baseline "m1e2_m1_seed{seed}" --seeds 42 \
        --runs "tune_ep50_m1_seed{seed},tune_drop30_m1_seed{seed},tune_hid512_m1_seed{seed}" \
        --out outputs/m1_tuning_leaderboard.csv
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

DELTA_METRICS: tuple[str, ...] = (
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


def run_pairwise(args: argparse.Namespace) -> None:
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


def _aggregate(
    training_dir: Path,
    pattern: str,
    seeds: list[int],
    split: str,
) -> tuple[list[int], dict[str, list[float]]] | None:
    """Collect metrics across the seeds that exist for this run pattern."""
    collected: dict[str, list[float]] = {m: [] for m in METRICS}
    seeds_found: list[int] = []
    for seed in seeds:
        try:
            metrics = load_split_metrics(training_dir, pattern, seed, split)
        except FileNotFoundError:
            continue
        seeds_found.append(seed)
        for metric in METRICS:
            collected[metric].append(float(metrics[metric]))
    if not seeds_found:
        return None
    return seeds_found, collected


def run_leaderboard(args: argparse.Namespace) -> None:
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not runs:
        raise SystemExit("--runs was provided but contains no run patterns.")

    rows: list[dict[str, object]] = []

    for split in args.splits:
        print(f"\n=== {split} ===")
        per_run: list[tuple[float, dict[str, object]]] = []

        base_agg = _aggregate(args.training_dir, args.baseline, args.seeds, split)

        for run in runs:
            agg = _aggregate(args.training_dir, run, args.seeds, split)
            if agg is None:
                print(f"  [skip] {run}: no results found for seeds {args.seeds}")
                continue
            seeds_found, collected = agg

            # Compare against the baseline restricted to the same seeds.
            base_collected: dict[str, list[float]] | None = None
            if base_agg is not None:
                _, base_all = base_agg
                matched = [s for s in base_agg[0] if s in seeds_found]
                if matched:
                    base_collected = {m: [] for m in METRICS}
                    for seed in matched:
                        metrics = load_split_metrics(args.training_dir, args.baseline, seed, split)
                        for metric in METRICS:
                            base_collected[metric].append(float(metrics[metric]))

            row: dict[str, object] = {
                "split": split,
                "run": run,
                "n_seeds": len(seeds_found),
                "seeds": " ".join(str(s) for s in seeds_found),
            }
            for metric in METRICS:
                row[metric] = round(mean(collected[metric]), 4)
                if metric in STD_METRICS:
                    row[f"{metric}_std"] = round(
                        pstdev(collected[metric]) if len(collected[metric]) > 1 else 0.0,
                        4,
                    )
            for metric in DELTA_METRICS:
                if base_collected is not None:
                    row[f"delta_{metric}"] = round(
                        mean(collected[metric]) - mean(base_collected[metric]), 4
                    )
                else:
                    row[f"delta_{metric}"] = ""
            rows.append(row)
            per_run.append((float(row["mae_db"]), row))

        if base_agg is not None and per_run:
            print(
                f"  baseline {args.baseline} "
                f"(seeds {base_agg[0]}) MAE {mean(base_agg[1]['mae_db']):.4f}"
            )
        for mae, row in sorted(per_run, key=lambda item: item[0]):
            print(
                f"  {row['run']:34s} MAE {row['mae_db']:.4f}"
                f" ({row['delta_mae_db']:+} vs base)"
                f" | Succ {row['success_rate_1db']:.1f}%"
                f" | Adh {row['instruction_adherence']:.1f}%"
                f" | seeds {row['seeds']}"
            )

    fieldnames = ["split", "run", "n_seeds", "seeds"]
    for metric in METRICS:
        fieldnames.append(metric)
        if metric in STD_METRICS:
            fieldnames.append(f"{metric}_std")
    fieldnames += [f"delta_{metric}" for metric in DELTA_METRICS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nLeaderboard table -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare M1 runs: pairwise (default) or leaderboard (--runs)."
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
        help="Candidate run name pattern (pairwise mode); use {seed} as placeholder.",
    )
    parser.add_argument(
        "--runs",
        default=None,
        help="Comma-separated run patterns for leaderboard mode; each may use {seed}.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test_human", "test_synthetic"],
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/m1_comparison.csv"))
    args = parser.parse_args()

    if args.runs:
        run_leaderboard(args)
    else:
        run_pairwise(args)


if __name__ == "__main__":
    main()
