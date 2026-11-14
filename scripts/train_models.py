"""Train and evaluate multimodal gain staging models (M1–M5, ablations, and DSP baselines)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from wadidaw.data.dataset import MusdbMultimodalDataset, SAMPLES_ROOT
from wadidaw.data.musdb import ROOT
from wadidaw.models.fusion import MODEL_REGISTRY
from wadidaw.training.evaluate import evaluate_model, make_baseline_predictor
from wadidaw.training.train import TrainConfig, train_one_run


OUTPUTS_DIR = ROOT / "outputs"


def _flatten_metric_row(
    model_key: str,
    model_title: str,
    split_label: str,
    seed: int | str,
    metrics: dict[str, object],
) -> dict[str, object]:
    per_stem = metrics.get("mae_per_stem_db", {})
    return {
        "model": model_key,
        "title": model_title,
        "eval_split": split_label,
        "seed": seed,
        "mae_db": metrics["mae_db"],
        "rmse_db": metrics["rmse_db"],
        "bias_db": metrics["bias_db"],
        "r2": metrics["r2"],
        "pearson_r": metrics["pearson_r"],
        "spearman_rho": metrics["spearman_rho"],
        "max_error_db": metrics["max_error_db"],
        "mae_vocals_db": per_stem.get("vocals", 0.0),
        "mae_drums_db": per_stem.get("drums", 0.0),
        "mae_bass_db": per_stem.get("bass", 0.0),
        "mae_other_db": per_stem.get("other", 0.0),
        "band_mae_db": metrics["band_mae_db"],
        "band_mse_norm": metrics["band_mse_norm"],
        "log_spectral_distance_db": metrics["log_spectral_distance_db"],
        "worst_band_error_db": metrics["worst_band_error_db"],
        "success_rate_1db": metrics["success_rate_1db"],
        "success_rate_band_001": metrics["success_rate_band_001"],
        "balance_mae_db": metrics["balance_mae_db"],
        "intent_direction_acc": metrics["intent_direction_acc"],
        "instruction_adherence": metrics["instruction_adherence"],
    }


def evaluate_baselines_table(
    val_manifest: Path,
    test_human_manifest: Path,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    val_loader = DataLoader(MusdbMultimodalDataset(val_manifest), batch_size=batch_size, shuffle=False)
    human_loader = (
        DataLoader(MusdbMultimodalDataset(test_human_manifest), batch_size=batch_size, shuffle=False)
        if test_human_manifest.is_file()
        else None
    )

    baseline_defs = (
        ("no_op", "Baseline: No-Op (0 dB)"),
        ("random", "Baseline: Random Gain"),
        ("dsp_equal_loudness", "Baseline: Audio-Only Equal Loudness DSP"),
    )
    for b_key, b_title in baseline_defs:
        predictor = make_baseline_predictor(b_key, seed=2026)
        val_m = evaluate_model(predictor, val_loader)
        rows.append(_flatten_metric_row(b_key, b_title, "validation", "deterministic", val_m))
        if human_loader is not None:
            hum_m = evaluate_model(predictor, human_loader)
            rows.append(_flatten_metric_row(b_key, b_title, "test_human", "deterministic", hum_m))

    return rows


def aggregate_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute `mean ± std` summary rows across seeds for each `(model, eval_split)`."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["model"]), str(r["title"]), str(r["eval_split"]))].append(r)

    numeric_cols = (
        "mae_db",
        "rmse_db",
        "bias_db",
        "r2",
        "pearson_r",
        "spearman_rho",
        "max_error_db",
        "mae_vocals_db",
        "mae_drums_db",
        "mae_bass_db",
        "mae_other_db",
        "band_mae_db",
        "band_mse_norm",
        "log_spectral_distance_db",
        "worst_band_error_db",
        "success_rate_1db",
        "success_rate_band_001",
        "balance_mae_db",
        "intent_direction_acc",
        "instruction_adherence",
    )

    summary_rows: list[dict[str, object]] = []
    for (model_key, title, split_label), group in grouped.items():
        agg: dict[str, object] = {
            "model": model_key,
            "title": title,
            "eval_split": split_label,
            "n_seeds": len(group),
        }
        for col in numeric_cols:
            vals = np.array([float(item[col]) for item in group], dtype=np.float64)
            agg[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{col}_std"] = round(float(np.std(vals)), 4)
        summary_rows.append(agg)

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate multimodal gain staging models on MUSDB18-HQ."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["m1", "m1_audio_only", "m1_text_only", "m2", "m3", "m4"],
        choices=sorted(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--text-backend",
        choices=("subword", "huggingface", "indobert"),
        default="subword",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-manifest", type=Path, default=SAMPLES_ROOT / "train_samples.json")
    parser.add_argument("--val-manifest", type=Path, default=SAMPLES_ROOT / "validation_samples.json")
    parser.add_argument("--test-human-manifest", type=Path, default=SAMPLES_ROOT / "test_human_samples.json")
    parser.add_argument("--test-synthetic-manifest", type=Path, default=SAMPLES_ROOT / "test_synthetic_samples.json")
    parser.add_argument(
        "--openl3",
        action="store_true",
        help="Use *_openl3_samples.json manifests produced by scripts/build_openl3_cache.py.",
    )
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.openl3:
        args.train_manifest = SAMPLES_ROOT / "train_openl3_samples.json"
        args.val_manifest = SAMPLES_ROOT / "validation_openl3_samples.json"
        args.test_human_manifest = SAMPLES_ROOT / "test_human_openl3_samples.json"
        args.test_synthetic_manifest = SAMPLES_ROOT / "test_synthetic_openl3_samples.json"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Evaluating non-learning DSP baselines...")
    all_rows = evaluate_baselines_table(
        val_manifest=args.val_manifest,
        test_human_manifest=args.test_human_manifest,
        batch_size=args.batch_size * 2,
    )

    for model_name in args.models:
        for seed in args.seeds:
            prefix = f"{args.run_prefix}_" if args.run_prefix else ""
            run_name = f"{prefix}{model_name}_seed{seed}"
            cfg = TrainConfig(
                model_name=model_name,
                run_name=run_name,
                seed=seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                text_backend=args.text_backend,
                device=args.device,
                train_manifest=str(args.train_manifest),
                val_manifest=str(args.val_manifest),
                test_human_manifest=str(args.test_human_manifest),
                test_synthetic_manifest=str(args.test_synthetic_manifest),
                overwrite=args.overwrite,
            )
            res = train_one_run(cfg)
            all_rows.append(
                _flatten_metric_row(
                    model_name,
                    MODEL_REGISTRY[model_name],
                    "validation",
                    seed,
                    res["validation"],
                )
            )
            if res.get("test_human") is not None:
                all_rows.append(
                    _flatten_metric_row(
                        model_name,
                        MODEL_REGISTRY[model_name],
                        "test_human",
                        seed,
                        res["test_human"],
                    )
                )
            if res.get("test_synthetic") is not None:
                all_rows.append(
                    _flatten_metric_row(
                        model_name,
                        MODEL_REGISTRY[model_name],
                        "test_synthetic",
                        seed,
                        res["test_synthetic"],
                    )
                )

    # Save per-run and aggregated CSV tables as specified in Section 10.3 & 15.4
    per_seed_path = OUTPUTS_DIR / "per_seed_results.csv"
    with per_seed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = aggregate_seed_rows(all_rows)
    main_csv_path = OUTPUTS_DIR / "main_results.csv"
    with main_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    m1_rows = [r for r in summary_rows if str(r["model"]).startswith("m1") or str(r["model"]).startswith("no_op") or str(r["model"]).startswith("dsp")]
    if m1_rows:
        m1_csv_path = OUTPUTS_DIR / "m1_results.csv"
        with m1_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(m1_rows[0].keys()))
            writer.writeheader()
            writer.writerows(m1_rows)

    print("\n=== SUMMARY TABLE (MEAN ± STD ACROSS SEEDS) ===")
    for row in summary_rows:
        print(
            f"{row['model']:16s} | {row['eval_split']:14s} | "
            f"MAE={row['mae_db_mean']:.3f}±{row['mae_db_std']:.3f} dB | "
            f"RMSE={row['rmse_db_mean']:.3f} dB | "
            f"BandMAE={row['band_mae_db_mean']:.3f} dB | "
            f"Succ<1dB={row['success_rate_1db_mean']:5.1f}% | "
            f"Adherence={row['instruction_adherence_mean']:5.1f}%"
        )
    print(f"\nSaved main results -> {main_csv_path}")


if __name__ == "__main__":
    main()
