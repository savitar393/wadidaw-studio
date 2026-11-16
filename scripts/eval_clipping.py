"""Audio clipping / peak evaluation for M2.

Predictions are applied to the original 44.1 kHz stereo stems and rendered with
``render_gain_staged_mix`` to measure the actual output peak level and clipping
rate.  Three conditions are compared per sample:

- ``m2_pred``   : predicted gains
- ``true_gain`` : ground-truth gains (oracle)
- ``no_op``     : 0 dB on every active stem

Example:
    python -m scripts.eval_clipping --run-prefix m2final --panns \
        --split test_human --out outputs/m2_final/clipping.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.utils.data import DataLoader

from wadidaw.data.dataset import SAMPLES_ROOT, MusdbMultimodalDataset
from wadidaw.data.musdb import (
    MUSDB_ROOT,
    ROOT,
    SPLIT_MANIFEST_PATH,
    load_excerpt_stems_44k,
    load_split_manifest,
    render_gain_staged_mix,
)
from wadidaw.models.fusion import build_model


TRAINING_DIR = ROOT / "outputs" / "training"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"

SPLIT_MANIFESTS = {
    "validation": "validation_samples.json",
    "test_human": "test_human_samples.json",
    "test_synthetic": "test_synthetic_samples.json",
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _manifest_path(split: str, panns: bool) -> Path:
    name = SPLIT_MANIFESTS[split]
    if panns:
        name = name.replace("_samples.json", "_panns_samples.json")
    return SAMPLES_ROOT / name


def _song_dirs() -> dict[str, Path]:
    manifest = load_split_manifest(SPLIT_MANIFEST_PATH)
    mapping: dict[str, Path] = {}
    for members in manifest["splits"].values():
        for member in members:
            mapping[str(member["song"])] = MUSDB_ROOT / str(member["relative_dir"])
    return mapping


def _load_model(run_name: str, device: torch.device) -> torch.nn.Module:
    results_path = TRAINING_DIR / f"{run_name}_results.json"
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    if not results_path.is_file():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    results = json.loads(results_path.read_text())
    config = results.get("config", {})
    model = build_model(
        str(config.get("model_name", "m2")),
        text_backend=str(config.get("text_backend", "subword")),
        **(config.get("model_kwargs") or {}),
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def _predict_all(
    model: torch.nn.Module,
    records: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = MusdbMultimodalDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    audio_key = getattr(model, "audio_input_key", "audio")
    predictions: list[np.ndarray] = []
    for batch in loader:
        audio = batch[audio_key].to(device=device, dtype=torch.float32)
        mask = batch["active_mask"].to(device=device)
        outputs = model(audio, batch["prompt"], active_mask=mask, clamp_output=True)
        predictions.append(outputs["gains_pred"].detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _render_metrics(
    stems: dict[str, np.ndarray],
    gains: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, float]:
    result = render_gain_staged_mix(
        stems_stereo=stems,
        gains_db=gains,
        active_mask=active_mask,
    )
    metrics = result["metrics"]
    return {
        "peak_linear": float(metrics["peak_linear"]),
        "peak_dbfs": float(metrics["peak_dbfs"]),
        "clipping_rate_pct": float(metrics["clipping_rate"]) * 100.0,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["eval_split"]), str(row["condition"]))].append(row)

    out: list[dict[str, object]] = []
    for (split, condition), group in grouped.items():
        peaks = [float(item["peak_dbfs"]) for item in group]
        clips = [float(item["clipping_rate_pct"]) for item in group]
        out.append(
            {
                "eval_split": split,
                "condition": condition,
                "n_samples": len(group),
                "peak_dbfs_mean": round(mean(peaks), 3),
                "peak_dbfs_max": round(max(peaks), 3),
                "clipping_rate_mean_pct": round(mean(clips), 4),
                "clipping_rate_max_pct": round(max(clips), 4),
                "samples_with_clipping_pct": round(
                    100.0 * sum(1 for value in clips if value > 0.0) / len(clips), 2
                ),
            }
        )
    out.sort(key=lambda row: (row["eval_split"], row["condition"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 audio clipping / peak evaluation.")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--split", default="test_human", choices=sorted(SPLIT_MANIFESTS))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "m2_clipping.csv")
    parser.add_argument("--per-sample-out", type=Path, default=None)
    parser.add_argument("--panns", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N samples (smoke test).")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    manifest = _manifest_path(args.split, args.panns)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    records = json.loads(manifest.read_text())
    if isinstance(records, dict) and "samples" in records:
        records = records["samples"]
    if args.limit is not None:
        records = records[: args.limit]

    song_dirs = _song_dirs()
    per_sample_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        run_name = f"{args.run_prefix}_m2_seed{seed}"
        if not (TRAINING_DIR / f"{run_name}_results.json").is_file():
            print(f"[skip] run not found: {run_name}")
            continue
        print(f"[run] {run_name} on {args.split} ({len(records)} samples)")
        model = _load_model(run_name, device)
        predictions = _predict_all(model, records, device, args.batch_size)

        for idx, record in enumerate(records):
            song = str(record["song"])
            if song not in song_dirs:
                raise KeyError(f"Song not found in split manifest: {song}")
            stems = load_excerpt_stems_44k(
                song_dir=song_dirs[song],
                start_sec=float(record["excerpt_start_sec"]),
                duration_sec=float(record.get("duration_sec", 8.0)),
            )
            active_mask = np.asarray(
                record.get("active_mask", [True, True, True, True]), dtype=bool
            )
            true_gains = np.asarray(record["gains_db"], dtype=np.float32)

            conditions = {
                "m2_pred": predictions[idx],
                "true_gain": true_gains,
                "no_op": np.zeros(4, dtype=np.float32),
            }
            for condition, gains in conditions.items():
                metrics = _render_metrics(stems, gains, active_mask)
                per_sample_rows.append(
                    {
                        "eval_split": args.split,
                        "seed": seed,
                        "id": str(record["id"]),
                        "song": song,
                        "condition": condition,
                        "peak_dbfs": round(metrics["peak_dbfs"], 3),
                        "clipping_rate_pct": round(metrics["clipping_rate_pct"], 4),
                    }
                )
            if (idx + 1) % 25 == 0 or idx + 1 == len(records):
                print(f"  rendered {idx + 1}/{len(records)}")

    if not per_sample_rows:
        raise SystemExit(f"No runs found for prefix '{args.run_prefix}'.")

    # Aggregate again over all seeds so the CSV has one row per (split, condition).
    final_rows = _aggregate(per_sample_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    if args.per_sample_out is not None:
        args.per_sample_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_rows)

    print("\n=== CLIPPING SUMMARY ===")
    for row in final_rows:
        print(
            f"{row['eval_split']:12s} | {row['condition']:10s} | "
            f"peak mean {row['peak_dbfs_mean']:7.2f} dBFS | max {row['peak_dbfs_max']:7.2f} | "
            f"clip mean {row['clipping_rate_mean_pct']:.4f}% | "
            f"samples clipped {row['samples_with_clipping_pct']:.2f}%"
        )
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()