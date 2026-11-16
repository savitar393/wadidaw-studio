"""Error analysis for trained M2 runs (per stem, per prompt family, per song).

The script loads the ``*_results.json`` config and ``*_best.pt`` checkpoint of
each requested run, collects per-sample predictions/ground truth on the chosen
splits, and aggregates parameter, spectral, success, adherence, and output
saturation metrics.

Outputs
-------
- ``outputs/error_analysis.md`` (or ``--report``)
- ``<out-dir>/error_analysis_overall.csv``
- ``<out-dir>/error_analysis_per_stem.csv``
- ``<out-dir>/error_analysis_per_prompt.csv``
- ``<out-dir>/error_analysis_per_song.csv``
- ``<out-dir>/error_analysis_worst_samples.csv``

Example:
    python -m scripts.error_analysis --run-prefix m2final --panns \
        --out-dir outputs/m2_final
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from wadidaw.data.dataset import SAMPLES_ROOT, MusdbMultimodalDataset
from wadidaw.data.features import GAIN_MAX_DB, GAIN_MIN_DB
from wadidaw.data.prompts import STEM_NAMES
from wadidaw.models.fusion import build_model
from wadidaw.training.evaluate import (
    _compute_mix_8band_and_64mel_db,
    compute_regression_and_spectral_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "outputs" / "training"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"

SPLIT_MANIFESTS = {
    "validation": "validation_samples.json",
    "test_human": "test_human_samples.json",
    "test_synthetic": "test_synthetic_samples.json",
}

METRIC_FIELDS = (
    "mae_db",
    "rmse_db",
    "bias_db",
    "band_mae_db",
    "success_rate_1db",
    "instruction_adherence",
)


@dataclass
class SplitArrays:
    split: str
    g_pred: np.ndarray
    g_true: np.ndarray
    loudness: np.ndarray
    balance: np.ndarray
    mask: np.ndarray
    band_pred: np.ndarray
    band_true: np.ndarray
    mel_pred: np.ndarray
    mel_true: np.ndarray
    ids: list[str]
    songs: list[str]
    prompts: list[str]
    templates: list[str]
    saturation: float


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _manifest_path(split: str, panns: bool) -> Path:
    name = SPLIT_MANIFESTS[split]
    if panns:
        name = name.replace("_samples.json", "_panns_samples.json")
    return SAMPLES_ROOT / name


def _load_run(run_name: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    results_path = TRAINING_DIR / f"{run_name}_results.json"
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    if not results_path.is_file():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    results = json.loads(results_path.read_text())
    config = results.get("config", {})
    model_name = str(config.get("model_name", results.get("model_name", "m2")))
    text_backend = str(config.get("text_backend", "subword"))
    model_kwargs = dict(config.get("model_kwargs") or {})

    model = build_model(model_name, text_backend=text_backend, **model_kwargs)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, results


@torch.no_grad()
def _collect_split(
    model: torch.nn.Module,
    split: str,
    manifest: Path,
    device: torch.device,
    batch_size: int,
) -> SplitArrays:
    dataset = MusdbMultimodalDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    audio_key = getattr(model, "audio_input_key", "audio")
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    ids: list[str] = []
    songs: list[str] = []
    prompts: list[str] = []
    templates: list[str] = []
    saturated = 0
    active_total = 0

    for batch in loader:
        audio = batch[audio_key].to(device=device, dtype=torch.float32)
        mask = batch["active_mask"].to(device=device)
        outputs = model(audio, batch["prompt"], active_mask=mask, clamp_output=True)
        g_pred = outputs["gains_pred"].detach().cpu().numpy()

        g_true = batch["gains_db"].numpy()
        mask_np = batch["active_mask"].numpy()
        mel_power = batch["stem_mel_power"].numpy()

        band_pred, mel_pred = _compute_mix_8band_and_64mel_db(mel_power, g_pred, mask_np)
        band_true, mel_true = _compute_mix_8band_and_64mel_db(mel_power, g_true, mask_np)

        collected["g_pred"].append(g_pred)
        collected["g_true"].append(g_true)
        collected["loudness"].append(batch["loudness_db"].numpy())
        collected["balance"].append(batch["balance_target"].numpy())
        collected["mask"].append(mask_np)
        collected["band_pred"].append(band_pred)
        collected["band_true"].append(band_true)
        collected["mel_pred"].append(mel_pred)
        collected["mel_true"].append(mel_true)

        ids.extend(str(x) for x in batch["id"])
        songs.extend(str(x) for x in batch["song"])
        prompts.extend(str(x) for x in batch["prompt"])
        templates.extend(str(x) for x in batch["prompt_template"])

        at_min = np.isclose(g_pred, GAIN_MIN_DB, atol=1e-6)
        at_max = np.isclose(g_pred, GAIN_MAX_DB, atol=1e-6)
        saturated += int(np.logical_and(at_min | at_max, mask_np).sum())
        active_total += int(mask_np.sum())

    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    return SplitArrays(
        split=split,
        g_pred=arrays["g_pred"],
        g_true=arrays["g_true"],
        loudness=arrays["loudness"],
        balance=arrays["balance"],
        mask=arrays["mask"],
        band_pred=arrays["band_pred"],
        band_true=arrays["band_true"],
        mel_pred=arrays["mel_pred"],
        mel_true=arrays["mel_true"],
        ids=ids,
        songs=songs,
        prompts=prompts,
        templates=templates,
        saturation=float(100.0 * saturated / max(active_total, 1)),
    )


def _metrics(data: SplitArrays, sel: np.ndarray | None = None) -> dict[str, object]:
    if sel is None:
        sel = np.arange(len(data.g_pred))
    if len(sel) == 0:
        return {}
    metrics = compute_regression_and_spectral_metrics(
        gains_pred=data.g_pred[sel],
        gains_true=data.g_true[sel],
        loudness_true=data.loudness[sel],
        balance_true=data.balance[sel],
        active_mask=data.mask[sel],
        band8_pred_db=data.band_pred[sel],
        band8_true_db=data.band_true[sel],
        mel64_pred_db=data.mel_pred[sel],
        mel64_true_db=data.mel_true[sel],
    )
    metrics["gain_saturation_rate"] = round(
        float(
            100.0
            * np.logical_and(
                np.isclose(data.g_pred[sel], GAIN_MIN_DB, atol=1e-6)
                | np.isclose(data.g_pred[sel], GAIN_MAX_DB, atol=1e-6),
                data.mask[sel],
            ).sum()
            / max(int(data.mask[sel].sum()), 1)
        ),
        2,
    )
    return metrics


def _group_indices(keys: Iterable[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(keys):
        groups[key].append(idx)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def _aggregate(rows: list[dict[str, object]], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)

    output: list[dict[str, object]] = []
    for keys, group in grouped.items():
        out = {field: value for field, value in zip(key_fields, keys)}
        out["n_samples"] = int(group[0].get("sample_count", 0))
        for metric in METRIC_FIELDS + ("gain_saturation_rate",):
            values = [float(item[metric]) for item in group if metric in item]
            if not values:
                continue
            out[f"{metric}_mean"] = round(mean(values), 4)
            out[f"{metric}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        output.append(out)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _md_table(rows: list[dict[str, object]], columns: list[tuple[str, str, int]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _, digits in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(_fmt(value, digits))
            else:
                cells.append(str(value) if value is not None else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_report(
    run_prefix: str,
    seeds: list[int],
    overall: list[dict[str, object]],
    per_stem: list[dict[str, object]],
    per_prompt: list[dict[str, object]],
    per_song: list[dict[str, object]],
    worst: list[dict[str, object]],
) -> str:
    lines: list[str] = [
        f"# Error Analysis — {run_prefix}",
        "",
        f"Run prefix: `{run_prefix}` · seeds: {', '.join(str(s) for s in seeds)} · "
        "metrik = mean ± std across seeds.",
        "",
        "## Ringkasan overall",
        "",
        _md_table(
            overall,
            [
                ("eval_split", "Split", 0),
                ("mae_db_mean", "MAE (dB)", 3),
                ("mae_db_std", "±", 3),
                ("rmse_db_mean", "RMSE (dB)", 3),
                ("band_mae_db_mean", "Band MAE (dB)", 3),
                ("success_rate_1db_mean", "Success <1 dB (%)", 2),
                ("instruction_adherence_mean", "Adherence (%)", 2),
                ("gain_saturation_rate_mean", "Gain saturation (%)", 2),
            ],
        ),
        "",
        "## Per stem",
        "",
    ]
    for split in ("validation", "test_human", "test_synthetic"):
        rows = [row for row in per_stem if row["eval_split"] == split]
        if not rows:
            continue
        lines += [
            f"### {split}",
            "",
            _md_table(
                rows,
                [
                    ("stem", "Stem", 0),
                    ("mae_db_mean", "MAE (dB)", 3),
                    ("mae_db_std", "±", 3),
                    ("rmse_db_mean", "RMSE (dB)", 3),
                    ("success_rate_1db_mean", "Success <1 dB (%)", 2),
                ],
            ),
            "",
        ]

    lines += ["## Per prompt family", ""]
    for split in ("test_human", "test_synthetic", "validation"):
        rows = [row for row in per_prompt if row["eval_split"] == split]
        if not rows:
            continue
        rows = sorted(rows, key=lambda item: float(item["mae_db_mean"]), reverse=True)
        worst_rows = rows[:5]
        best_rows = rows[-5:][::-1]
        lines += [
            f"### {split} — 5 keluarga prompt terburuk",
            "",
            _md_table(
                worst_rows,
                [
                    ("prompt_template", "Prompt family", 0),
                    ("n_samples", "n", 0),
                    ("mae_db_mean", "MAE (dB)", 3),
                    ("band_mae_db_mean", "Band MAE", 3),
                    ("success_rate_1db_mean", "Success (%)", 2),
                    ("instruction_adherence_mean", "Adherence (%)", 2),
                ],
            ),
            "",
            f"### {split} — 5 keluarga prompt terbaik",
            "",
            _md_table(
                best_rows,
                [
                    ("prompt_template", "Prompt family", 0),
                    ("n_samples", "n", 0),
                    ("mae_db_mean", "MAE (dB)", 3),
                    ("band_mae_db_mean", "Band MAE", 3),
                    ("success_rate_1db_mean", "Success (%)", 2),
                    ("instruction_adherence_mean", "Adherence (%)", 2),
                ],
            ),
            "",
        ]

    lines += ["## Per lagu", ""]
    for split in ("test_human", "test_synthetic", "validation"):
        rows = [row for row in per_song if row["eval_split"] == split]
        if not rows:
            continue
        rows = sorted(rows, key=lambda item: float(item["mae_db_mean"]), reverse=True)
        lines += [
            f"### {split} — 10 lagu terburuk",
            "",
            _md_table(
                rows[:10],
                [
                    ("song", "Lagu", 0),
                    ("n_samples", "n", 0),
                    ("mae_db_mean", "MAE (dB)", 3),
                    ("rmse_db_mean", "RMSE (dB)", 3),
                    ("success_rate_1db_mean", "Success (%)", 2),
                ],
            ),
            "",
        ]

    lines += ["## Sampel terburuk", ""]
    for split in ("test_human", "test_synthetic", "validation"):
        rows = [row for row in worst if row["eval_split"] == split][:10]
        if not rows:
            continue
        lines += [
            f"### {split}",
            "",
            _md_table(
                rows,
                [
                    ("song", "Lagu", 0),
                    ("prompt_template", "Prompt family", 0),
                    ("sample_mae_db", "MAE (dB)", 3),
                    ("prompt", "Prompt", 0),
                ],
            ),
            "",
        ]

    lines += [
        "## Catatan",
        "",
        "- `gain_saturation_rate` = persentase prediksi gain yang menyentuh batas "
        f"clipping output ({GAIN_MIN_DB:+.0f}/{GAIN_MAX_DB:+.0f} dB).",
        "- Clipping audio (peak/clipping mix) dihitung terpisah oleh "
        "`scripts/eval_clipping.py`.",
        "- Keluarga prompt pada `test_human` adalah 16 template manusia; pada "
        "`test_synthetic`/`validation` adalah template sintetis.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stem / per-prompt / per-song error analysis.")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--splits", nargs="+", default=["validation", "test_human", "test_synthetic"],
                        choices=sorted(SPLIT_MANIFESTS))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directory for CSV tables (default: outputs/<run-prefix>_analysis).")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "error_analysis.md")
    parser.add_argument("--panns", action="store_true",
                        help="Use *_panns_samples.json manifests.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the batch size stored in the run config.")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.out_dir or (ROOT / "outputs" / f"{args.run_prefix}_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict[str, object]] = []
    stem_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    song_rows: list[dict[str, object]] = []
    worst_rows: list[dict[str, object]] = []

    used_seeds: list[int] = []
    for seed in args.seeds:
        run_name = f"{args.run_prefix}_m2_seed{seed}"
        if not (TRAINING_DIR / f"{run_name}_results.json").is_file():
            print(f"[skip] run not found: {run_name}")
            continue
        print(f"[run] {run_name}")
        used_seeds.append(seed)
        model, results = _load_run(run_name, device)
        batch_size = int(args.batch_size or results.get("config", {}).get("batch_size", 16))

        for split in args.splits:
            manifest = _manifest_path(split, args.panns)
            if not manifest.is_file():
                print(f"[skip] manifest not found: {manifest}")
                continue
            data = _collect_split(model, split, manifest, device, batch_size)

            overall = _metrics(data)
            overall_rows.append({"eval_split": split, **overall})
            overall_rows[-1]["gain_saturation_rate"] = data.saturation

            stem_metrics = overall.get("mae_per_stem_db", {})
            rmse_stem = overall.get("rmse_per_stem_db", {})
            for stem_idx, stem in enumerate(STEM_NAMES):
                stem_mask = data.mask[:, stem_idx]
                stem_err = np.abs(
                    data.g_pred[:, stem_idx] - data.g_true[:, stem_idx]
                )[stem_mask]
                stem_rows.append(
                    {
                        "eval_split": split,
                        "stem": stem,
                        "sample_count": int(stem_mask.sum()),
                        "mae_db": stem_metrics.get(stem, 0.0),
                        "rmse_db": rmse_stem.get(stem, 0.0),
                        "success_rate_1db": round(
                            float(100.0 * np.mean(stem_err < 1.0)) if len(stem_err) else 0.0,
                            2,
                        ),
                    }
                )

            for template, sel in _group_indices(data.templates).items():
                row = _metrics(data, sel)
                if row:
                    prompt_rows.append({"eval_split": split, "prompt_template": template, **row})

            for song, sel in _group_indices(data.songs).items():
                row = _metrics(data, sel)
                if row:
                    song_rows.append({"eval_split": split, "song": song, **row})

            sample_mae = np.abs(data.g_pred - data.g_true).sum(axis=1) / np.maximum(
                data.mask.sum(axis=1), 1
            )
            for idx in np.argsort(sample_mae)[::-1][:20]:
                worst_rows.append(
                    {
                        "eval_split": split,
                        "seed": seed,
                        "id": data.ids[idx],
                        "song": data.songs[idx],
                        "prompt_template": data.templates[idx],
                        "prompt": data.prompts[idx],
                        "sample_mae_db": round(float(sample_mae[idx]), 4),
                    }
                )

    if not used_seeds:
        raise SystemExit(f"No runs found for prefix '{args.run_prefix}'.")

    overall = _aggregate(overall_rows, ("eval_split",))
    # Per-stem aggregation is keyed by split + stem; metrics come from the
    # overall per-seed row, so flatten them into explicit metric columns first.
    stem_flat = [
        {
            "eval_split": row["eval_split"],
            "stem": row["stem"],
            "sample_count": row["sample_count"],
            "mae_db": row["mae_db"],
            "rmse_db": row["rmse_db"],
            "success_rate_1db": row["success_rate_1db"],
        }
        for row in stem_rows
    ]
    per_stem = _aggregate(stem_flat, ("eval_split", "stem"))
    per_prompt = _aggregate(prompt_rows, ("eval_split", "prompt_template"))
    per_song = _aggregate(song_rows, ("eval_split", "song"))

    # Worst samples: average error across seeds for the same sample id.
    grouped_worst: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in worst_rows:
        grouped_worst[(str(row["eval_split"]), str(row["id"]))].append(row)
    worst_agg: list[dict[str, object]] = []
    for (split, sample_id), group in grouped_worst.items():
        worst_agg.append(
            {
                "eval_split": split,
                "id": sample_id,
                "song": group[0]["song"],
                "prompt_template": group[0]["prompt_template"],
                "prompt": group[0]["prompt"],
                "sample_mae_db": round(mean(float(g["sample_mae_db"]) for g in group), 4),
            }
        )
    worst_agg.sort(key=lambda row: (row["eval_split"], -float(row["sample_mae_db"])))
    worst_by_split: list[dict[str, object]] = []
    for split in sorted({row["eval_split"] for row in worst_agg}, key=lambda s: s):
        worst_by_split.extend([row for row in worst_agg if row["eval_split"] == split][:20])

    _write_csv(out_dir / "error_analysis_overall.csv", overall)
    _write_csv(out_dir / "error_analysis_per_stem.csv", per_stem)
    _write_csv(out_dir / "error_analysis_per_prompt.csv", per_prompt)
    _write_csv(out_dir / "error_analysis_per_song.csv", per_song)
    _write_csv(out_dir / "error_analysis_worst_samples.csv", worst_by_split)

    report = _build_report(
        run_prefix=args.run_prefix,
        seeds=used_seeds,
        overall=overall,
        per_stem=per_stem,
        per_prompt=per_prompt,
        per_song=per_song,
        worst=worst_by_split,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n")
    print(f"Wrote report -> {args.report}")
    print(f"Wrote CSV tables -> {out_dir}")


if __name__ == "__main__":
    main()