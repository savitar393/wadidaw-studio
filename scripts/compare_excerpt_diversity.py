"""Launch the controlled SAC comparison explicitly, or summarize completed runs."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import train_music_sac as trainer


SEEDS = (42, 123, 2026)
CONDITIONS = {"midpoint": (0.5,), "three_excerpts": (0.2, 0.5, 0.8)}
CONTROLS = {
    "steps": 20_000, "warmup": 1_000, "alpha": 0.1, "batch_size": 64,
    "eval_every": 2_000, "episodes_per_song": 10,
}
METRICS = ("mean_error", "success_rate", "mean_return", "mean_steps")


def experiment_commands(prefix):
    commands = []
    for condition, fractions in CONDITIONS.items():
        for seed in SEEDS:
            run_name = f"{prefix}_{condition}_seed{seed}"
            trainer.output_paths(run_name)  # Validate names before any filesystem writes.
            command = [sys.executable, "-m", "scripts.train_music_sac"]
            for name, value in CONTROLS.items():
                command.extend(("--" + name.replace("_", "-"), str(value)))
            command.extend(("--seed", str(seed), "--run-name", run_name))
            command.extend(("--excerpt-fractions", *map(str, fractions)))
            commands.append((condition, seed, command))
    return commands


def load_runs(prefix):
    """Reject partial runs, unequal budgets, changed sources, or different tasks."""
    runs = []
    reference = None
    expected_steps = list(range(2000, 20001, 2000))
    task_fields = ("song", "song_position", "excerpt_position", "excerpt_fraction", "target_id", "seed")
    for condition, seed, command in experiment_commands(prefix):
        expected_config = json.loads(json.dumps(vars(trainer.parse_args(command[3:]))))
        path = trainer.output_paths(expected_config["run_name"])[2]
        result = json.loads(path.read_text())
        if result.get("status") != "complete" or result.get("config") != expected_config:
            raise ValueError(f"Incomplete or incompatible run: {path}")
        common = {key: result[key] for key in (
            "manifest_sha256", "source_sha256", "versions", "training_songs", "validation_protocol",
        )}
        if reference is None:
            reference = common
        elif common != reference:
            raise ValueError(f"Data, code, versions or validation protocol changed: {path}")
        protocol = result["validation_protocol"]
        tasks = protocol["tasks"]
        if (protocol["split"] != "validation"
                or protocol["excerpt_fractions"] != [0.2, 0.5, 0.8]
                or protocol["targets_per_song_excerpt"] != 10 or len(tasks) != 120):
            raise ValueError(f"Expected the fixed 120-task validation protocol: {path}")
        identities = [(t["song_position"], t["excerpt_position"], t["target_id"]) for t in tasks]
        expected_identities = [(s, e, t) for s in range(4) for e in range(3) for t in range(10)]
        if identities != expected_identities:
            raise ValueError(f"Missing, duplicated or reordered validation tasks: {path}")
        for task in tasks:
            s, e, t = task["song_position"], task["excerpt_position"], task["target_id"]
            if task["seed"] != 50000 + s * 1000 + e * 100 + t or task["excerpt_fraction"] != (0.2, 0.5, 0.8)[e]:
                raise ValueError(f"Incorrect validation target seeds or excerpts: {path}")
        validation_songs = {task["song"] for task in tasks}
        if len(validation_songs) != 4 or validation_songs & {m["song"] for m in result["training_songs"]}:
            raise ValueError(f"Invalid song isolation: {path}")
        history = result["history"]
        if [row["step"] for row in history] != expected_steps:
            raise ValueError(f"Incomplete learning curve: {path}")
        if not np.isfinite([[row[key] for key in METRICS] for row in history]).all():
            raise ValueError(f"Nonfinite learning curve: {path}")
        best = min(history, key=lambda row: row["mean_error"])
        if best["step"] != result["best_step"]:
            raise ValueError(f"Best checkpoint does not match validation error: {path}")
        for selection, row in (("best", best), ("final", history[-1])):
            validation = result[f"{selection}_validation"]
            actual_tasks = [{key: episode[key] for key in task_fields} for episode in validation["episodes"]]
            if validation["episode_count"] != 120 or actual_tasks != tasks:
                raise ValueError(f"Checkpoint evaluated on different validation tasks: {path}")
            measured = trainer.summarize(validation["episodes"])
            if any(not np.isclose(validation[key], measured[key]) or not np.isclose(row[key], measured[key])
                   for key in METRICS):
                raise ValueError(f"Checkpoint metrics disagree with episode records: {path}")
        runs.append({"condition": condition, "seed": seed, "result": result})
    return runs


def performance_groups(validation):
    yield "overall", "all", validation
    for song, metrics in validation["per_song"].items():
        yield "song", song, metrics
    for fraction, metrics in validation["per_excerpt"].items():
        yield "excerpt", fraction, metrics
    for song, excerpts in validation["per_song_excerpt"].items():
        for fraction, metrics in excerpts.items():
            yield "song_excerpt", f"{song}/{fraction}", metrics


def plot_learning_curves(runs, output, prefix):
    # This is the only optional dependency; training itself needs no plotting package.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    labels = {
        "mean_error": "Spectral error (normalized MSE; lower is better)",
        "success_rate": "Success rate (%)",
        "mean_return": "Mean episode return",
        "mean_steps": "Mean episode length (actions)",
    }
    colors = {"midpoint": "#2563eb", "three_excerpts": "#b45309"}
    names = {"midpoint": "A: midpoint (0.5)", "three_excerpts": "B: three excerpts (0.2, 0.5, 0.8)"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    try:
        for axis, metric in zip(axes.flat, METRICS):
            for condition in CONDITIONS:
                histories = [r["result"]["history"] for r in runs if r["condition"] == condition]
                steps = [row["step"] for row in histories[0]]
                values = np.array([[row[metric] for row in history] for history in histories])
                mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
                style = "-" if condition == "midpoint" else "--"
                for series in values:
                    axis.plot(steps, series, color=colors[condition], linestyle=style, alpha=0.23, linewidth=0.8)
                axis.fill_between(steps, mean - std, mean + std, color=colors[condition], alpha=0.12)
                axis.plot(steps, mean, color=colors[condition], linestyle=style,
                          marker="o" if condition == "midpoint" else "s", markersize=3,
                          linewidth=2, label=names[condition])
            axis.set_title(labels[metric], fontsize=10)
            axis.set_xlabel("Training environment transitions")
            axis.set_xlim(0, CONTROLS["steps"])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
            if metric in ("mean_error", "success_rate", "mean_steps"):
                axis.set_ylim(bottom=0)
            if metric == "success_rate":
                axis.set_ylim(0, 100)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        figure.suptitle(f"{prefix}: SAC excerpt diversity", fontsize=14)
        figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=2, frameon=False)
        figure.text(0.5, 0.035,
                    "Each checkpoint: 4 validation songs × 3 excerpts × 10 fixed targets.\n"
                    "Thin lines: individual seeds. Thick lines: mean of 3 seeds. Shading: ± sample SD, not a confidence interval. No smoothing.",
                    ha="center", fontsize=8)
        figure.subplots_adjust(top=0.85, bottom=0.15, hspace=0.42, wspace=0.25)
        figure.savefig(output / "learning_curves.svg")
        figure.savefig(output / "learning_curves.png", dpi=160)
    finally:
        plt.close(figure)


def write_csv(path, rows):
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(prefix, output):
    runs = load_runs(prefix)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output}")
    require_plotting()
    per_seed = []
    grouped = defaultdict(list)
    for run in runs:
        result = run["result"]
        for selection in ("best", "final"):
            for group, member, metrics in performance_groups(result[f"{selection}_validation"]):
                row = {
                    "condition": run["condition"], "seed": run["seed"], "selection": selection,
                    "best_step": result["best_step"],
                    "checkpoint_step": result["best_step"] if selection == "best" else CONTROLS["steps"],
                    "group": group, "member": member,
                    "episode_count": metrics["episode_count"],
                    **{metric: metrics[metric] for metric in METRICS},
                }
                if not np.isfinite([row[metric] for metric in METRICS]).all():
                    raise ValueError("Cannot report nonfinite validation metrics.")
                per_seed.append(row)
                grouped[(run["condition"], selection, group, member)].append(row)
    aggregate = []
    for (condition, selection, group, member), rows in grouped.items():
        if len(rows) != 3 or {row["seed"] for row in rows} != set(SEEDS):
            raise ValueError("Every reported group must include all three independent seeds.")
        for metric in METRICS:
            values = [row[metric] for row in rows]
            aggregate.append({
                "condition": condition, "selection": selection, "group": group,
                "member": member, "metric": metric, "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)), "n_seeds": len(values),
            })
    paired = []
    for selection in ("best", "final"):
        for metric in METRICS:
            differences = [
                next(r["result"][f"{selection}_validation"][metric] for r in runs if r["condition"] == "three_excerpts" and r["seed"] == seed)
                - next(r["result"][f"{selection}_validation"][metric] for r in runs if r["condition"] == "midpoint" and r["seed"] == seed)
                for seed in SEEDS
            ]
            paired.append({"selection": selection, "metric": metric, "comparison": "B minus A",
                           "mean": float(np.mean(differences)), "std": float(np.std(differences, ddof=1)), "n_seeds": 3})
    report = {
        "prefix": prefix, "controls": CONTROLS, "seeds": SEEDS, "conditions": CONDITIONS,
        "validation_protocol": runs[0]["result"]["validation_protocol"],
        "provenance": {key: runs[0]["result"][key] for key in ("manifest_sha256", "source_sha256", "versions")},
        "learning_curves": [{"condition": run["condition"], "seed": run["seed"], "history": run["result"]["history"]} for run in runs],
        "uncertainty": "Sample standard deviation across training seeds (ddof=1); not across episodes or a confidence interval.",
        "selection": "Best checkpoint minimizes validation mean_error; final is the 20,000-transition checkpoint. Validation is also used for selection.",
        "per_seed": per_seed, "aggregate": aggregate, "paired_differences": paired,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "aggregate.csv", aggregate)
    with (output / "comparison.json").open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    plot_learning_curves(runs, output, prefix)
    for row in aggregate:
        if row["group"] == "overall":
            print(f"{row['condition']:15s} {row['selection']:5s} {row['metric']:14s}: {row['mean']:.6f} ± {row['std']:.6f}")
    print("Comparison outputs:", output)
    return report


def require_plotting():
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Learning curves require Matplotlib: python -m pip install matplotlib") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="excerpt_diversity_v1")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly launch all six full training runs, then report.")
    mode.add_argument("--report-only", action="store_true", help="Report six completed runs without training or model evaluation.")
    args = parser.parse_args(argv)
    commands = experiment_commands(args.prefix)
    output = args.output_dir or trainer.ROOT / "outputs" / "experiments" / args.prefix
    if not args.report_only:
        for _, _, command in commands:
            print(shlex.join(command))
    if not args.run and not args.report_only:
        print("Preview only. Add --run to launch training, or --report-only to summarize completed runs.")
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {output}")
    if args.run:
        # Check all six names before launching the first expensive job.
        for _, _, command in commands:
            run_name = trainer.parse_args(command[3:]).run_name
            for path in trainer.output_paths(run_name):
                if path.exists():
                    raise FileExistsError(f"Refusing to overwrite {path}; choose a new --prefix.")
        require_plotting()
        for _, _, command in commands:
            subprocess.run(command, cwd=trainer.ROOT, check=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    write_report(args.prefix, output)


if __name__ == "__main__":
    main()
