
import argparse
import csv
import hashlib
import json
import platform
import random
import re
from collections import defaultdict, deque
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from wadidaw import multi_song_env
from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer
from wadidaw.rl.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "training"
VALIDATION_FRACTIONS = (0.2, 0.5, 0.8)
VALIDATION_TARGETS = 10


def evaluation_tasks(env, episodes_per_song):
    """A fixed task grid, independent of training RNGs and excerpt settings."""
    return [
        {
            "song": member["song"],
            "song_position": position,
            "excerpt_position": excerpt_position,
            "excerpt_fraction": fraction,
            "target_id": episode,
            "seed": 50_000 + position * 1000 + excerpt_position * 100 + episode,
        }
        for position, member in enumerate(env.members)
        for excerpt_position, fraction in enumerate(env.excerpt_fractions)
        for episode in range(episodes_per_song)
    ]


def summarize(records):
    return {
        "episode_count": len(records),
        "success_rate": float(100 * np.mean([r["success"] for r in records])),
        "mean_error": float(np.mean([r["error"] for r in records])),
        "mean_initial_error": float(np.mean([r["initial_error"] for r in records])),
        "mean_return": float(np.mean([r["return"] for r in records])),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
    }


def evaluate(env, agent, policy, episodes_per_song):
    """Evaluate all configured validation excerpts; legacy midpoint envs work."""
    if env.split != "validation":
        raise ValueError("Evaluation is restricted to the validation partition.")
    if episodes_per_song < 1 or episodes_per_song > 100:
        raise ValueError("Expected 1 to 100 targets per song/excerpt pair.")
    if policy not in ("sac", "no_op", "random"):
        raise ValueError(policy)
    records = []
    by_song, by_excerpt, by_pair = defaultdict(list), defaultdict(list), defaultdict(list)
    for task in evaluation_tasks(env, episodes_per_song):
        observation, info = env.reset(
            seed=task["seed"],
            options={
                "song_position": task["song_position"],
                "excerpt_position": task["excerpt_position"],
            },
        )
        initial_error = info["error"]
        episode_return = 0.0
        rng = np.random.default_rng(task["seed"] + 12345)
        for step in range(env.current_env.max_steps):
            if policy == "sac":
                action = agent.act(observation, deterministic=True)
            elif policy == "no_op":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        result = {
            **task,
            "excerpt_start_seconds": info["excerpt_start_seconds"],
            "success": float(info["success"]),
            "error": float(info["error"]),
            "initial_error": float(initial_error),
            "return": float(episode_return),
            "steps": step + 1,
        }
        records.append(result)
        fraction = str(task["excerpt_fraction"])
        by_song[task["song"]].append(result)
        by_excerpt[fraction].append(result)
        by_pair[(task["song"], fraction)].append(result)
    return {
        **summarize(records),
        "per_song": {key: summarize(rows) for key, rows in by_song.items()},
        "per_excerpt": {key: summarize(rows) for key, rows in by_excerpt.items()},
        "per_song_excerpt": {
            song: {fraction: summarize(by_pair[(song, fraction)]) for fraction in by_excerpt}
            for song in by_song
        },
        "episodes": records,
    }


def output_paths(run_name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Run name must be a simple filename using letters, digits, _, - or .")
    return (
        CHECKPOINT_DIR / f"{run_name}_best.pt",
        METRICS_DIR / f"{run_name}_metrics.csv",
        METRICS_DIR / f"{run_name}_results.json",
    )


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def experiment_metadata(args, train_env, val_env):
    source_files = (
        "scripts/train_music_sac.py", "wadidaw/music_env.py",
        "wadidaw/multi_song_env.py", "wadidaw/rl/sac.py",
        "wadidaw/rl/networks.py", "wadidaw/rl/replay_buffer.py",
    )
    return {
        "config": vars(args),
        "training_songs": train_env.members,
        "manifest_sha256": hashlib.sha256(multi_song_env.MANIFEST.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "numpy", "gymnasium", "soundfile")},
        },
        "validation_protocol": {
            "split": "validation",
            "excerpt_fractions": list(VALIDATION_FRACTIONS),
            "targets_per_song_excerpt": VALIDATION_TARGETS,
            "tasks": evaluation_tasks(val_env, VALIDATION_TARGETS),
            "error_definition": "mean squared band dB difference divided by 12**2",
        },
    }


def train(args):
    if (
        args.steps < 1
        or args.warmup < 0
        or args.eval_every < 1
        or args.episodes_per_song != VALIDATION_TARGETS
        or args.batch_size < 1
        or not 0 <= args.seed < 2**32
        or not np.isfinite(args.alpha)
        or args.alpha <= 0
    ):
        raise ValueError("Invalid training configuration; validation requires ten targets per song/excerpt.")

    paths = output_paths(args.run_name)
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new --run-name.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents simultaneous jobs from sharing a run name.
    with paths[2].open("x") as file:
        json.dump({"status": "running", "config": vars(args)}, file, indent=2)
    seed_training(args.seed)
    with ExitStack() as resources:
        return _train(args, paths, resources)


def _train(args, paths, resources):
    checkpoint_path, metrics_path, results_path = paths
    train_env = MultiSongMixingEnv(split="train", excerpt_fractions=args.excerpt_fractions)
    resources.callback(train_env.close)
    val_env = MultiSongMixingEnv(split="validation", excerpt_fractions=VALIDATION_FRACTIONS)
    resources.callback(val_env.close)
    if len(val_env.members) != 4:
        raise ValueError("The validation protocol requires exactly four validation songs.")
    metadata = experiment_metadata(args, train_env, val_env)

    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        alpha=args.alpha,
    )

    buffer = ReplayBuffer(
        capacity=50_000,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=args.seed,
    )

    print("WadiDAW — Multi-song SAC training")
    print("Observation dimensions:", obs_dim)
    print("Action dimensions:", action_dim)
    print("Training songs:", len(train_env.members))
    print("Validation songs:", len(val_env.members))
    print("Steps:", args.steps)
    print("Alpha:", args.alpha)
    print("Training seed:", args.seed)
    print("Training excerpt fractions:", tuple(args.excerpt_fractions))
    print("Validation tasks:", len(metadata["validation_protocol"]["tasks"]))

    # Establish fair validation baselines.
    print("\nEvaluating validation baselines...")

    baselines = {}
    for policy in ("no_op", "random"):
        result = evaluate(
            val_env,
            agent,
            policy,
            args.episodes_per_song,
        )
        baselines[policy] = {key: value for key, value in result.items() if key != "episodes"}

        print(
            f"{policy.upper():6s} | "
            f"Success: {result['success_rate']:5.1f}% | "
            f"Error: {result['mean_error']:.6f}"
        )

    observation, info = train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)

    episode_count = 0
    best_error = float("inf")
    history = []
    validation_history = []
    recent_updates = deque(maxlen=500)

    print("\nStarting training...")

    for global_step in range(1, args.steps + 1):

        # Step 1: Collect an experience.
        if global_step <= args.warmup:
            action = train_env.action_space.sample()
        else:
            action = agent.act(
                observation,
                deterministic=False,
            )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = train_env.step(action)

        buffer.add(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )

        # Step 2: Reset when an episode ends.
        if terminated or truncated:
            episode_count += 1

            observation, info = train_env.reset(
                seed=args.seed + episode_count
            )
        else:
            observation = next_observation

        # Step 3: Train SAC using a replay minibatch.
        if (
            global_step > args.warmup
            and len(buffer) >= args.batch_size
        ):
            batch = buffer.sample(args.batch_size)
            metrics = agent.update(batch)
            recent_updates.append(metrics)

        # Step 4: Evaluate on unseen validation songs.
        if (
            global_step % args.eval_every == 0
            or global_step == args.steps
        ):
            result = evaluate(
                val_env,
                agent,
                "sac",
                args.episodes_per_song,
            )

            actor_loss = (
                float(np.mean([
                    m["actor_loss"] for m in recent_updates
                ]))
                if recent_updates else None
            )

            critic_loss = (
                float(np.mean([
                    m["critic_loss"] for m in recent_updates
                ]))
                if recent_updates else None
            )

            record = {
                "step": global_step,
                "success_rate": result["success_rate"],
                "mean_error": result["mean_error"],
                "mean_return": result["mean_return"],
                "mean_steps": result["mean_steps"],
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            }

            history.append(record)
            validation_history.append({
                "step": global_step,
                **{key: value for key, value in result.items() if key != "episodes"},
            })

            print(
                f"Step {global_step:6d} | "
                f"Success: {result['success_rate']:5.1f}% | "
                f"Error: {result['mean_error']:.6f} | "
                f"Return: {result['mean_return']:.4f}"
            )

            # Select by validation error, not training error.
            if result["mean_error"] < best_error:
                best_error = result["mean_error"]
                best_step = global_step
                best_validation = result

                torch.save(
                    {
                        **metadata,
                        "step": global_step,
                        "obs_dim": obs_dim,
                        "action_dim": action_dim,
                        "alpha": args.alpha,
                        "validation_error": best_error,
                        "actor_state_dict":
                            agent.actor.state_dict(),
                        "q1_state_dict":
                            agent.q1.state_dict(),
                        "q2_state_dict":
                            agent.q2.state_dict(),
                        "target_q1_state_dict":
                            agent.target_q1.state_dict(),
                        "target_q2_state_dict":
                            agent.target_q2.state_dict(),
                    },
                    checkpoint_path,
                )

                print("  -> New best checkpoint saved")

    # Save learning curves as CSV.
    with metrics_path.open("x", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # Reuse the measurements made when saving the best actor. This avoids
    # evaluating the same 120 deterministic tasks a second time.
    results = {
        "status": "complete",
        **metadata,
        "best_step": best_step,
        "best_validation": best_validation,
        "final_validation": result,
        "baselines": baselines,
        "history": history,
        "validation_history": validation_history,
    }
    with results_path.open("w") as file:
        json.dump(results, file, indent=2, allow_nan=False)

    print("\nBEST CHECKPOINT")
    print("Step:", best_step)
    print(
        "Validation success:",
        f"{best_validation['success_rate']:.1f}%",
    )
    print(
        "Validation error:",
        f"{best_validation['mean_error']:.6f}",
    )

    print("\nPER-SONG VALIDATION")
    for song, metrics in best_validation["per_song"].items():
        print(
            f"{song}: "
            f"error={metrics['mean_error']:.6f}, "
            f"success={metrics['success_rate']:.1f}%"
        )

    print("\nCheckpoint:", checkpoint_path)
    print("Training metrics:", metrics_path)
    print("Detailed validation results:", results_path)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps", type=int, default=20_000
    )
    parser.add_argument(
        "--warmup", type=int, default=1_000
    )
    parser.add_argument(
        "--batch-size", type=int, default=64
    )
    parser.add_argument(
        "--eval-every", type=int, default=2_000
    )
    parser.add_argument(
        "--episodes-per-song", type=int, default=10,
        help="Targets per validation song/excerpt pair; fixed at 10 for controlled runs.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1
    )
    parser.add_argument(
        "--run-name", default="music_sac_v1"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--excerpt-fractions", type=float, nargs="+", default=(0.5,),
        help="Training excerpt start fractions; validation always uses 0.2, 0.5, 0.8.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
