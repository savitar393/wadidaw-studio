
import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch

from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer
from wadidaw.rl.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "training"


def evaluate(env, agent, policy, episodes_per_song):
    """Evaluate every validation song on fixed target mixes."""
    records = []
    per_song = {}

    for position, member in enumerate(env.members):
        song_records = []

        for episode in range(episodes_per_song):
            seed = 50_000 + position * 1000 + episode

            observation, info = env.reset(
                seed=seed,
                options={"song_position": position},
            )

            initial_error = info["error"]
            episode_return = 0.0
            rng = np.random.default_rng(seed + 12345)

            for step in range(env.current_env.max_steps):
                if policy == "sac":
                    action = agent.act(
                        observation,
                        deterministic=True,
                    )

                elif policy == "no_op":
                    action = np.zeros(
                        env.action_space.shape,
                        dtype=np.float32,
                    )

                elif policy == "random":
                    action = rng.uniform(
                        -1.0,
                        1.0,
                        size=env.action_space.shape,
                    ).astype(np.float32)

                else:
                    raise ValueError(policy)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_return += reward

                if terminated or truncated:
                    break

            result = {
                "success": float(info["success"]),
                "error": float(info["error"]),
                "initial_error": float(initial_error),
                "return": float(episode_return),
                "steps": step + 1,
            }

            records.append(result)
            song_records.append(result)

        per_song[member["song"]] = {
            "success_rate": 100.0 * np.mean(
                [r["success"] for r in song_records]
            ),
            "mean_error": float(np.mean(
                [r["error"] for r in song_records]
            )),
        }

    return {
        "success_rate": float(
            100.0 * np.mean([r["success"] for r in records])
        ),
        "mean_error": float(
            np.mean([r["error"] for r in records])
        ),
        "mean_initial_error": float(
            np.mean([r["initial_error"] for r in records])
        ),
        "mean_return": float(
            np.mean([r["return"] for r in records])
        ),
        "mean_steps": float(
            np.mean([r["steps"] for r in records])
        ),
        "per_song": per_song,
    }


def train(args):
    if (
        args.steps < 1
        or args.warmup < 0
        or args.eval_every < 1
        or args.episodes_per_song < 1
        or args.batch_size < 1
    ):
        raise ValueError("Invalid training configuration.")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_env = MultiSongMixingEnv(split="train")
    val_env = MultiSongMixingEnv(split="validation")

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
        seed=42,
    )

    checkpoint_path = (
        CHECKPOINT_DIR / f"{args.run_name}_best.pt"
    )

    metrics_path = (
        METRICS_DIR / f"{args.run_name}_metrics.csv"
    )

    print("WadiDAW — Multi-song SAC training")
    print("Observation dimensions:", obs_dim)
    print("Action dimensions:", action_dim)
    print("Training songs:", len(train_env.members))
    print("Validation songs:", len(val_env.members))
    print("Steps:", args.steps)
    print("Alpha:", args.alpha)

    # Establish fair validation baselines.
    print("\nEvaluating validation baselines...")

    for policy in ("no_op", "random"):
        result = evaluate(
            val_env,
            agent,
            policy,
            args.episodes_per_song,
        )

        print(
            f"{policy.upper():6s} | "
            f"Success: {result['success_rate']:5.1f}% | "
            f"Error: {result['mean_error']:.6f}"
        )

    observation, info = train_env.reset(seed=42)
    train_env.action_space.seed(42)

    episode_count = 0
    best_error = float("inf")
    history = []
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
                seed=42 + episode_count
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
                if recent_updates else float("nan")
            )

            critic_loss = (
                float(np.mean([
                    m["critic_loss"] for m in recent_updates
                ]))
                if recent_updates else float("nan")
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

            print(
                f"Step {global_step:6d} | "
                f"Success: {result['success_rate']:5.1f}% | "
                f"Error: {result['mean_error']:.6f} | "
                f"Return: {result['mean_return']:.4f}"
            )

            # Select by validation error, not training error.
            if result["mean_error"] < best_error:
                best_error = result["mean_error"]

                torch.save(
                    {
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
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    # Load the checkpoint selected by validation error.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )

    final_validation = evaluate(
        val_env,
        agent,
        "sac",
        args.episodes_per_song,
    )

    print("\nBEST CHECKPOINT")
    print("Step:", checkpoint["step"])
    print(
        "Validation success:",
        f"{final_validation['success_rate']:.1f}%",
    )
    print(
        "Validation error:",
        f"{final_validation['mean_error']:.6f}",
    )

    print("\nPER-SONG VALIDATION")
    for song, metrics in final_validation["per_song"].items():
        print(
            f"{song}: "
            f"error={metrics['mean_error']:.6f}, "
            f"success={metrics['success_rate']:.1f}%"
        )

    print("\nCheckpoint:", checkpoint_path)
    print("Training metrics:", metrics_path)

    train_env.close()
    val_env.close()


if __name__ == "__main__":
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
        "--episodes-per-song", type=int, default=10
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1
    )
    parser.add_argument(
        "--run-name", default="music_sac_v1"
    )

    train(parser.parse_args())
