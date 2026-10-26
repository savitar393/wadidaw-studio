import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer
from wadidaw.rl.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "training"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints"


def evaluate(agent, num_episodes=50):
    """Evaluate on fixed, unseen target mixes."""
    env = MixingEnv()

    successes = []
    final_errors = []
    initial_errors = []
    returns = []
    lengths = []

    for episode in range(num_episodes):
        # Separate from training seeds.
        obs, info = env.reset(seed=10_000 + episode)
        initial_errors.append(info["error"])

        total_reward = 0.0

        for step in range(env.max_steps):
            # Use the learned policy without sampling noise.
            action = agent.act(obs, deterministic=True)

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                break

        successes.append(float(terminated))
        final_errors.append(info["error"])
        returns.append(total_reward)
        lengths.append(step + 1)

    env.close()

    return {
        "success_rate": float(np.mean(successes) * 100),
        "mean_error": float(np.mean(final_errors)),
        "mean_initial_error": float(np.mean(initial_errors)),
        "mean_reward": float(np.mean(returns)),
        "mean_steps": float(np.mean(lengths)),
    }


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    # Small MLPs are often faster with fewer CPU threads.
    torch.set_num_threads(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    env = MixingEnv()
    env.action_space.seed(42)

    agent = SACAgent(alpha=args.alpha)

    buffer = ReplayBuffer(
        capacity=50_000,
        obs_dim=12,
        action_dim=4,
        seed=42,
    )

    observation, _ = env.reset(seed=42)

    completed_episodes = 0
    best_error = float("inf")
    latest_losses = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
    }

    history = []

    print("Starting SAC training...")
    print(f"Training steps: {args.steps}")
    print(f"Warmup steps: {args.warmup}")
    print(f"Batch size: {args.batch_size}")

    for global_step in range(1, args.steps + 1):

        # 1. Choose an action.
        if global_step <= args.warmup:
            action = env.action_space.sample()
        else:
            action = agent.act(
                observation,
                deterministic=False,
            )

        # 2. Execute action in the mixing environment.
        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # 3. Store the transition.
        buffer.add(
            observation,
            action,
            reward,
            next_observation,
            terminated,
            truncated,
        )

        # 4. Continue or reset the episode.
        if terminated or truncated:
            completed_episodes += 1

            observation, _ = env.reset(
                seed=42 + completed_episodes
            )
        else:
            observation = next_observation

        # 5. Perform one SAC gradient update.
        if (
            global_step >= args.warmup
            and len(buffer) >= args.batch_size
        ):
            batch = buffer.sample(args.batch_size)
            latest_losses = agent.update(batch)

        # 6. Periodically evaluate the trained policy.
        if (
            global_step % args.eval_every == 0
            or global_step == args.steps
        ):
            results = evaluate(agent)

            record = {
                "step": global_step,
                "success_rate": results["success_rate"],
                "mean_error": results["mean_error"],
                "mean_initial_error": results[
                    "mean_initial_error"
                ],
                "mean_reward": results["mean_reward"],
                "mean_steps": results["mean_steps"],
                "actor_loss": latest_losses["actor_loss"],
                "critic_loss": latest_losses["critic_loss"],
            }

            history.append(record)

            print(
                f"Step {global_step:6d} | "
                f"Success: {results['success_rate']:5.1f}% | "
                f"Error: {results['mean_error']:.5f} | "
                f"Reward: {results['mean_reward']:.4f}"
            )

            # Save the best policy measured by final error.
            if results["mean_error"] < best_error:
                best_error = results["mean_error"]

                torch.save(
                    {
                        "step": global_step,
                        "actor_state_dict": agent.actor.state_dict(),
                        "q1_state_dict": agent.q1.state_dict(),
                        "q2_state_dict": agent.q2.state_dict(),
                        "alpha": agent.alpha,
                        "mean_error": best_error,
                    },
                    CHECKPOINT_DIR / "best_sac_actor.pt",
                )

                print("  -> Best actor checkpoint saved")

    # Save training metrics for later plotting.
    history_path = OUTPUT_DIR / "sac_metrics.csv"

    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)

    env.close()

    print("\nTraining finished!")
    print("Completed episodes:", completed_episodes)
    print("Best evaluation error:", round(best_error, 6))
    print("Metrics saved to:", history_path)
    print(
        "Best model:",
        CHECKPOINT_DIR / "best_sac_actor.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
    )
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

    args = parser.parse_args()

    train(args)
