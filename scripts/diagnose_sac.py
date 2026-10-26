import argparse
from pathlib import Path

import numpy as np
import torch

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]


def feedback_action(observation):
    """Compute gain corrections using only observable features."""

    current = np.maximum(observation[4:8], 1e-6)
    target = np.maximum(observation[8:12], 1e-6)

    # Current and target share the same normalization.
    correction_db = 20.0 * np.log10(target / current)

    # The environment applies 2 dB per action unit.
    return np.clip(
        correction_db / 2.0, -1.0, 1.0
    ).astype(np.float32)


def evaluate(name, agent, seed_base, episodes):
    env = MixingEnv()

    errors = []
    successes = []
    lengths = []

    for episode in range(episodes):
        observation, info = env.reset(
            seed=seed_base + episode
        )

        # Show one complete trajectory for diagnosis.
        trace = episode == 0 and name != "no_op"

        if trace:
            print(f"\n{name.upper()} — Example episode")
            print("Target gains:", env.target_gains.round(2))
            print("Initial error:", round(info["error"], 6))

        for step in range(env.max_steps):
            if name == "no_op":
                action = np.zeros(4, dtype=np.float32)

            elif name == "feedback":
                action = feedback_action(observation)

            elif name == "sac":
                action = agent.act(
                    observation, deterministic=True
                )

            else:
                raise ValueError(name)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            if trace:
                print(
                    f"Step {step + 1:2d} | "
                    f"error={info['error']:.6f} | "
                    f"action={np.round(action, 2)} | "
                    f"gains={np.round(info['gains_db'], 2)}"
                )

            if terminated or truncated:
                break

        errors.append(info["error"])
        successes.append(float(terminated))
        lengths.append(step + 1)

    env.close()

    return {
        "error": float(np.mean(errors)),
        "success": float(np.mean(successes) * 100),
        "steps": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT / "outputs/checkpoints/best_sac_actor.pt"
        ),
    )

    parser.add_argument(
        "--seed-base", type=int, default=20_000
    )

    parser.add_argument(
        "--episodes", type=int, default=50
    )

    args = parser.parse_args()

    torch.set_num_threads(1)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    agent = SACAgent()
    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )
    agent.actor.eval()

    print("Loaded checkpoint:", Path(args.checkpoint).name)
    print("Training step:", checkpoint["step"])
    print("Test seed base:", args.seed_base)

    for name in ["no_op", "feedback", "sac"]:
        result = evaluate(
            name, agent, args.seed_base, args.episodes
        )

        print(
            f"\n{name.upper()} SUMMARY\n"
            f"Success rate: {result['success']:.1f}%\n"
            f"Mean final error: {result['error']:.6f}\n"
            f"Mean steps: {result['steps']:.2f}"
        )


if __name__ == "__main__":
    main()
