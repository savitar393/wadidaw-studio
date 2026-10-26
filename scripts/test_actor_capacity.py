from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.networks import GaussianActor


ROOT = Path(__file__).resolve().parents[1]


def feedback_action(obs):
    """Teacher uses observations, not hidden target gains."""
    current = np.maximum(obs[4:8], 1e-6)
    target = np.maximum(obs[8:12], 1e-6)

    correction_db = 20.0 * np.log10(target / current)

    return np.clip(
        correction_db / 2.0, -1.0, 1.0
    ).astype(np.float32)


def make_dataset(count, seed_base):
    env = MixingEnv()
    rng = np.random.default_rng(seed_base)

    observations = []
    labels = []

    for i in range(count):
        env.reset(seed=seed_base + i)

        # Create a variety of valid mixing states.
        if i % 3 == 0:
            gains = np.zeros(4, dtype=np.float32)

        elif i % 3 == 1:
            gains = rng.uniform(
                -12, 10, size=4
            ).astype(np.float32)

        else:
            gains = np.clip(
                env.target_gains
                + rng.normal(0, 2, size=4),
                -18,
                12,
            ).astype(np.float32)

        env.gains = gains

        features = env._features_from_gains(gains)
        obs = env._observation(features)

        observations.append(obs)
        labels.append(feedback_action(obs))

    env.close()

    return (
        torch.from_numpy(np.stack(observations)),
        torch.from_numpy(np.stack(labels)),
    )


@torch.no_grad()
def evaluate(actor):
    env = MixingEnv()
    errors = []
    successes = []
    lengths = []

    actor.eval()

    for episode in range(50):
        obs, _ = env.reset(seed=20_000 + episode)

        for step in range(env.max_steps):
            x = torch.from_numpy(obs).unsqueeze(0)

            mean, _ = actor(x)
            action = torch.tanh(mean)[0].cpu().numpy()

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated or truncated:
                break

        errors.append(info["error"])
        successes.append(float(terminated))
        lengths.append(step + 1)

    env.close()

    return (
        np.mean(errors),
        100 * np.mean(successes),
        np.mean(lengths),
    )


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    print("Generating training observations...")
    x_train, y_train = make_dataset(8000, 0)
    x_val, y_val = make_dataset(1500, 10_000)

    actor = GaussianActor()
    optimizer = torch.optim.Adam(
        actor.parameters(), lr=1e-3
    )

    best_val_loss = float("inf")
    best_weights = None

    print("Training actor with supervised feedback...")

    for epoch in range(1, 81):
        actor.train()

        permutation = torch.randperm(len(x_train))

        for indices in permutation.split(128):
            x = x_train[indices]
            y = y_train[indices]

            mean, _ = actor(x)
            predictions = torch.tanh(mean)

            loss = F.mse_loss(predictions, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        actor.eval()

        with torch.no_grad():
            val_mean, _ = actor(x_val)

            val_loss = F.mse_loss(
                torch.tanh(val_mean), y_val
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(actor.state_dict())

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:2d} | "
                f"Validation action MSE: {val_loss:.6f}"
            )

    actor.load_state_dict(best_weights)

    error, success, steps = evaluate(actor)

    print("\nSUPERVISED ACTOR — HELD-OUT TEST")
    print(f"Mean final error: {error:.6f}")
    print(f"Success rate: {success:.1f}%")
    print(f"Mean steps: {steps:.2f}")

    checkpoint_dir = ROOT / "outputs/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"actor_state_dict": actor.state_dict()},
        checkpoint_dir / "supervised_feedback_actor.pt",
    )

    print("Supervised checkpoint saved.")


if __name__ == "__main__":
    main()
