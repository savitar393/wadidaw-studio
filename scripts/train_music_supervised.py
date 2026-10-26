
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.rl.networks import GaussianActor
from scripts.train_music_sac import evaluate


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    ROOT / "outputs/checkpoints/music_supervised_cross_v1.pt"
)


def make_dataset(split, samples_per_song, seed_base):
    env = MultiSongMixingEnv(split=split)
    rng = np.random.default_rng(seed_base)

    observations = []
    labels = []

    for position, member in enumerate(env.members):
        for i in range(samples_per_song):

            seed = seed_base + position * 10000 + i

            env.reset(
                seed=seed,
                options={"song_position": position},
            )

            music = env.current_env

            # Generate different kinds of mixing states.
            if i % 3 == 0:
                gains = np.zeros(4, dtype=np.float32)

            elif i % 3 == 1:
                gains = rng.uniform(
                    -12.0, 10.0, size=4
                ).astype(np.float32)

            else:
                gains = np.clip(
                    music.target_gains
                    + rng.normal(0, 3, size=4),
                    -18.0,
                    12.0,
                ).astype(np.float32)

            music.gains = gains

            # Include different remaining action budgets.
            music.steps = int(rng.integers(0, 12))

            # Observation contains measured audio features,
            # but not the hidden target gains.
            features = music._features(music.gains)
            observation = music._observation(features)

            # Oracle supplies the training label only.
            correct_action = np.clip(
                (music.target_gains - music.gains) / 2.0,
                -1.0,
                1.0,
            ).astype(np.float32)

            observations.append(observation)
            labels.append(correct_action)

    env.close()

    return (
        torch.from_numpy(np.stack(observations)),
        torch.from_numpy(np.stack(labels)),
    )


class SupervisedPolicy:
    """Adapter for our existing SAC evaluation function."""

    def __init__(self, actor):
        self.actor = actor

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)

        mean, _ = self.actor(x)

        return (
            torch.tanh(mean)
            .squeeze(0)
            .cpu()
            .numpy()
        )


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    print("Preparing training data from 12 songs...")
    x_train, y_train = make_dataset(
        "train",
        samples_per_song=600,
        seed_base=100_000,
    )

    print("Preparing validation data from 4 songs...")
    x_val, y_val = make_dataset(
        "validation",
        samples_per_song=200,
        seed_base=300_000,
    )

    print("Training observations:", tuple(x_train.shape))
    print("Validation observations:", tuple(x_val.shape))

    obs_dim = x_train.shape[1]

    actor = GaussianActor(
        obs_dim=obs_dim,
        action_dim=4,
    )

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=1e-3,
    )

    best_val_loss = float("inf")
    best_weights = None

    print("\nTraining supervised actor...")

    for epoch in range(1, 81):
        actor.train()

        permutation = torch.randperm(len(x_train))

        for indices in permutation.split(128):
            x = x_train[indices]
            target_actions = y_train[indices]

            mean, _ = actor(x)
            predicted_actions = torch.tanh(mean)

            loss = F.mse_loss(
                predicted_actions,
                target_actions,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        actor.eval()

        with torch.no_grad():
            mean, _ = actor(x_val)

            val_loss = F.mse_loss(
                torch.tanh(mean),
                y_val,
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(
                actor.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:2d} | "
                f"Validation action MSE: {val_loss:.6f}"
            )

    actor.load_state_dict(best_weights)
    actor.eval()

    # Same validation songs and target seeds as SAC.
    val_env = MultiSongMixingEnv(split="validation")

    results = evaluate(
        val_env,
        SupervisedPolicy(actor),
        "sac",
        episodes_per_song=10,
    )

    val_env.close()

    print("\nSUPERVISED ACTOR — VALIDATION")
    print(f"Best action MSE: {best_val_loss:.6f}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Mean final error: {results['mean_error']:.6f}")
    print(f"Mean steps: {results['mean_steps']:.2f}")

    print("\nPER-SONG RESULTS")
    for song, metrics in results["per_song"].items():
        print(
            f"{song}: "
            f"error={metrics['mean_error']:.6f}, "
            f"success={metrics['success_rate']:.1f}%"
        )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": 4,
            "validation_error": results["mean_error"],
        },
        CHECKPOINT_PATH,
    )

    print("\nCheckpoint saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
