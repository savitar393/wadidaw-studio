from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.models.differentiable_mixer import (
    DifferentiableBandMixer,
)
from wadidaw.models.inverse_gain_net import InverseGainNet
from scripts.train_music_sac import evaluate


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "outputs/checkpoints/music_differentiable_v1.pt"
)


def create_dataset(split, samples_per_song, seed_base):
    """Generate observations without hidden gain labels."""

    env = MultiSongMixingEnv(split=split)
    rng = np.random.default_rng(seed_base)

    observations = []

    for position in range(len(env.members)):
        for i in range(samples_per_song):
            seed = seed_base + position * 10_000 + i

            env.reset(
                seed=seed,
                options={"song_position": position},
            )

            music = env.current_env

            # Mix of initial and intermediate gain states.
            if i % 3 == 0:
                gains = np.zeros(4, dtype=np.float32)
                music.steps = 0

            elif i % 3 == 1:
                gains = rng.uniform(
                    -10.0, 10.0, size=4
                ).astype(np.float32)
                music.steps = int(
                    rng.integers(0, music.max_steps)
                )

            else:
                gains = rng.uniform(
                    -5.0, 5.0, size=4
                ).astype(np.float32)
                music.steps = int(
                    rng.integers(0, music.max_steps)
                )

            music.gains = gains

            features = music._features(music.gains)
            obs = music._observation(features)

            assert obs.shape == (101,)
            assert env.observation_space.contains(obs)

            observations.append(obs)

    env.close()

    return torch.from_numpy(
        np.stack(observations)
    )


def reconstruction_loss(model, mixer, observations):
    """Train using spectral error, not target gain labels."""

    predicted_gains = model(observations)

    # The mixer preserves gradients through predicted gains.
    predicted_bands = mixer(
        observations,
        predicted_gains,
    )

    # Target bands are part of the observation.
    target_bands = (
        observations[:, 12:20].to(torch.float64)
        * 100.0
    )

    return (
        (predicted_bands - target_bands) / 12.0
    ).square().mean()


@torch.no_grad()
def validation_loss(model, mixer, observations):
    model.eval()

    total = 0.0
    count = 0

    for batch in observations.split(256):
        loss = reconstruction_loss(model, mixer, batch)

        total += loss.item() * len(batch)
        count += len(batch)

    return total / count


class InversePolicy:
    """Apply predicted absolute gains using bounded actions."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.from_numpy(
            np.asarray(observation, dtype=np.float32)
        ).unsqueeze(0)

        predicted_gains = (
            self.model(x)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # Current gain values are provided in the observation.
        current_gains = observation[:4] * 18.0

        # Maximum adjustment is +/- 2 dB per step.
        return np.clip(
            (predicted_gains - current_gains) / 2.0,
            -1.0,
            1.0,
        ).astype(np.float32)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)

    print("Generating training data...")
    x_train = create_dataset(
        "train", 600, 100_000
    )

    print("Generating validation data...")
    x_val = create_dataset(
        "validation", 200, 300_000
    )

    print("Training observations:", tuple(x_train.shape))
    print("Validation observations:", tuple(x_val.shape))
    print("No hidden target gain labels used.")

    model = InverseGainNet(obs_dim=x_train.shape[1])
    mixer = DifferentiableBandMixer()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    best_val_loss = float("inf")
    best_weights = None
    patience = 15
    epochs_without_improvement = 0

    print("\nTraining differentiable inverse model...")

    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(x_train))

        total_loss = 0.0

        for indices in order.split(128):
            batch = x_train[indices]

            loss = reconstruction_loss(
                model, mixer, batch
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )

            optimizer.step()

            total_loss += loss.item() * len(batch)

        train_loss = total_loss / len(x_train)
        val_loss = validation_loss(model, mixer, x_val)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train loss: {train_loss:.6f} | "
                f"Validation loss: {val_loss:.6f}"
            )

        if epochs_without_improvement >= patience:
            print("Early stopping at epoch", epoch)
            break

    if best_weights is None:
        raise RuntimeError("No valid checkpoint selected.")

    model.load_state_dict(best_weights)
    model.eval()

    # Evaluate full episodes on the SAME validation
    # songs and target seeds used for earlier baselines.
    val_env = MultiSongMixingEnv(split="validation")

    results = evaluate(
        val_env,
        InversePolicy(model),
        "sac",
        episodes_per_song=10,
    )

    val_env.close()

    print("\nDIFFERENTIABLE MODEL — VALIDATION")
    print(f"Best reconstruction loss: {best_val_loss:.6f}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Mean final error: {results['mean_error']:.6f}")
    print(f"Mean steps: {results['mean_steps']:.2f}")

    print("\nPER-SONG VALIDATION")
    for song, metrics in results["per_song"].items():
        print(
            f"{song}: "
            f"error={metrics['mean_error']:.6f}, "
            f"success={metrics['success_rate']:.1f}%"
        )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": x_train.shape[1],
            "best_reconstruction_loss": best_val_loss,
            "validation_error": results["mean_error"],
        },
        CHECKPOINT,
    )

    print("\nCheckpoint saved:", CHECKPOINT)


if __name__ == "__main__":
    main()
