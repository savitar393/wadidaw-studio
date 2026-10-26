import numpy as np
import torch

from wadidaw.multi_song_env import MultiSongMixingEnv
from wadidaw.models.differentiable_mixer import (
    DifferentiableBandMixer,
)


def main():
    torch.set_num_threads(1)

    env = MultiSongMixingEnv(split="validation")
    mixer = DifferentiableBandMixer()

    maximum_difference = 0.0

    for position in range(len(env.members)):
        for episode in range(3):
            obs, info = env.reset(
                seed=50_000 + position * 1000 + episode,
                options={"song_position": position},
            )

            music = env.current_env

            candidate_gains = [
                np.zeros(4, dtype=np.float32),
                np.array([1, -3, 5, -2], dtype=np.float32),
                music.target_gains.copy(),
            ]

            for gains in candidate_gains:
                predicted = mixer(
                    torch.from_numpy(obs),
                    torch.from_numpy(gains),
                ).detach().numpy()

                actual = music._features(gains)

                difference = float(
                    np.max(np.abs(predicted - actual))
                )

                maximum_difference = max(
                    maximum_difference,
                    difference,
                )

    print(
        "Maximum spectral difference:",
        f"{maximum_difference:.8f} dB",
    )

    assert maximum_difference < 0.001

    # Test a gradient through the audio model.
    observation, _ = env.reset(
        seed=50_000,
        options={"song_position": 0},
    )

    obs_tensor = torch.from_numpy(observation).double()

    gains = torch.zeros(
        4,
        dtype=torch.float64,
        requires_grad=True,
    )

    target = obs_tensor[12:20] * 100.0

    def loss_for(candidate_gains):
        predicted = mixer(obs_tensor, candidate_gains)
        return ((predicted - target) / 12.0).square().mean()

    loss = loss_for(gains)
    loss.backward()

    analytical = gains.grad.detach().numpy().copy()

    # Independently approximate derivatives numerically.
    numerical = np.zeros(4)
    epsilon = 0.001

    for i in range(4):
        positive = gains.detach().clone()
        negative = gains.detach().clone()

        positive[i] += epsilon
        negative[i] -= epsilon

        numerical[i] = (
            loss_for(positive).item()
            - loss_for(negative).item()
        ) / (2.0 * epsilon)

    print("\nInitial reconstruction loss:", loss.item())
    print("Autograd derivatives:", analytical.round(6))
    print("Numerical derivatives:", numerical.round(6))

    assert np.isfinite(analytical).all()

    assert np.allclose(
        analytical,
        numerical,
        rtol=1e-3,
        atol=1e-5,
    )

    print("\nDifferentiable mixer tests passed!")
    env.close()


if __name__ == "__main__":
    main()
