import numpy as np
from scipy.optimize import least_squares

from wadidaw.multi_song_env import MultiSongMixingEnv


PAIRS = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3),
]


def predict_band_db(observation, gains_db):
    """Predict mixed spectral features from observation alone."""

    obs = np.asarray(observation, dtype=np.float64)

    if obs.shape != (101,):
        raise ValueError("Expected a 101-feature observation.")

    # Recover individual stem energies from their dB values.
    stem_db = obs[20:52].reshape(4, 8) * 100.0
    stem_energy = 10.0 ** (stem_db / 10.0)

    # Recover normalized cross-stem interactions.
    correlations = obs[52:100].reshape(6, 8)

    # Convert candidate gains from dB to linear amplitude.
    gains_db = np.asarray(gains_db, dtype=np.float64)
    amplitudes = 10.0 ** (gains_db / 20.0)

    # Diagonal contribution: sum(a_i^2 * G_ii).
    power = np.sum(
        amplitudes[:, None] ** 2 * stem_energy,
        axis=0,
    )

    # Off-diagonal contribution:
    # 2 * a_i * a_j * rho_ij * sqrt(G_ii * G_jj).
    for pair_index, (i, j) in enumerate(PAIRS):
        cross_energy = (
            correlations[pair_index]
            * np.sqrt(stem_energy[i] * stem_energy[j])
        )

        power += (
            2.0
            * amplitudes[i]
            * amplitudes[j]
            * cross_energy
        )

    power = np.maximum(power, 1e-12)

    return np.clip(
        10.0 * np.log10(power),
        -120.0,
        20.0,
    )


def solve_from_observation(observation):
    """Find gains using no hidden environment information."""

    target_db = (
        observation[12:20].astype(np.float64) * 100.0
    )

    current_gains = (
        observation[:4].astype(np.float64) * 18.0
    )

    def residual(gains):
        predicted_db = predict_band_db(
            observation, gains
        )
        return (predicted_db - target_db) / 12.0

    result = least_squares(
        residual,
        x0=current_gains,
        bounds=(-18.0, 12.0),
        max_nfev=300,
    )

    return (
        result.x.astype(np.float32),
        float(np.mean(result.fun ** 2)),
    )


def main():
    env = MultiSongMixingEnv(split="validation")

    successes = []
    errors = []
    fit_errors = []
    gain_differences = []
    consistency_errors = []

    print("OBSERVATION-ONLY INVERSE MIXING")
    print("Validation songs only. No hidden mixing matrices.\n")

    for position, member in enumerate(env.members):
        song_successes = []
        song_errors = []

        for episode in range(10):
            seed = 50_000 + position * 1000 + episode

            obs, info = env.reset(
                seed=seed,
                options={"song_position": position},
            )

            # Verify that our reconstructed model reproduces
            # the current spectrum contained in the observation.
            current_gains = obs[:4] * 18.0

            predicted_current = predict_band_db(
                obs, current_gains
            )
            observed_current = obs[4:12] * 100.0

            consistency = float(np.max(
                np.abs(predicted_current - observed_current)
            ))
            consistency_errors.append(consistency)

            # Infer the target gains using the observation only.
            estimated_gains, fit_error = (
                solve_from_observation(obs)
            )

            # Hidden target gains are used ONLY to report
            # estimation accuracy, never inside the solver.
            music = env.current_env

            gain_difference = float(np.mean(
                np.abs(
                    estimated_gains - music.target_gains
                )
            ))

            if position == 0 and episode == 0:
                print("EXAMPLE")
                print(
                    "True gains (reporting only):",
                    music.target_gains.round(3)
                )
                print(
                    "Estimated gains:",
                    estimated_gains.round(3)
                )
                print(
                    "Model consistency:",
                    f"{consistency:.8f} dB"
                )
                print()

            # Apply the estimated gains through normal actions.
            for step in range(music.max_steps):
                action = np.clip(
                    (estimated_gains - music.gains) / 2.0,
                    -1.0,
                    1.0,
                ).astype(np.float32)

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                if terminated or truncated:
                    break

            successes.append(float(info["success"]))
            errors.append(float(info["error"]))
            fit_errors.append(fit_error)
            gain_differences.append(gain_difference)

            song_successes.append(float(info["success"]))
            song_errors.append(float(info["error"]))

        print(
            f"{member['song']} | "
            f"success={np.mean(song_successes) * 100:.1f}% | "
            f"error={np.mean(song_errors):.6f}"
        )

    print("\nSUMMARY — 40 VALIDATION TASKS")
    print(
        "Success rate:",
        f"{np.mean(successes) * 100:.1f}%"
    )
    print(
        "Mean final error:",
        f"{np.mean(errors):.6f}"
    )
    print(
        "Mean feature-fit error:",
        f"{np.mean(fit_errors):.8f}"
    )
    print(
        "Mean gain difference:",
        f"{np.mean(gain_differences):.6f} dB"
    )
    print(
        "Worst observation-model discrepancy:",
        f"{np.max(consistency_errors):.8f} dB"
    )

    env.close()


if __name__ == "__main__":
    main()
