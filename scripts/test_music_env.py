
from pathlib import Path

import numpy as np
import soundfile as sf
from gymnasium.utils.env_checker import check_env

from wadidaw.music_env import MusicMixingEnv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "music_env_check"


def main():
    env = MusicMixingEnv()

    check_env(env, skip_render_check=True)
    print("Gymnasium check passed!")

    obs, info = env.reset(seed=42)

    print("\nSong:", env.song_name)
    print("Stems:", env.stem_names)
    print("Observation shape:", obs.shape)
    print("Target gains:", env.target_gains.round(2))
    print("Initial error:", round(info["error"], 6))

    initial_audio = env.render_mix()
    target_audio = env.render_mix(env.target_gains)

    # Oracle controller, for validation only.
    for step in range(env.max_steps):
        action = np.clip(
            (env.target_gains - env.gains) / 2.0,
            -1.0,
            1.0,
        ).astype(np.float32)

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(
            f"Step {step + 1}: "
            f"error={info['error']:.6f}, "
            f"reward={reward:.4f}"
        )

        if terminated or truncated:
            break

    print("\nSuccess:", info["success"])

    final_audio = env.render_mix()

    # Use the same safety scale for all three files.
    peak = max(
        np.max(np.abs(initial_audio)),
        np.max(np.abs(target_audio)),
        np.max(np.abs(final_audio)),
    )

    scale = min(1.0, 0.95 / max(float(peak), 1e-9))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for name, audio in [
        ("initial.wav", initial_audio),
        ("target.wav", target_audio),
        ("reconstructed.wav", final_audio),
    ]:
        sf.write(
            OUTPUT / name,
            audio * scale,
            env.sample_rate,
        )

    print("Audio saved to:", OUTPUT)
    env.close()


if __name__ == "__main__":
    main()
