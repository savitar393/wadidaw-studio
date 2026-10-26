from pathlib import Path

import numpy as np
import torch

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.sac import SACAgent


ROOT = Path(__file__).resolve().parents[1]


def feedback_action(observation):
    current = np.maximum(observation[4:8], 1e-6)
    target = np.maximum(observation[8:12], 1e-6)

    correction_db = 20.0 * np.log10(target / current)

    return np.clip(
        correction_db / 2.0, -1.0, 1.0
    ).astype(np.float32)


def immediate_error(env, action):
    """Evaluate one action without modifying the environment."""
    next_gains = np.clip(
        env.gains + 2.0 * action, -18.0, 12.0
    )

    features = env._features_from_gains(next_gains)
    return env._error(features)


@torch.no_grad()
def predicted_q(agent, observation, action):
    obs = torch.as_tensor(
        observation, dtype=torch.float32
    ).unsqueeze(0)

    act = torch.as_tensor(
        action, dtype=torch.float32
    ).unsqueeze(0)

    return torch.minimum(
        agent.q1(obs, act),
        agent.q2(obs, act),
    ).item()


def main():
    torch.set_num_threads(1)

    checkpoint_path = (
        ROOT / "outputs/checkpoints/best_sac_actor.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    agent = SACAgent(alpha=checkpoint["alpha"])

    agent.actor.load_state_dict(
        checkpoint["actor_state_dict"]
    )
    agent.q1.load_state_dict(
        checkpoint["q1_state_dict"]
    )
    agent.q2.load_state_dict(
        checkpoint["q2_state_dict"]
    )

    agent.actor.eval()
    agent.q1.eval()
    agent.q2.eval()

    env = MixingEnv()

    actor_errors = []
    feedback_errors = []
    q_preferences = []
    actual_preferences = []

    for episode in range(50):
        observation, _ = env.reset(seed=20_000 + episode)

        actor_action = agent.act(
            observation, deterministic=True
        )
        teacher_action = feedback_action(observation)

        actor_error = immediate_error(env, actor_action)
        teacher_error = immediate_error(env, teacher_action)

        actor_q = predicted_q(
            agent, observation, actor_action
        )
        teacher_q = predicted_q(
            agent, observation, teacher_action
        )

        actor_errors.append(actor_error)
        feedback_errors.append(teacher_error)

        q_preferences.append(teacher_q > actor_q)
        actual_preferences.append(
            teacher_error < actor_error
        )

        if episode < 5:
            print(f"\nEpisode {episode + 1}")
            print(f"  SAC action: {actor_action.round(3)}")
            print(f"  Feedback:   {teacher_action.round(3)}")
            print(f"  SAC next error: {actor_error:.6f}")
            print(f"  Feedback next error: {teacher_error:.6f}")
            print(f"  SAC Q-value: {actor_q:.4f}")
            print(f"  Feedback Q-value: {teacher_q:.4f}")

    env.close()

    print("\nSUMMARY — 50 UNSEEN INITIAL STATES")
    print(
        "Mean immediate error, SAC:",
        f"{np.mean(actor_errors):.6f}",
    )
    print(
        "Mean immediate error, feedback:",
        f"{np.mean(feedback_errors):.6f}",
    )
    print(
        "Feedback actually reduces error more often:",
        f"{100 * np.mean(actual_preferences):.1f}%",
    )
    print(
        "Critics prefer feedback actions:",
        f"{100 * np.mean(q_preferences):.1f}%",
    )


if __name__ == "__main__":
    main()
