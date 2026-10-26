
import torch

from wadidaw.mixing_env import MixingEnv
from wadidaw.rl.replay_buffer import ReplayBuffer
from wadidaw.rl.networks import GaussianActor, QNetwork


def main():
    torch.manual_seed(42)

    env = MixingEnv()
    env.action_space.seed(42)

    buffer = ReplayBuffer(
        capacity=128,
        obs_dim=12,
        action_dim=4,
    )

    observation, _ = env.reset(seed=42)

    # Collect a small amount of real experience.
    for step in range(64):
        action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = (
            env.step(action)
        )

        buffer.add(
            observation,
            action,
            reward,
            next_obs,
            terminated,
            truncated,
        )

        if terminated or truncated:
            observation, _ = env.reset(
                seed=43 + step
            )
        else:
            observation = next_obs

    batch = buffer.sample(32)

    observations = torch.from_numpy(
        batch["observations"]
    )
    stored_actions = torch.from_numpy(
        batch["actions"]
    )

    actor = GaussianActor()
    q1 = QNetwork()
    q2 = QNetwork()

    # Actor forward pass.
    actions, log_probs, deterministic = (
        actor.sample(observations)
    )

    # Critics evaluate stored actions.
    q1_values = q1(observations, stored_actions)
    q2_values = q2(observations, stored_actions)

    assert actions.shape == (32, 4)
    assert log_probs.shape == (32, 1)
    assert q1_values.shape == (32, 1)
    assert q2_values.shape == (32, 1)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()

    assert (
        (actions >= -1.0) & (actions <= 1.0)
    ).all()

    # Test critic gradient flow.
    critic_probe = (
        q1_values.square() + q2_values.square()
    ).mean()

    critic_probe.backward()

    assert any(
        p.grad is not None for p in q1.parameters()
    )
    assert any(
        p.grad is not None for p in q2.parameters()
    )

    # Freeze critics while testing actor gradients.
    for network in (q1, q2):
        network.zero_grad(set_to_none=True)

        for parameter in network.parameters():
            parameter.requires_grad_(False)

    new_actions, new_log_probs, _ = (
        actor.sample(observations)
    )

    actor_probe = (
        0.2 * new_log_probs
        - torch.minimum(
            q1(observations, new_actions),
            q2(observations, new_actions),
        )
    ).mean()

    actor_probe.backward()

    assert any(
        p.grad is not None for p in actor.parameters()
    )

    print("Actor actions:", tuple(actions.shape))
    print("Log probabilities:", tuple(log_probs.shape))
    print("Critic 1:", tuple(q1_values.shape))
    print("Critic 2:", tuple(q2_values.shape))
    print("Deterministic:", tuple(deterministic.shape))
    print("Action bounds: OK")
    print("Gradient flow: OK")

    env.close()


if __name__ == "__main__":
    main()
