from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from wadidaw.rl.networks import GaussianActor, QNetwork


class SACAgent:
    def __init__(
        self,
        obs_dim=12,
        action_dim=4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        learning_rate=3e-4,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Main trainable networks.
        self.actor = GaussianActor(obs_dim, action_dim)
        self.q1 = QNetwork(obs_dim, action_dim)
        self.q2 = QNetwork(obs_dim, action_dim)

        # Slowly updated copies of the critics.
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        for target in (self.target_q1, self.target_q2):
            for param in target.parameters():
                param.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters())
            + list(self.q2.parameters()),
            lr=learning_rate,
        )

    
    @torch.no_grad()
    def act(self, observation, deterministic=False):
        """Choose an action from the actor policy."""
        obs = torch.as_tensor(
            observation, dtype=torch.float32
        ).unsqueeze(0)

        if deterministic:
            # Do not sample random noise during evaluation.
            mean, _ = self.actor(obs)
            action = torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(obs)

        return action.squeeze(0).cpu().numpy()


    def update(self, batch):
        """Perform one SAC gradient update."""

        obs = torch.as_tensor(
            batch["observations"], dtype=torch.float32
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32
        )
        next_obs = torch.as_tensor(
            batch["next_observations"], dtype=torch.float32
        )
        terminated = torch.as_tensor(
            batch["terminated"], dtype=torch.float32
        )

        # 1. Calculate the Bellman target.
        with torch.no_grad():
            next_actions, next_log_probs, _ = (
                self.actor.sample(next_obs)
            )

            next_q = torch.minimum(
                self.target_q1(next_obs, next_actions),
                self.target_q2(next_obs, next_actions),
            )

            target_q = rewards + self.gamma * (
                1.0 - terminated
            ) * (
                next_q - self.alpha * next_log_probs
            )

        # 2. Train both critics.
        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)

        critic_loss = (
            F.mse_loss(q1_pred, target_q)
            + F.mse_loss(q2_pred, target_q)
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Freeze critic parameters for actor training.
        # Gradients must still flow through Q(s, action)
        # back into the actor's sampled actions.
        for critic in (self.q1, self.q2):
            for param in critic.parameters():
                param.requires_grad_(False)

        # 4. Train the actor.
        new_actions, log_probs, _ = (
            self.actor.sample(obs)
        )

        q_new = torch.minimum(
            self.q1(obs, new_actions),
            self.q2(obs, new_actions),
        )

        actor_loss = (
            self.alpha * log_probs - q_new
        ).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Re-enable critic gradients for the next update.
        for critic in (self.q1, self.q2):
            for param in critic.parameters():
                param.requires_grad_(True)

        # 5. Soft-update target critics (Polyak averaging).
        with torch.no_grad():
            pairs = (
                (self.q1, self.target_q1),
                (self.q2, self.target_q2),
            )

            for online, target in pairs:
                for p, target_p in zip(
                    online.parameters(),
                    target.parameters(),
                ):
                    target_p.lerp_(p, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_target_q": target_q.mean().item(),
        }
