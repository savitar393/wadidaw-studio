
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim=12,
        action_dim=4,
        hidden_dim=128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim, action_dim
        )

        self.log_std_head = nn.Linear(
            hidden_dim, action_dim
        )

    def forward(self, observations):
        features = self.encoder(observations)

        mean = self.mean_head(features)

        log_std = self.log_std_head(features)
        log_std = log_std.clamp(-5.0, 2.0)

        return mean, log_std

    def sample(self, observations):
        mean, log_std = self(observations)

        std = log_std.exp()
        distribution = Normal(mean, std)

        # Reparameterized sampling for backpropagation.
        raw_action = distribution.rsample()

        # Bound every action to (-1, 1).
        action = torch.tanh(raw_action)

        # Correct Gaussian log probability for tanh.
        log_prob = distribution.log_prob(raw_action)

        correction = 2.0 * (
            math.log(2.0)
            - raw_action
            - F.softplus(-2.0 * raw_action)
        )

        log_prob = (
            log_prob - correction
        ).sum(dim=-1, keepdim=True)

        deterministic_action = torch.tanh(mean)

        return action, log_prob, deterministic_action


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim=12,
        action_dim=4,
        hidden_dim=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                obs_dim + action_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations, actions):
        x = torch.cat(
            (observations, actions), dim=-1
        )
        return self.network(x)
