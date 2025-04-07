import torch
from torch import nn

from pufferlib.pytorch import layer_init
import pufferlib.models

from puffer_phc.policies.discriminator_policy import DiscriminatorPolicy


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

        # Point to the original policy's methods
        self.set_deterministic_action = self.policy.set_deterministic_action
        self.discriminate = self.policy.discriminate
        self.update_obs_rms = self.policy.update_obs_rms
        self.update_amp_obs_rms = self.policy.update_amp_obs_rms

    @property
    def mean_bound_loss(self):
        return self.policy.mean_bound_loss


class LSTMCriticPolicy(DiscriminatorPolicy):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # Actor: Original PHC network
        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1536)),
            nn.SiLU(),
            layer_init(nn.Linear(1536, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.SiLU(),
            layer_init(nn.Linear(512, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )
        self.mu = None

        ### Critic with LSTM
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.ReLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the actor
        self.obs_pointer = self.obs_norm(obs)

        # NOTE: hidden goes through LSTM, then to the value (critic head)
        return self.critic_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.actor_mlp(self.obs_pointer)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: hidden from LSTM goes to the critic head
        value = self.value(hidden)
        return probs, value


# NOTE: 13.5M params, Worked for simple motions, but not capable for many, complex motions
class LSTMActorPolicy(DiscriminatorPolicy):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.SiLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.SiLU(),
            layer_init(nn.Linear(1024, hidden_size)),
            nn.SiLU(),
        )

        self.mu = nn.Sequential(
            nn.SiLU(),  # handle the LSTM output
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.input_size, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            # nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            # nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the obs to use in the critic
        self.obs_pointer = self.obs_norm(obs)
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            # mean_violation = nn.functional.relu(torch.abs(mu) - 1)  # bound hard coded to 1
            # self.mean_bound_loss = mean_violation.mean()
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value
