from typing import Callable, Dict, Sequence, List, Optional
import torch
from torch import nn

from pufferlib.pytorch import layer_init

from puffer_phc.policies.discriminator_policy import DiscriminatorPolicy
from puffer_phc.config import PolicyConfig


def mlp(layer_sizes: Sequence[int], activation: Callable, **activation_kwargs: Optional[Dict]) -> List[nn.Module]:
    layers = []

    for a, b in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.append(layer_init(nn.Linear(a, b)))
        layers.append(activation(**activation_kwargs))

    layers.append(layer_init(nn.Linear(layer_sizes[-2], layer_sizes[-1])))

    return layers


class PHCPolicy(DiscriminatorPolicy):
    def __init__(self, env, cfg: PolicyConfig, device: str):
        super().__init__(env, cfg.hidden_size)

        self.cfg = cfg
        self.device = device

        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            *mlp([self.input_size] + list(cfg.layer_sizes) + [cfg.hidden_size], nn.SiLU),
            nn.LayerNorm(cfg.hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            *mlp([self.input_size] + list(cfg.layer_sizes) + [cfg.hidden_size], nn.SiLU),
            nn.LayerNorm(cfg.hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(cfg.hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the critic
        # breakpoint()
        self.obs_pointer = self.obs_norm(obs.to(self.device))
        return self.actor_mlp(self.obs_pointer), None

    def decode_actions(self, hidden, lookup=None):
        mu = self.mu(hidden)
        std = torch.exp(self.sigma).expand_as(mu)

        if self._deterministic_action is True:
            std = torch.clamp(std, max=1e-6)

        probs = torch.distributions.Normal(mu, std)

        # Mean bound loss
        if self.training:
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value
