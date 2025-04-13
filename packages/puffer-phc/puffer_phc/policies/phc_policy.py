from typing import Callable, Dict, Sequence, List, Optional
import torch
from torch import nn

from pufferlib.pytorch import layer_init

from puffer_phc.policies.discriminator_policy import DiscriminatorPolicy


def mlp(layer_sizes: Sequence[int], activation: Callable, **activation_kwargs: Optional[Dict]) -> List[nn.Module]:
    layers = []

    for a, b in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        layers.append(layer_init(nn.Linear(a, b)))
        layers.append(activation(**activation_kwargs))

    layers.append(layer_init(nn.Linear(layer_sizes[-2], layer_sizes[-1])))

    return layers


class PHCPolicy(DiscriminatorPolicy):
    def __init__(self, env, hidden_size: int, layer_sizes: Sequence[int]):
        super().__init__(env, hidden_size)

        # NOTE: Original PHC network + LayerNorm
        self.actor_mlp = nn.Sequential(
            *mlp([self.input_size] + list(layer_sizes) + [hidden_size], nn.SiLU),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
            *mlp([self.input_size] + list(layer_sizes) + [hidden_size], nn.SiLU),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=0.01),
        )

    def encode_observations(self, obs):
        # Remember the normed obs to use in the critic
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
            self.mean_bound_loss = self.bound_loss(mu)

        # NOTE: Separate critic network takes input directly
        value = self.critic_mlp(self.obs_pointer)
        return probs, value


class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()

        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x)) / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        with torch.no_grad():
            x = x.float()
            assert x.dim() == 2, "x must be 2D"
            mean = x.mean(0, keepdim=True)
            var = x.var(0, unbiased=False, keepdim=True)
            weight = 1 / self.count
            self.running_mean = self.running_mean * (1 - weight) + mean * weight
            self.running_var = self.running_var * (1 - weight) + var * weight
            self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]
