import torch
from torch import nn

from pufferlib.pytorch import layer_init

from puffer_phc.policies.discriminator_policy import DiscriminatorPolicy


class PHCPolicy(DiscriminatorPolicy):
    def __init__(self, env, hidden_size=512):
        super().__init__(env, hidden_size)

        # NOTE: Original PHC network + LayerNorm
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
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # NOTE: Original PHC network + LayerNorm
        self.critic_mlp = nn.Sequential(
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
