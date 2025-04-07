import torch
from torch import nn

from pufferlib.pytorch import layer_init

from puffer_phc.policies.running_norm import RunningNorm


class DiscriminatorPolicy(nn.Module):
    def __init__(self, env, hidden_size=512):
        super().__init__()
        self.is_continuous = True
        self._deterministic_action = False

        self.input_size = env.single_observation_space.shape[0]
        self.action_size = env.single_action_space.shape[0]

        # Assume the action space is symmetric (low=-high)
        self.soft_bound = 0.9 * env.single_action_space.high[0]

        self.obs_norm = torch.jit.script(RunningNorm(self.input_size))

        ### Actor
        self.actor_mlp = None
        self.mu = nn.Sequential(
            layer_init(nn.Linear(hidden_size, self.action_size), std=0.01),
        )

        # NOTE: Original PHC uses a constant std. Something to experiment?
        self.sigma = nn.Parameter(
            torch.zeros(self.action_size, requires_grad=False, dtype=torch.float32),
            requires_grad=False,
        )
        nn.init.constant_(self.sigma, -2.9)

        ### Critic
        self.critic_mlp = None

        ### Discriminator
        self.use_amp_obs = env.amp_observation_space is not None
        self.amp_obs_norm = None

        if self.use_amp_obs:
            amp_obs_size = env.amp_observation_space.shape[0]
            self.amp_obs_norm = torch.jit.script(RunningNorm(amp_obs_size))

            self._disc_mlp = nn.Sequential(
                layer_init(nn.Linear(amp_obs_size, 1024)),
                nn.ReLU(),
                layer_init(nn.Linear(1024, hidden_size)),
                nn.ReLU(),
            )
            self._disc_logits = layer_init(torch.nn.Linear(hidden_size, 1))

        self.obs_pointer = None
        self.mean_bound_loss = None

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, obs):
        raise NotImplementedError

    def decode_actions(self, hidden, lookup=None):
        raise NotImplementedError

    def set_deterministic_action(self, value):
        self._deterministic_action = value

    def discriminate(self, amp_obs):
        if not self.use_amp_obs:
            return None

        norm_amp_obs = self.amp_obs_norm(amp_obs)
        disc_mlp_out = self._disc_mlp(norm_amp_obs)
        disc_logits = self._disc_logits(disc_mlp_out)
        return disc_logits

    # NOTE: Used for network weight regularization
    # def disc_logit_weights(self):
    #     return torch.flatten(self._disc_logits.weight)

    # def disc_weights(self):
    #     weights = []
    #     for m in self._disc_mlp.modules():
    #         if isinstance(m, nn.Linear):
    #             weights.append(torch.flatten(m.weight))

    #     weights.append(torch.flatten(self._disc_logits.weight))
    #     return weights

    def update_obs_rms(self, obs):
        self.obs_norm.update(obs)

    def update_amp_obs_rms(self, amp_obs):
        if not self.use_amp_obs:
            return

        self.amp_obs_norm.update(amp_obs)

    def bound_loss(self, mu):
        mu_loss = torch.zeros_like(mu)
        mu_loss = torch.where(mu > self.soft_bound, (mu - self.soft_bound) ** 2, mu_loss)
        mu_loss = torch.where(mu < -self.soft_bound, (mu + self.soft_bound) ** 2, mu_loss)
        return mu_loss.mean()
