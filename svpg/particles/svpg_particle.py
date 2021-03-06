import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import orthogonal_init
from .distributions import Categorical, DiagGaussian


class SVPGParticleCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SVPGParticleCritic, self).__init__()

        # I * H * h * 1
        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, 1))
        )

    def forward(self, x):
        return self.critic(x)


class SVPGParticleActorBase(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SVPGParticleActorBase, self).__init__()

        # Here is not add output layer.
        # In SVPGParticle Class, add output layer depending on actor type(discrete or continuous).
        self.actor_hidden = nn.Sequential(
            orthogonal_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.actor_hidden(x)


class SVPGParticle(nn.Module):
    """Implements a AC architecture for a Discrete Advantage
    Actor Critic Policy, used inside of SVPG

    Input_dim: params dimension
    Output_dim: same as input
    """

    def __init__(self, input_dim, output_dim, hidden_dim, discrete, freeze=False):
        super(SVPGParticle, self).__init__()

        self.critic = SVPGParticleCritic(input_dim, output_dim, hidden_dim)
        self.actor_hidden = SVPGParticleActorBase(input_dim, hidden_dim)

        if discrete:
            self.dist = Categorical(hidden_dim, output_dim)
        else:
            self.dist = DiagGaussian(hidden_dim, output_dim)

        if freeze:
            self.freeze()

        self.reset()

    def forward(self, x):
        actor_hidden = self.actor_hidden(x)
        dist = self.dist(actor_hidden)
        value = self.critic(x)

        return dist, value

    def freeze(self):
        """Freeze the params, set gradient to false"""
        for param in self.critic.parameters():
            param.requires_grad = False

        for param in self.actor_hidden.parameters():
            param.requires_grad = False

        for param in self.dist.parameters():
            param.requires_grad = False

    def reset(self):
        self.saved_log_probs = []   # Store action log probability
        self.saved_klds = []    # Store KL_divergence of the current particle and prior particle
        self.rewards = []
