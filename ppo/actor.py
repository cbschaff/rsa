"""Defines networks for Residual PPO experiments."""
from dl.rl.modules import ActorCriticBase, Policy
from dl.modules import DiagGaussian
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


class FeedForwardActorCriticBase(ActorCriticBase):
    """Policy and Value networks."""

    def __init__(self, *args, nunits=128, **kwargs):
        """Init."""
        self.nunits = nunits
        super().__init__(*args, **kwargs)

    def build(self):
        """Build."""
        inshape = (self.observation_space.spaces[0].shape[0]
                   + self.observation_space.spaces[1].shape[0])
        self.fc1 = nn.Linear(inshape, self.nunits)
        self.fc2 = nn.Linear(self.nunits, self.nunits)
        self.dist = DiagGaussian(self.nunits, self.action_space.shape[0])
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

        self.vf_fc1 = nn.Linear(inshape, self.nunits)
        self.vf_fc2 = nn.Linear(self.nunits, self.nunits)
        self.vf_out = nn.Linear(self.nunits, 1)

    def forward(self, x):
        """Forward."""
        ob = torch.cat(x, axis=1)
        net = ob
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        pi = self.dist(net)

        net = ob
        net = F.relu(self.vf_fc1(net))
        net = F.relu(self.vf_fc2(net))
        vf = self.vf_out(net)

        return pi, vf


@gin.configurable
def ppo_policy_fn(env, nunits=128):
    """Create a policy network."""
    return Policy(FeedForwardActorCriticBase(env.observation_space,
                                             env.action_space,
                                             nunits=nunits))
