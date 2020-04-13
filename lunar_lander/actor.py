"""Defines networks for Lunar Lander experiments."""
from dl.rl.modules import PolicyBase, ContinuousQFunctionBase
from dl.rl.modules import QFunction, UnnormActionPolicy
from dl.modules import TanhDelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


class FeedForwardPolicyBase(PolicyBase):
    """Policy network."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(9, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dist = TanhDelta(256, self.action_space.shape[0])

    def forward(self, x):
        """Forward."""
        x = x[:, :9]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.dist(x)


class AppendActionFeedForwardQFBase(ContinuousQFunctionBase):
    """Q network."""

    def build(self):
        """Build."""
        nin = 9 + self.action_space.shape[0]
        self.fc1 = nn.Linear(nin, 256)
        self.fc2 = nn.Linear(256, 256)
        self.qvalue = nn.Linear(256, 1)

    def forward(self, x, a):
        """Forward."""
        x = x[:, :9]
        x = F.relu(self.fc1(torch.cat([x, a], dim=1)))
        x = F.relu(self.fc2(x))
        return self.qvalue(x)


@gin.configurable
def lunar_lander_policy_fn(env):
    """Create a policy network."""
    return UnnormActionPolicy(FeedForwardPolicyBase(env.observation_space,
                                                    env.action_space))


@gin.configurable
def lunar_lander_qf_fn(env):
    """Create a qfunction network."""
    return QFunction(AppendActionFeedForwardQFBase(env.observation_space,
                                                   env.action_space))
