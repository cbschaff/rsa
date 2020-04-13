"""Base actors on which residuals are learned."""
import numpy as np
import torch
from residual_shared_autonomy.imitation_learning import BCNet
from residual_shared_autonomy.lunar_lander import lunar_lander_policy_fn
from residual_shared_autonomy.drone_sim import drone_ppo_policy_fn
from dl import Checkpointer
import gin
import os


@gin.configurable
class ZeroActor(object):
    """Output random actions."""

    def __init__(self, env):
        """Init."""
        self.action_space = env.action_space
        self.batch_size = env.num_envs

    def __call__(self, ob):
        """Act."""
        return np.zeros([self.batch_size] + list(self.action_space.shape),
                        dtype=np.float32)


@gin.configurable
class RandomActor(object):
    """Output random actions."""

    def __init__(self, env):
        """Init."""
        self.action_space = env.action_space
        self.batch_size = env.num_envs

    def __call__(self, ob):
        """Act."""
        return np.asarray([self.action_space.sample()
                           for _ in range(self.batch_size)])


@gin.configurable
class PolicyActor(object):
    """policy actor"""

    def __init__(self, pi, device):
        self.pi = pi
        self.device = device

    def __call__(self, ob):
        """Act."""
        if isinstance(ob, np.ndarray):
            ob = torch.from_numpy(ob).to(self.device)
        return self.pi(ob).action.cpu().numpy()


@gin.configurable
class LunarLanderActor(object):
    """Lunar Lander actor."""

    def __init__(self, env, logdir, device):
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        if not torch.cuda.is_available():
            device = 'cpu'
        self.pi = lunar_lander_policy_fn(env)
        self.pi.to(device)
        self.pi.load_state_dict(self.ckptr.load()['pi'])
        self.pi.eval()
        self.device = device

    def __call__(self, ob):
        """Act."""
        with torch.no_grad():
            if isinstance(ob, np.ndarray):
                ob = torch.from_numpy(ob).to(self.device)
            return self.pi(ob).action.cpu().numpy()


@gin.configurable
class DroneReacherActor(object):
    """DroneReacher actor."""

    def __init__(self, env, logdir, device):
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        if not torch.cuda.is_available():
            device = 'cpu'
        self.pi = drone_ppo_policy_fn(env)
        self.pi.to(device)
        self.pi.load_state_dict(self.ckptr.load()['pi'])
        self.pi.eval()
        self.device = device

    def __call__(self, ob):
        """Act."""
        with torch.no_grad():
            if isinstance(ob, np.ndarray):
                ob = torch.from_numpy(ob).to(self.device)
            return self.pi(ob).action.cpu().numpy()


@gin.configurable
class LaggyActor(object):
    """Laggy actor"""

    def __init__(self, env, actor_cls, repeat_prob):
        self.actor = actor_cls(env)
        self.repeat_prob = repeat_prob
        self.action = None

    def __call__(self, ob):
        """Act."""
        if self.action is None or np.random.rand() > self.repeat_prob:
            self.action = self.actor(ob)
        return self.action


@gin.configurable
class NoisyActor(object):
    """Noisy actor"""

    def __init__(self, env, actor_cls, eps):
        self.actor = actor_cls(env)
        self.eps = eps

    def __call__(self, ob):
        """Act."""
        action = self.actor(ob)
        if np.random.rand() < self.eps:
            action = np.random.uniform(-1, 1, action.shape).astype(action.dtype)
        return action


@gin.configurable
class BCActor(object):
    """Actor trained with Behavioral cloning"""

    def __init__(self, env, logdir, device):
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.net = BCNet()
        self.net.to(device)
        self.net.load_state_dict(self.ckptr.load()['model'])

    def __call__(self, ob):
        """Act."""
        with torch.no_grad():
            dist = self.net(torch.from_numpy(ob).to(self.device))
            ac = dist.sample().cpu().numpy()
            return np.clip(ac, -1., 1.)


@gin.configurable
class BCMultiActor(object):
    """Use multiple actors trained with Behavioral cloning"""

    def __init__(self, env, logdir, device, switch_prob=0.001):
        dirs = [x for x in os.listdir(logdir) if os.path.isdir(
                                            os.path.join(logdir, x, 'ckpts'))]

        self.ckptrs = [Checkpointer(os.path.join(logdir, x, 'ckpts'))
                       for x in dirs]
        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.nets = [BCNet() for _ in dirs]
        for net, ckptr in zip(self.nets, self.ckptrs):
            net.to(device)
            net.load_state_dict(ckptr.load()['model'])
        self.current_actor = np.random.choice(self.nets)
        self.switch_prob = switch_prob

    def __call__(self, ob):
        """Act."""
        if np.random.rand() < self.switch_prob:
            self.current_actor = np.random.choice(self.nets)
        with torch.no_grad():
            if isinstance(ob, np.ndarray):
                ob = torch.from_numpy(ob)
            dist = self.current_actor(ob.to(self.device))
            ac = dist.sample().cpu().numpy()
            return np.clip(ac, -1., 1.)


if __name__ == '__main__':
    import gym
    import residual_shared_autonomy.envs
    from dl.rl import ensure_vec_env
    import time

    env = gym.make("LunarLanderRandomContinuous-v2")
    env = ensure_vec_env(env)

    # actor = OrnsteinUhlenbeckActor(env, 0.5)
    actor = RandomActor(env)

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0
        time.sleep(1.)

        while not done:
            ob, r, done, _ = env.step(actor(ob))
            env.render()
            reward += r
        print(reward)
