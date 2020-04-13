"""Residual Policy environment wrapper."""
from gym.spaces import Tuple, Box
from baselines.common.vec_env import VecEnvWrapper
import numpy as np


class ResidualWrapper(VecEnvWrapper):
    """Wrapper for residual policy learning.

    https://arxiv.org/abs/1812.06298

    Requires a callable which returns an action. The chosen action is added to
    the observation.
    """

    def __init__(self, venv, act_fn):
        """Init."""
        super().__init__(venv)
        self.act_fn = act_fn
        self._ob = None
        if not isinstance(self.action_space, Box):
            raise ValueError("ResidualWrapper can only be used with continuous "
                             "action spaces.")
        self.observation_space = Tuple([self.observation_space,
                                        self.action_space])
        bound = self.action_space.high - self.action_space.low
        self.action_space = Box(-bound, bound)
        self._action = None

    def reset(self):
        """Reset."""
        ob = self.venv.reset()
        self._action = np.asarray(self.act_fn(ob))
        if hasattr(self.act_fn, 'reset'):
            # reset joystick to zero action
            self.act_fn.reset()
        return (ob, self._norm_action(self._action))

    def step(self, action):
        """Step."""
        action = self._add_actions(np.asarray(action), self._action)
        ob, rs, dones, infos = self.venv.step(action)
        for i, info in enumerate(infos):
            info['action'] = action[i]
            info['assistant_action'] = np.asarray(action)[i]
            info['player_action'] = self._action[i]
        self._action = self.act_fn(ob)
        return (ob, self._norm_action(self._action)), rs, dones, infos

    def _clip_action(self, ac):
        return np.maximum(np.minimum(ac, self.venv.action_space.high),
                          self.venv.action_space.low)

    def _add_actions(self, ac1, ac2):
        return self._clip_action(ac1 + ac2)

    def _norm_action(self, ac):
        high = self.venv.action_space.high
        low = self.venv.action_space.low
        return 2 * (ac - low) / (high - low) - 1.0

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()


if __name__ == '__main__':
    import unittest
    import gym
    from dl.rl import ensure_vec_env, VecFrameStack

    class ZeroActor(object):
        """Output zeros."""

        def __init__(self, action_space):
            """Init."""
            self.action_space = action_space

        def __call__(self, ob):
            """Act."""
            batch_size = ob.shape[0]
            return np.zeros_like([self.action_space.sample()
                                  for _ in range(batch_size)])

    class RandomActor(object):
        """Output random actions."""

        def __init__(self, action_space):
            """Init."""
            self.action_space = action_space

        def __call__(self, ob):
            """Act."""
            batch_size = ob.shape[0]
            return np.asarray([self.action_space.sample()
                               for _ in range(batch_size)])

    class Test(unittest.TestCase):
        """Tests."""

        def test_zero_actor(self):
            """Test."""
            env = gym.make("LunarLanderContinuous-v2")
            env = ensure_vec_env(env)
            actor = ZeroActor(env.action_space)
            env = ResidualWrapper(env, actor)
            ob, ac = env.reset()
            assert np.allclose(ac, 0)
            assert ac.shape == (1, *env.action_space.shape)
            assert ob.shape == (1, *env.observation_space.spaces[0].shape)
            assert ac.shape == (1, *env.observation_space.spaces[1].shape)
            assert isinstance(env.observation_space, Tuple)

            residual_ac = [env.action_space.sample()]
            (ob, ac), _, _, infos = env.step(residual_ac)
            rac = np.minimum(np.maximum(residual_ac[0], -1), 1)
            assert np.allclose(infos[0]['action'], rac)
            assert np.allclose(ac, 0)
            assert ac.shape == (1, *env.action_space.shape)
            assert ob.shape == (1, *env.observation_space.spaces[0].shape)
            assert ac.shape == (1, *env.observation_space.spaces[1].shape)

        def test_random_actor(self):
            """Test."""
            env = gym.make("LunarLanderContinuous-v2")
            env = ensure_vec_env(env)
            actor = RandomActor(env.action_space)
            env = ResidualWrapper(env, actor)
            ob, ac = env.reset()
            assert ac.shape == (1, *env.action_space.shape)
            assert ob.shape == (1, *env.observation_space.spaces[0].shape)
            assert ac.shape == (1, *env.observation_space.spaces[1].shape)
            assert isinstance(env.observation_space, Tuple)

            for _ in range(10):
                residual_ac = [env.action_space.sample()]
                (ob, ac_next), _, _, infos = env.step(residual_ac)
                rac = np.minimum(np.maximum(residual_ac[0] + ac[0], -1), 1)
                assert np.allclose(infos[0]['action'], rac)
                ac = ac_next

        def test_vec_env_wrapper(self):
            """Test."""
            env = gym.make("LunarLanderContinuous-v2")
            env = ensure_vec_env(env)
            actor = RandomActor(env.action_space)
            env = ResidualWrapper(env, actor)
            env = VecFrameStack(env, 4)
            ob, ac = env.reset()
            assert ac.shape == (1, 4*env.action_space.shape[0])
            assert ob.shape == (1, *env.observation_space.spaces[0].shape)
            assert ac.shape == (1, *env.observation_space.spaces[1].shape)
            assert isinstance(env.observation_space, Tuple)

            for _ in range(10):
                residual_ac = [env.action_space.sample()]
                (ob, ac_next), _, _, infos = env.step(residual_ac)
                rac = np.minimum(np.maximum(residual_ac[0] + ac[0][-2:], -1), 1)
                assert np.allclose(infos[0]['action'], rac)
                ac = ac_next

    unittest.main()
