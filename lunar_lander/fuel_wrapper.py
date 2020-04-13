from baselines.common.vec_env import VecEnvWrapper
from residual_shared_autonomy import ResidualWrapper
from residual_shared_autonomy.lunar_lander import env_lander
import numpy as np
import gin


class LunarLanderFuelWrapper(VecEnvWrapper):
    """Wrap the ResidualWrapper to modify fuel cost for the co-pilot.

    Don't make the co-pilot pay for fuel that the pilot wants to use.
    """

    def __init__(self, venv):
        assert isinstance(venv, ResidualWrapper)
        super().__init__(venv)

    def step(self, actions):
        pilot_actions = self.venv._action
        applied_actions = self.venv._add_actions(np.asarray(actions),
                                                 pilot_actions)
        m_pow_pilot, s_pow_pilot = self._get_power_from_action(pilot_actions)
        m_pow, s_pow = self._get_power_from_action(applied_actions)

        obs, rews, dones, infos = self.venv.step(actions)

        fuel_diff = self._fuel_cost(m_pow - m_pow_pilot, s_pow - s_pow_pilot)

        # print(fuel_diff.shape, rews.shape)
        # print(actions, pilot_actions)
        # print(fuel_diff, m_pow, m_pow_pilot, s_pow, s_pow_pilot)
        # print(lunar_lander_gym.FUEL_COST_MAIN)

        # Don't incentivise co-pilot for reducing the fuel costs of pilot.
        # The co pilot is only incentivised to not add extra fuel costs.
        rews = rews + np.minimum(0.0, fuel_diff)
        return obs, rews, dones, infos

    def _get_power_from_action(self, ac):
        main, side = ac[:, 0], ac[:, 1]
        main_on = main > 0.
        side_on = np.abs(side) > 0.5
        m_power = main_on * (np.clip(main, 0.0, 1.0) + 1.0) * 0.5
        s_power = side_on * np.clip(np.abs(side), 0.5, 1.0)
        return m_power, s_power

    def _fuel_cost(self, main_pow, side_pow):
        return (env_lander.FUEL_COST_MAIN * main_pow
                + env_lander.FUEL_COST_SIDE * side_pow)

    def reset(self):
        """Reset."""
        return self.venv.reset()

    def step_wait(self):
        """Step wait."""
        return self.venv.step_wait()


@gin.configurable
def lunar_lander_fuel_wrapper(env):
    return LunarLanderFuelWrapper(env)


if __name__ == '__main__':
    import unittest
    import gym
    from dl.rl import ensure_vec_env, VecFrameStack

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

        def test_vec_env_wrapper(self):
            """Test."""
            env = gym.make("LunarLanderContinuous-v2")
            env = ensure_vec_env(env)
            actor = RandomActor(env.action_space)
            env = ResidualWrapper(env, actor)
            env = LunarLanderFuelWrapper(env)
            env = VecFrameStack(env, 4)
            ob, ac = env.reset()
            assert ac.shape == (1, 4*env.action_space.shape[0])
            assert ob.shape == (1, *env.observation_space.spaces[0].shape)
            assert ac.shape == (1, *env.observation_space.spaces[1].shape)

            for _ in range(10):
                residual_ac = [env.action_space.sample()]
                (ob, ac_next), _, _, infos = env.step(residual_ac)
                rac = np.minimum(np.maximum(residual_ac[0] + ac[0][-2:], -1), 1)
                assert np.allclose(infos[0]['action'], rac)
                ac = ac_next

    unittest.main()
