from residual_shared_autonomy.lunar_lander.actor import lunar_lander_policy_fn, lunar_lander_qf_fn
from residual_shared_autonomy.lunar_lander.fuel_wrapper import LunarLanderFuelWrapper
try:
    from residual_shared_autonomy.lunar_lander.joystick_agent import LunarLanderJoystickActor
    from residual_shared_autonomy.lunar_lander.keyboard_agent import LunarLanderKeyboardActor
except:
    pass

from gym.envs.registration import register

register(
    id='LunarLanderRandom-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_lander:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomContinuous-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_lander:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomNoGoal-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_lander:LunarLanderNoGoal',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomContinuousNoGoal-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_lander:LunarLanderContinuousNoGoal',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderRandomConstrained-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_lander:LunarLanderConstrained',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='LunarLanderReacher-v2',
    entry_point='residual_shared_autonomy.lunar_lander.env_reacher:LunarLanderReacher',
    max_episode_steps=1000,
    reward_threshold=200
)
