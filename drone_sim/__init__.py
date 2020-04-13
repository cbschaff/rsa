from residual_shared_autonomy.drone_sim.sim import Drone
from residual_shared_autonomy.drone_sim.rendering import DroneRenderer, DroneFollower
from residual_shared_autonomy.drone_sim.actor import drone_ppo_policy_fn
try:
    from residual_shared_autonomy.drone_sim.joystick_agent import DroneJoystickActor
except:
    pass

from gym.envs.registration import register

register(
    id='DroneReacherHuman-v0',
    entry_point='residual_shared_autonomy.drone_sim.env:DroneReacherHuman',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='DroneReacherBot-v0',
    entry_point='residual_shared_autonomy.drone_sim.env:DroneReacherBot',
    max_episode_steps=1000,
    reward_threshold=200
)

register(
    id='DroneStabilizer-v0',
    entry_point='residual_shared_autonomy.drone_sim.env:DroneStabilizer',
    max_episode_steps=1000,
    reward_threshold=200
)
