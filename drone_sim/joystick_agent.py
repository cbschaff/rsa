"""LunarLander Joystick Agent."""
import pygame
import numpy as np
import time


#####################################
# Change these to match your joystick
THRUST_AXIS = 1
ROLL_AXIS = 3
PITCH_AXIS = 4
YAW_AXIS = 0
#####################################


class DroneJoystickActor(object):
    """Joystick Controller for Drone tasks.

    The left stick controls upward force and yaw.
    The right stick controls roll and pitch.
    """

    def __init__(self, env, fps=50):
        """Init."""
        if env.num_envs > 1:
            raise ValueError("Only one env can be controlled with the joystick")
        self.env = env
        self.human_agent_action = np.zeros((1, 4), dtype=np.float32)
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        if len(joysticks) != 1:
            raise ValueError("There must be exactly 1 joystick connected."
                             f"Found {len(joysticks)}")
        self.joy = joysticks[0]
        self.joy.init()
        pygame.init()
        self.axes = [THRUST_AXIS, ROLL_AXIS, PITCH_AXIS, YAW_AXIS]
        self.t = None
        self.fps = fps

    def _get_human_action(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                for i, ax in enumerate(self.axes):
                    if event.axis == ax:
                        self.human_agent_action[0, i] = event.value
                        break
        return self.human_agent_action

    def __call__(self, ob):
        """Act."""
        if self.t and (time.time() - self.t) < 1. / self.fps:
            st = 1. / self.fps - (time.time() - self.t)
            if st > 0.:
                time.sleep(st)
        self.t = time.time()
        self.env.render()
        action = self._get_human_action()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.


if __name__ == '__main__':
    import gym
    import residual_shared_autonomy.drone_sim
    from dl.rl import ensure_vec_env

    env = gym.make("DroneReacherHuman-v0")
    env = ensure_vec_env(env)

    actor = DroneJoystickActor(env)

    for _ in range(10):
        ob = env.reset()
        actor.reset()
        env.render()
        done = False
        reward = 0.0

        while not done:
            ob, r, done, _ = env.step(actor(ob))
            reward += r
        print(reward)
