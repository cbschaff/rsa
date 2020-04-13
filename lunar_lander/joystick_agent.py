"""LunarLander Joystick Agent."""
import pygame
import numpy as np
import time

#####################################
# Change these to match your joystick
UP_AXIS = 4
SIDE_AXIS = 3
#####################################


class LunarLanderJoystickActor(object):
    """Joystick Controller for Lunar Lander."""

    def __init__(self, env, fps=50):
        """Init."""
        if env.num_envs > 1:
            raise ValueError("Only one env can be controlled with the joystick.")
        self.env = env
        self.human_agent_action = np.array([[0., 0.]], dtype=np.float32)  # noop
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        if len(joysticks) != 1:
            raise ValueError("There must be exactly 1 joystick connected."
                             f"Found {len(joysticks)}")
        self.joy = joysticks[0]
        self.joy.init()
        pygame.init()
        self.t = None
        self.fps = fps

    def _get_human_action(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == SIDE_AXIS:
                    self.human_agent_action[0, 1] = event.value
                elif event.axis == UP_AXIS:
                    self.human_agent_action[0, 0] = -1.0 * event.value
        if abs(self.human_agent_action[0, 0]) < 0.1:
            self.human_agent_action[0, 0] = 0.0
        return self.human_agent_action

    def __call__(self, ob):
        """Act."""
        self.env.render()
        action = self._get_human_action()
        if self.t and (time.time() - self.t) < 1. / self.fps:
            st = 1. / self.fps - (time.time() - self.t)
            if st > 0.:
                time.sleep(st)
        self.t = time.time()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.


if __name__ == '__main__':
    import gym
    import residual_shared_autonomy.lunar_lander
    from dl.rl import ensure_vec_env
    import argparse

    parser = argparse.ArgumentParser(description='play')
    parser.add_argument('--reacher', action='store_true', help='play lunar reacher')
    args = parser.parse_args()


    if args.reacher:
        env = gym.make("LunarLanderReacher-v2")
    else:
        env = gym.make("LunarLanderRandomContinuous-v2")
    env = ensure_vec_env(env)

    actor = LunarLanderJoystickActor(env)

    for _ in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0

        while not done:
            ob, r, done, _ = env.step(actor(ob))
            reward += r
        print(reward)
