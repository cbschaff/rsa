import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from residual_shared_autonomy.drone_sim import Drone
from residual_shared_autonomy.drone_sim import DroneRenderer, DroneFollower

import numpy as np


FPS = 50
N_OBS_DIM = 12
MAX_NUM_STEPS = 1500
SIMS_PER_STEP = 1
DRONE_RADIUS = 0.5
DRONE_LENGTH = 2.0
DRONE_MASS_CENTER = 6.0
DRONE_MASS_ROTOR = 1.0
CAMERA_OFFSET = (-20, 0, -10)
CAMERA_ALPHA_POS = 0.5
CAMERA_ALPHA_ROT = 0.9
MAX_ACC = (20, 1.0)  # upward acceleration, angular acceleration
BOUNDS = (40, 40, -100.)
TARGET_RADIUS = 3.
ACTION_NOISE = 0.05


class DroneEnv(gym.Env, EzPickle):
    """Reacher task in simple drone simulator."""

    goal_in_state = True
    goal_in_reward = True

    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self):
        """Init."""
        EzPickle.__init__(self)
        self.seed()

        self.renderer = DroneRenderer()
        self.camera = DroneFollower(CAMERA_OFFSET, CAMERA_ALPHA_POS,
                                    CAMERA_ALPHA_ROT)
        self.observation_space = spaces.Box(-1., +1,
                                            (N_OBS_DIM + 3
                                             * self.goal_in_state,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)

        self.reset()

    def _out_of_bounds(self, s):
        bx, by, bz = BOUNDS
        if np.abs(s[0]) >= bx:
            return True
        if np.abs(s[1]) >= by:
            return True
        if s[2] < bz or s[2] >= 0.0:
            return True
        return False

    def _at_target(self, s):
        d = np.sqrt(np.sum((s[:3] - self.target) ** 2))
        return d < TARGET_RADIUS

    def _ac_to_force_torque(self, ac):
        range = MAX_ACC[0] - self.drone._g
        # center upward acceleration at -self.drone.g
        force = np.array([
            0.0,
            0.0,
            -self.drone.m * (range * -ac[0] + self.drone._g)
        ])
        torque = np.array([
            MAX_ACC[1] * ac[1] * self.drone.jx,
            MAX_ACC[1] * ac[2] * self.drone.jy,
            MAX_ACC[1] * ac[3] * self.drone.jz,
        ])
        return force, torque

    def _norm_state(self, state):
        bx, by, bz = BOUNDS
        state[0] /= bx
        state[1] /= by
        state[2] /= bz
        if self.goal_in_state:
            state[-3] /= bx
            state[-2] /= by
            state[-1] /= bz
        return state.astype(np.float32)

    def seed(self, seed=None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset environment."""
        self.drone = Drone(0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                           DRONE_MASS_CENTER, DRONE_RADIUS, DRONE_MASS_ROTOR,
                           DRONE_LENGTH)
        self.camera.reset(self.drone.pos_i, self.drone.ori_v)
        self.t = 0
        self.shaping = None
        state = self.drone.state()
        self.target = np.random.uniform(-40., 0., (3,))
        self.target[:2] += 20.
        self.target[2] /= 4.
        self.target[2] -= TARGET_RADIUS
        if self.goal_in_state:
            target_v = self.target - state[:3]
            # target_v = self.drone.rotate_i_to_b(target_v)
            state = np.concatenate([state, target_v])
        return self._norm_state(state)

    def step(self, action):
        """Step environment."""
        action = action + np.random.normal(scale=ACTION_NOISE,
                                           size=action.shape)
        action = np.clip(action, -1.0, 1.0)
        force, torque = self._ac_to_force_torque(action)
        for _ in range(SIMS_PER_STEP):
            state = self.drone.step(torque, force, 1.0/FPS)
            self.camera.step(self.drone.pos_i, self.drone.ori_v)
        self.t += 1
        reward = 0.0
        if self.goal_in_state:
            target_v = self.target - state[:3]
            # target_v = self.drone.rotate_i_to_b(target_v)
            state = np.concatenate([state, target_v])
        dist_to_targ = np.sqrt(np.sum(state[-3:] ** 2))

        # shape reward if goal is to reach target, penalize if goal
        # is to stabilize
        # shaping for low velocity and staying upright and distance to target.
        shaping = (-np.sqrt((state ** 2)[3:6].sum())    # linear velocities
                   - np.sqrt((state ** 2)[9:12].sum())  # angular rates
                   - np.sqrt((state ** 2)[6:8].sum()))  # roll and pitch
        if self.goal_in_reward:
            reward -= 0.01 * dist_to_targ

        if self.shaping is None:
            self.shaping = shaping
        else:
            reward += shaping - self.shaping
            self.shaping = shaping

        done = False
        success = False
        crash = False
        timeout = False
        if self._out_of_bounds(state):
            done = True
            crash = True
            if self.goal_in_reward:
                reward -= 100.
            else:
                reward -= 10.
        elif self._at_target(state) and self.goal_in_reward:
            done = True
            success = True
            reward += 10.
        elif self.t >= MAX_NUM_STEPS:
            done = True
            timeout = True

        info = {'success': success, 'crash': crash, 'timeout': timeout}
        return self._norm_state(state), reward, done, info

    def render(self, mode='human'):
        """Render."""
        if self.goal_in_reward:
            self.renderer.set_target(self.target, TARGET_RADIUS)
        self.renderer.set_camera(self.camera.camera_pos,
                                 self.camera.camera_angle)
        self.renderer.draw_world(self.drone.pos_i, self.drone.ori_v,
                                 DRONE_RADIUS, DRONE_LENGTH, BOUNDS[:2])
        if mode == 'rgb_array':
            return self.renderer.get_pixels()

    def close(self):
        import pygame
        pygame.display.quit()
        self.renderer.initialized = False

    def state_dict(self):
        """Get env state."""
        return {'rng': self.np_random.get_state()}

    def load_state_dict(self, state_dict):
        """Load env state."""
        self.np_random.set_state(state_dict['rng'])


class DroneReacherBot(DroneEnv):
    """Reach the target."""

    goal_in_state = True
    goal_in_reward = True


class DroneReacherHuman(DroneEnv):
    """Reach the target."""

    goal_in_state = False
    goal_in_reward = True


class DroneStabilizer(DroneEnv):
    """Don't Crash."""

    goal_in_state = True
    goal_in_reward = False


if __name__ == '__main__':
    import time
    env = DroneStabilizer()
    done = False
    env.render()
    while not done:
        ac = env.action_space.sample()
        ac[0] = -1.0  # np.random.uniform(0.5, 1.0)
        s, r, done, _ = env.step(ac)
        env.render()
        time.sleep(0.02)
