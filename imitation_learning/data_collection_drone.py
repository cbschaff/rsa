"""Collect human data from DroneReacher."""
import gym
import residual_shared_autonomy.drone_sim
from residual_shared_autonomy.drone_sim.joystick_agent import DroneJoystickActor
from dl.rl import ensure_vec_env
import torch
import time
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data collection.')
    parser.add_argument('datafile', type=str, help='datafile')
    parser.add_argument('neps', type=int, help='# of episodes')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite data')
    args = parser.parse_args()

    env = gym.make("DroneReacherBot-v0")
    env = ensure_vec_env(env)

    if os.path.exists(args.datafile):
        if not args.overwrite:
            raise ValueError('Data file {} already exists. Add --overwrite to '
                             ' overwrite the file.'.format(args.datafile))

    actor = DroneJoystickActor(env)

    transitions = []
    for i in range(args.neps):
        print(i)
        ob = env.reset()
        actor.reset()
        env.render()
        done = False
        time.sleep(1.)

        while not done:
            action = actor(ob)
            ob_next, reward, done, _ = env.step(action)
            transitions.append([ob, action.copy(), reward, done])
            ob = ob_next
        time.sleep(0.5)

    obs, actions, rewards, dones = zip(*transitions)
    data = {'obs': torch.Tensor(obs).squeeze(),
            'actions': torch.Tensor(actions).squeeze(),
            'rewards': torch.Tensor(rewards).squeeze(),
            'dones': torch.Tensor(dones).squeeze()}

    torch.save(data, args.datafile)
