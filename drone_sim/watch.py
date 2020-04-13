"""Render trained agents."""
import dl
import argparse
from dl.rl import rl_record, misc, PPO, VecFrameStack
import residual_shared_autonomy.drone_sim
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script.')
    parser.add_argument('logdir', type=str, help='logdir')
    args = parser.parse_args()

    dl.load_config(os.path.join(args.logdir, 'config.gin'))

    t = PPO(args.logdir, nenv=1)
    t.load()
    t.pi.eval()
    env = t.env
    misc.set_env_to_eval_mode(env)
    os.makedirs(os.path.join(t.logdir, 'video'), exist_ok=True)
    outfile = os.path.join(t.logdir, 'video',
                           t.ckptr.format.format(t.t) + '.mp4')
    rl_record(env, t.pi, 10, outfile, t.device)
    t.close()
