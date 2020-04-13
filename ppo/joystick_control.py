"""Render trained agents."""
import dl
import argparse
from residual_shared_autonomy.ppo import ConstrainedResidualPPO
from residual_shared_autonomy.lunar_lander import LunarLanderJoystickActor
from residual_shared_autonomy.drone_sim import DroneJoystickActor
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Script.')
    parser.add_argument('logdir', type=str, help='logdir')
    parser.add_argument('--drone', action='store_true', help='conrol drone env')
    parser.add_argument('--reacher', action='store_true', help='conrol luanr reacher env')
    args = parser.parse_args()

    if args.drone:
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="DroneReacherBot-v0"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=DroneJoystickActor)
    elif args.reacher:
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="LunarLanderReacher-v2"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=LunarLanderJoystickActor)
    else:
        dl.load_config(os.path.join(args.logdir, 'config.gin'),
                       ['make_env.env_id="LunarLanderRandomContinuous-v2"'])
        trainer = ConstrainedResidualPPO(args.logdir, nenv=1,
                                         base_actor_cls=LunarLanderJoystickActor)
    trainer.load()
    trainer.evaluate()
    trainer.close()
