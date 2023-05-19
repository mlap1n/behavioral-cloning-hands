from typing import Callable, Optional
import argparse
import warnings

from tf_agents.environments import tf_py_environment
from tf_agents.environments.dm_control_wrapper import DmControlWrapper
from tf_agents.environments import wrappers
from dm_control import composer

from utility import options
from trainer.ppo_trainer import PPOTrainer
from tasks.utils import (reach_site_vision,
                         reach_prop_vision,
                         lift_brick_vision,
                         lift_large_box_vision,
                         place_brick_vision,
                         )

warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_env():
    tasks = [reach_site_vision(),
            reach_prop_vision(),
            lift_brick_vision(),
            lift_large_box_vision(),
             place_brick_vision()]
    for task in tasks:
        env = composer.Environment(task)
        env = DmControlWrapper(env)
        env = wrappers.TimeLimit(env, duration=300)
        yield env


def main(modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None):
    parser = options.get_train_parser()
    args = options.parse_args(parser, modify_parser=modify_parser)

    task = reach_site_vision()

    train_env = composer.Environment(task)
    eval_env = composer.Environment(task)

    train_env = DmControlWrapper(train_env)
    eval_env = DmControlWrapper(eval_env)

    train_env = wrappers.TimeLimit(train_env, duration=300)
    eval_env = wrappers.TimeLimit(eval_env, duration=400)

    #train_env = tf_agents.environments.ParallelPyEnvironment(create_env)

    train_env = tf_py_environment.TFPyEnvironment(train_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env)

    trainer = PPOTrainer(args=args,
                         env=train_env,
                         eval_env=eval_env)

    trainer.train()


if __name__ == "__main__":
    main()
