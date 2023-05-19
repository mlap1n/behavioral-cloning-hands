from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer

import numpy as np

from tasks.utils import (reach_site_vision,
                         reach_prop_vision,
                         lift_brick_vision,
                         lift_large_box_vision,
                         place_brick_vision,
                         )

_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


def create_env():
    tasks = [reach_site_vision(),
             reach_prop_vision(),
             lift_brick_vision(),
             lift_large_box_vision(),
             place_brick_vision()]
    for task in tasks:
        env = composer.Environment(task)
        yield env

class RobotTest(parameterized.TestCase):
    """Tests run on all the tasks."""
    def _validate_observation(self, observation, observation_spec):
        self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
        for name, array_spec in observation_spec.items():
            array_spec.validate(observation[name])

    def _validate_reward_range(self, reward):
        self.assertIsInstance(reward, float)
        self.assertBetween(reward, 0, 1)

    def _validate_discount(self, discount):
        self.assertIsInstance(discount, float)
        self.assertBetween(discount, 0, 1)

    def test_task_runs(self, task_name=None):
        seed = 666
        tasks = [reach_site_vision,
                 reach_prop_vision,
                 lift_brick_vision,
                 lift_large_box_vision,
                 place_brick_vision]

        for task in tasks:
            env = composer.Environment(task())
            random_state = np.random.RandomState(seed)

            observation_spec = env.observation_spec()
            action_spec = env.action_spec()
            self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
            self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

            # Run a partial episode, check observations, rewards, discount.
            for _ in range(_NUM_EPISODES):
                time_step = env.reset()
                for _ in range(_NUM_STEPS_PER_EPISODE):
                    self._validate_observation(time_step.observation, observation_spec)
                    if time_step.first():
                        self.assertIsNone(time_step.reward)
                        self.assertIsNone(time_step.discount)
                    else:
                        self._validate_reward_range(time_step.reward)
                        self._validate_discount(time_step.discount)
                    action = random_state.uniform(action_spec.minimum, action_spec.maximum)
                    env.step(action)

if __name__ == '__main__':
    absltest.main()
