import unittest
import matplotlib.animation as animation
import matplotlib
import matplotlib.pyplot as plt

import sys

import numpy as np

from dm_control import composer
from dm_control import viewer

from tasks.utils import (reach_site_vision,
                         reach_prop_vision,
                         lift_brick_vision,
                         lift_large_box_vision,
                         place_brick_vision,
                         )


def display_video(frames, framerate=30, name_file="video.gif"):
    print(frames[0].shape)
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    f = f"results/{name_file}"
    writergif = animation.PillowWriter(fps=framerate)
    anim.save(f, writer=writergif)


class CheckAllEnvs(unittest.TestCase):
    def setUp(self):
        self.seed = 50

    def test_check_display(self):
        task = reach_site_vision()
        random_state = np.random.RandomState(self.seed)
        env = composer.Environment(task, random_state=random_state)
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()

        print("action:")
        print(action_spec, sep="\n")

        print("obs:")
        print(obs_spec, sep="\n")

        def sample_random_action():
            return env.random_state.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype, copy=False)

        frames = []
        timestep = env.reset()
        for _ in range(60):
            timestep = env.step(sample_random_action())
            frames.append([env.physics.render(height=480, width=640)])
        all_frames = np.concatenate(frames, axis=0)
        display_video(all_frames, 30)

    def test_check_launcher(self):
        task = reach_site_vision()
        random_state = np.random.RandomState(self.seed)
        env = composer.Environment(task, random_state=random_state)
        viewer.launch(env)

if __name__ == "__main__":
    unittest.main()
