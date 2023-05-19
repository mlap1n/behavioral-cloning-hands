import numpy as np
import matplotlib.animation as animation
import matplotlib
import matplotlib.pyplot as plt
from dm_control import composer
from dm_control.composer import variation
from envs.arenas import BaseArena
from envs import cameras


class BaseTask(composer.Task):
    def __init__(self,
                 robot,
                 arena=None,
                 obs_settings=None,
                 workspace=None,
                 control_timestep=None,
                 cfg=None,
                 args=None):
        self.texture_path = None
        self.num_agents = 1
        self.duration = 30
        self.framerate = 30
        self.frames = []
        self.video = True

        self._robot = robot
        self._arena = BaseArena() if arena is None else arena
        self._robot_coord_init_pos = [-0.1, -0.9, 0.11]
        self._robot_quat_init_pos = [1, 0, 0, 0.75]
        self.num_substeps = 30

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        self._arena.attach(self._robot)

        # Configure and enable observables
        self._robot.observables.joints_torque.enabled = True
        self._robot.observables.joints_vel.enabled = True
        self._robot.observables.sensors_touch_fingertips.enabled = True
        self._robot.observables.sensors_touch_fingerpads.enabled = True
        self._robot.observables.sensors_accelerometer.enabled = True
        self._robot.observables.sensors_gyro.enabled = True
        self._robot.observables.egocentric_camera.enabled = True

        if control_timestep is None:
            self.control_timestep = self.num_substeps * self.physics_timestep
        else:
            self.control_timestep = control_timestep

        defined_cameras = [cameras.front_far,
                           cameras.front_close,
                           cameras.left_close,
                           cameras.left_far,
                           cameras.right_close,
                           cameras.right_far,]

        for i in defined_cameras:
            self._task_observables = cameras.add_camera_observables(
                self._arena,
                obs_settings,
                i,
            )

    @property
    def root_entity(self):
        return self._arena

    @property
    def robot(self):
        return self._robot

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)

        init_pos, quat = self.robot_init_pos(random_state)
        self._robot.set_pose(physics, position=init_pos, quaternion=quat)
        self._robot.rsi(physics, close_factors=random_state.uniform())

    def get_time(self) -> float:
        """
        Return the simulation time in seconds
        """
        return self.physics.data.time

    def get_world_state(self):
        gravity = np.expand_dims(self.physics.model.opt.gravity, axis=0)
        wind = np.expand_dims(self.physics.model.opt.wind, axis=0)
        magnetic = np.expand_dims(self.physics.model.opt.magnetic, axis=0)
        return np.concatenate((gravity, wind, magnetic), axis=1)

    def get_reward(self, physics):
        return 0.0

    def reset(self):
        '''
        '''
        self.physics.reset()
        self.task = self._sample_task()
        obs = self._get_obs()
        return obs

    def get_default_obs(self):
        pass

    def get_default_act(self):
        pass

    def robot_init_pos(self, random_state):
        pos = variation.evaluate(self._robot_coord_init_pos, random_state=random_state)
        quat = variation.evaluate(self._robot_quat_init_pos, random_state=random_state)
        return pos, quat

    def _is_done(self):
        pass

    def render(self, name_file="video.gif"):
        np.savetxt("logs/acc", self.acc)
        height, width, _ = self.frames[0].shape
        dpi = 70
        orig_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        matplotlib.use(orig_backend)
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        im = ax.imshow(self.frames[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        interval = 1000 / self.framerate
        anim = animation.FuncAnimation(fig=fig, func=update, frames=self.frames,
                                    interval=interval, blit=True, repeat=False)

        f = f"results/{name_file}"
        writergif = animation.PillowWriter(fps=self.framerate)
        anim.save(f, writer=writergif)

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def close(self):
        pass
