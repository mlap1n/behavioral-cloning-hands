import os
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf
import tf_agents
from tf_agents.utils import common
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.metrics import tf_metrics
from tf_agents.train.utils import strategy_utils
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.policies import policy_saver

from trainer.ppo_agent import make_agent, make_networks


def get_eval_metrics():
    eval_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]
    return eval_metrics

def get_step_metrics():
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
    ]
    return step_metrics

def get_train_metrics(batch_size):
    train_metrics = [
        tf_metrics.AverageReturnMetric(batch_size=batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=batch_size),
    ]
    return train_metrics


def save_video(frames, framerate=30, name_file="video.gif"):
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
    FFwriter = animation.FFMpegWriter(fps=framerate)
    #writergif = animation.PillowWriter(fps=framerate)
    anim.save(f, writer=FFwriter)


class PPOTrainer:
    """
    Trainer class for PPO to update policies.
    """
    def __init__(self, env, eval_env, args=None, cfg=None):
        self.name_task = "reach_site_vision"
        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)
        with self.strategy.scope():
            self.actor_net, self.critic_net = make_networks(env=env,
                                                            strategy=self.strategy,
                                                            actor_net_layer=(512, 128),
                                                            value_net_layer=(256, 128),
                                                            lstm_size=(256,),
                                                            dropout_layer_params=(0.1,0.1),
                                                            use_cnn=True)
            self.agent = make_agent(actor_net=self.actor_net,
                                    critic_net=self.critic_net,
                                    env=env,
                                    strategy=self.strategy)
            self.agent.initialize()

        self.env = env
        self.eval_env = eval_env

        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy

        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.replay_buffer_capacity = 2000
        self.collect_episodes_per_iteration = 4
        self.max_checkpoints = 10
        self.checkpoints_dir = "./checkpoints"
        self.tb_log_dir = "./tensorboard"
        self.iterations = 10_000
        self.train_log_interval = 200
        self.eval_interval = 400
        self.num_eval_episodes = 50
        self.save_interval = 200
        self.print_info = True

        if self.print_info:
            print("Observation spec: {} \n".format(self.env.observation_spec()))
            print("Action spec: {} \n".format(self.env.action_spec()))
            print("policy: {} \n".format(self.agent.policy))

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=self.replay_buffer_capacity)

        self.eval_metrics = get_eval_metrics()
        self.step_metrics = get_step_metrics()
        self.train_metrics = get_train_metrics(self.env.batch_size)
        self.all_train_metrics = self.step_metrics + self.train_metrics

        self.collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.all_train_metrics,
            num_episodes=self.collect_episodes_per_iteration
        )

        self.agent.train = common.function(self.agent.train, autograph=True)

        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoints_dir,
            max_to_keep=self.max_checkpoints,
            agent=self.agent,
            policy=self.eval_policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step,
            metrics=metric_utils.MetricsGroup(self.all_train_metrics, 'train_metrics')
        )
        self.tf_policy_saver = policy_saver.PolicySaver(self.eval_policy, train_step=self.global_step)

        if self.tb_log_dir is not None:
            train_log_dir = os.path.join(self.tb_log_dir, "train")
            eval_log_dir = os.path.join(self.tb_log_dir, "eval")
            self.summary_writer = tf.summary.create_file_writer(train_log_dir, flush_millis=1000)
            self.eval_summary_writer = tf.summary.create_file_writer(eval_log_dir, flush_millis=1000)

        if os.path.exists(self.checkpoints_dir) and os.path.isdir(self.checkpoints_dir):
            if not os.listdir(self.checkpoints_dir):
                time_step = env.reset()
            else:
                self.train_checkpointer.initialize_or_restore()
        else:
            print("\'{}\' - doesn't exists or is't a directory".format(self.checkpoints_dir))

    def eval(self):
        print("start eval mode...")
        avg_return = tf_agents.eval.metric_utils.eager_compute(
            metrics=self.eval_metrics,
            environment=self.eval_env,
            policy=self.eval_policy,
            num_episodes=self.num_eval_episodes,
            train_step=self.global_step,
            use_function=False,
        )

        with self.eval_summary_writer.as_default():
            tf.summary.scalar("average eval reward",
                              float(avg_return["AverageReturn"].numpy()),
                              step=self.global_step.numpy())
        if self.print_info:
            print('================================================')
            print(f'|step = {self.global_step.numpy()}| average eval reward = {avg_return["AverageReturn"].numpy()}|')
            print('================================================')

    def make_video(self):
        print("video recording...")
        time_step = self.eval_env.reset()
        state = self.eval_policy.get_initial_state(self.eval_env.batch_size)
        frames = []
        while not time_step.is_last():
            policy_step = self.eval_policy.action(time_step, state)
            state = policy_step.state
            time_step = self.eval_env.step(policy_step.action)
            frames.append([self.eval_env.render()[0]])
        all_frames = np.concatenate(frames, axis=0)
        name = self.name_task + "_" + str(self.global_step.numpy()) + ".mp4"
        save_video(all_frames, 30, name)
        print(f"file saved: results/{name}")
        print("done!")

    def train(self):
        self.agent.train_step_counter.assign(0)
        with self.strategy.scope():
            print("start training agent...")
            for i in range(self.iterations):
                self.collect_driver.run()

                trajectories = self.replay_buffer.gather_all()
                train_loss = self.agent.train(experience=trajectories)

                self.replay_buffer.clear()
                step = self.global_step.numpy()

                with self.summary_writer.as_default():
                    for train_metric in self.train_metrics:
                        train_metric.tf_summaries(train_step=step, step_metrics=self.step_metrics)

                if step % self.train_log_interval == 0:
                    if self.print_info:
                        print(f'step = {step} | train loss = {train_loss.loss}|')

                if step % self.eval_interval == 0 and step != 0:
                    self.eval()
                    self.make_video()

                if step % self.save_interval == 0 and step != 0:
                    self.train_checkpointer.save(global_step=step)
                    policy_dir = os.path.join(self.checkpoints_dir, 'saved_model', 'policy_' + ('%d' % step).zfill(9))
                    self.tf_policy_saver.save(policy_dir)
                    print("save step: {}".format(step))

        print("================")
        print("training is done")
        print("================")
