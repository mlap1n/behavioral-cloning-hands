import os
import pickle

import numpy as np

import tensorflow as tf
from tf_agents.policies.policy_saver import PolicySaver


class Loader:
    def __init__(self, dir=None, name="", time_ext="", save_interval=1, custom_path=None):
        dir = os.path.split(__file__)[0] if dir is None else dir
        self.custom_path = custom_path

        self.policy_save_dir = os.path.join(dir, "models", name.format(time_ext))
        self.save_interval = save_interval

        if not os.path.exists(self.policy_save_dir):
            print("Directory {} does not exist;".format(self.policy_save_dir),
                  "Creating it now...")
            os.makedirs(self.policy_save_dir, exist_ok=True)

        if self.use_separate_agents:
            # Get train and evaluation policies
            self.train_savers = [PolicySaver(self.collect_policies[i],
                                             batch_size=None) for i in
                                 range(self.num_agents)]
            self.eval_savers = [PolicySaver(self.eval_policies[i],
                                            batch_size=None) for i in
                                range(self.num_agents)]

        else:
            # Get train and evaluation policy savers
            self.train_saver = PolicySaver(self.collect_policy, batch_size=None)
            self.eval_saver = PolicySaver(self.eval_policy, batch_size=None)

    def save_policies(self, epochs_done=0, is_final=False):
        # If final, just use 'FINAL'
        if is_final:
            epochs_done = "FINAL"

        # Iterate through training policies and save each of them
        for i, train_saver in enumerate(self.train_savers):
            if self.custom_path is None:
                train_save_dir = os.path.join(self.policy_save_dir, "train",
                                                "epochs_{}".format(epochs_done),
                                                "agent_{}".format(i))
            else:
                train_save_dir = os.path.join(self.policy_save_dir, "train",
                                                "epochs_{}".format(
                                                    self.custom_path),
                                                "agent_{}".format(i))
            if not os.path.exists(train_save_dir):
                os.makedirs(train_save_dir, exist_ok=True)
            train_saver.save(train_save_dir)

        print("Training policies saved...")

        # Iterate through eval policies
        for i, eval_saver in enumerate(self.eval_savers):
            eval_save_dir = os.path.join(self.policy_save_dir, "eval",
                                            "epochs_{}".format(epochs_done),
                                            "agent_{}".format(i))
            if not os.path.exists(eval_save_dir):
                os.makedirs(eval_save_dir, exist_ok=True)
            eval_saver.save(eval_save_dir)

        print("Eval policies saved...")

        # Save parameters in a file
        agent_params = {'normalize_obs': self.train_env.normalize,
                        'use_lstm': self.use_lstm,
                        'frame_stack': self.use_multiple_frames,
                        'num_frame_stack': self.env.num_frame_stack,
                        'obs_size': self.size}

        # Save as pkl parameter file
        params_path = os.path.join(self.policy_save_dir, "parameters.pkl")
        with open(params_path, "w") as pkl_file:
            pickle.dump(agent_params, pkl_file)
        pkl_file.close()

    def load_policies(self, eval_model_path=None, train_model_path=None):
        # Load evaluation and/or training policies from path
        if eval_model_path is not None:
            self.eval_policy = tf.saved_model.load(eval_model_path)
            print("Loading evaluation policy from: {}".format(eval_model_path))

        if train_model_path is not None:
            self.collect_policy = tf.saved_model.load(train_model_path)
            print("Loading training policy from: {}".format(train_model_path))


def normalize(val, current_min_val, current_max_val, new_min_val, new_max_val, clip=False):
    val = np.array(val, dtype=float)
    current_min_val = np.array(current_min_val, dtype=float)
    current_max_val = np.array(current_max_val, dtype=float)
    new_min_val = np.array(new_min_val, dtype=float)
    new_max_val = np.array(new_max_val, dtype=float)

    if clip:
        return np.clip((new_max_val - new_min_val) / (current_max_val - current_min_val) * (val - current_max_val) +
                       new_max_val, new_min_val, new_max_val)
    else:
        return (new_max_val - new_min_val) / (current_max_val - current_min_val) * (val - current_max_val) + new_max_val
