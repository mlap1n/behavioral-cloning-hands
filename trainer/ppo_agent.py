import tensorflow as tf
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.train.utils import spec_utils
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.agents import PPOClipAgent


class ImageLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape=(84, 84, 3)):
        super(ImageLayer, self).__init__()
        self.reshape = tf.keras.layers.Reshape(input_shape)
        self.rescale = tf.keras.layers.Rescaling(scale=1/255.0)
        self.cv_model = tf.keras.applications.ResNet50(include_top=False,
                                                       weights="imagenet")
        self.cv_model.trainable = False
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(64)

    def call(self, x):
        x = self.reshape(x)
        x = self.rescale(x)
        x = self.cv_model(x)
        x = self.avg_pool(x)
        x = self.linear(x)
        return x


def make_networks(env,
                  strategy,
                  actor_net_layer,
                  value_net_layer,
                  lstm_size,
                  dropout_layer_params,
                  use_cnn):
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    if use_cnn:
        with strategy.scope():
            preprocessing_layers = {
                'MuJoCo Model/egocentric_camera': tf.keras.models.Sequential([
                    ImageLayer(),
                    tf.keras.layers.Flatten(),
                ]),
                'MuJoCo Model/joints_torque': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'MuJoCo Model/joints_vel': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'MuJoCo Model/sensors_touch_fingerpads': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'MuJoCo Model/sensors_touch_fingertips': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'unnamed_model/angular_velocity': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'unnamed_model/linear_velocity': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'unnamed_model/orientation': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
                'unnamed_model/position': tf.keras.models.Sequential([tf.keras.layers.Flatten()]),
            }

            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

            actor_net = ActorDistributionRnnNetwork(obs_spec,
                                                    act_spec,
                                                    preprocessing_layers=preprocessing_layers,
                                                    preprocessing_combiner=preprocessing_combiner,
                                                    input_fc_layer_params=actor_net_layer,
                                                    lstm_size=lstm_size,
                                                    output_fc_layer_params=(128,),
            )
            value_net = ValueRnnNetwork(obs_spec,
                                        preprocessing_layers=preprocessing_layers,
                                        preprocessing_combiner=preprocessing_combiner,
                                        input_fc_layer_params=value_net_layer,
                                        lstm_size=lstm_size,
                                        output_fc_layer_params=(128,),
            )
    else:
        actor_net = ActorDistributionNetwork(obs_spec,
                                             act_spec,
                                             fc_layer_params=actor_net_layer,
        )
        value_net = ValueNetwork(obs_spec,
                                 fc_layer_params=value_net_layer,
        )
    return actor_net, value_net


def make_agent(env,
               strategy,
               actor_net,
               critic_net,
               lr=4e-4):
    obs_spec, act_spec, ts_spec = spec_utils.get_tensor_specs(env)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    with strategy.scope():
        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent = PPOClipAgent(ts_spec,
                            act_spec,
                            optimizer=optimizer,
                            actor_net=actor_net,
                            value_net=critic_net,
                            train_step_counter=global_step,
                            entropy_regularization=1e-2,
                            importance_ratio_clipping=0.2,
        )
    return agent
