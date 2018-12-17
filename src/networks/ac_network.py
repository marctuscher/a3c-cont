import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp



class AC_Network():
    def __init__(self, env, scope, trainer, entropy_beta):
        self.env = env
        self.scope = scope
        self.trainer = trainer


        with tf.variable_scope(self.scope):
            self.actions = tf.placeholder(tf.float32, [None, *self.env.action_space.shape])
            self.advantages = tf.placeholder(tf.float32, [None])
            self.rewards = tf.placeholder(tf.float32, [None])
            self.inputs = tf.placeholder(tf.float32, [None, *self.env.observation_space.shape])
            lstm_cell = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)

            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            hidden = slim.fully_connected(self.inputs ,256, activation_fn=tf.nn.relu)
            rnn_in = tf.expand_dims(hidden, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in
            )
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            with tf.variable_scope('actor'):
                hidden_policy_layer = slim.fully_connected(rnn_out, 128, tf.nn.relu)
                mu = slim.fully_connected(hidden_policy_layer, self.env.action_space.shape[0], tf.nn.tanh)
                sigma = slim.fully_connected(hidden_policy_layer, self.env.action_space.shape[0], tf.nn.softplus)

            with tf.variable_scope('critic'):
                hidden_value_layer = slim.fully_connected(rnn_out, 128, tf.nn.relu)
                self.v = slim.fully_connected(hidden_value_layer, 1, activation_fn=None)

            normal_dist = tfp.distributions.MultivariateNormalDiag(mu, sigma)
            self.a = tf.clip_by_value(tf.squeeze(normal_dist.sample(1)), self.env.action_space.low, self.env.action_space.high)
            if self.scope != "global":
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.rewards - tf.squeeze(self.v)))

                log_prob = normal_dist.log_prob(self.actions)
                self.entropy = -normal_dist.entropy()
                self.policy_loss = -tf.reduce_sum(log_prob * self.advantages)
                self.loss = self.value_loss + self.policy_loss + self.entropy * entropy_beta

                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 20.0)
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
