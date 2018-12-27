from src.networks.ac_network import AC_Network
from src.utils import update_target_graph, discount
import tensorflow as tf
import numpy as np


class Worker():

    def __init__(self, env, worker_id, trainer, model_path, global_episodes,max_global_steps, entropy_beta, value_coeff):
        self.env = env
        self.name = "worker_"+ str(worker_id)
        self.worker_id = worker_id
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.max_global_steps = max_global_steps
        self.increment = self.global_episodes.assign_add(1)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.summary_writer = tf.summary.FileWriter("logs/train_" + str(self.worker_id))
        self.local_net = AC_Network(self.env, self.name, self.model_path, self.trainer, entropy_beta, value_coeff)

        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, observations, rewards, actions, values, sess, gamma, bootstrap_value):


        rewards_plus = np.asarray(rewards + [bootstrap_value])
        discounted_rewards = discount(rewards_plus, gamma)[:-1]
        value_plus = np.asarray(values + [bootstrap_value])
        advs = np.array(rewards) + gamma * value_plus[1:] - value_plus[:-1]
        advs = discount(advs, gamma)

        feed_dict = {self.local_net.advantages:advs,
            self.local_net.inputs:observations,
            self.local_net.actions:actions,
                     self.local_net.rewards: discounted_rewards,
            self.local_net.state_in[0]:self.batch_rnn_state[0],
            self.local_net.state_in[1]:self.batch_rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_net.value_loss,
                                                                self.local_net.policy_loss,
            self.local_net.entropy,
            self.local_net.grad_norms,
            self.local_net.var_norms,
            self.local_net.state_out,
            self.local_net.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(observations),p_l / len(observations),e_l / len(observations), g_n,v_n


    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop() and episode_count != self.max_global_steps:
                sess.run(self.update_local_ops)
                obs_buffer = []
                reward_buffer = []
                actions_buffer = []
                values_buffer = []

                episode_values = 0
                episode_obs = []
                episode_reward = 0
                episode_step_count = 0
                done = False
                ob = self.env.reset()
                episode_obs.append(ob)
                rnn_state = self.local_net.state_init
                self.batch_rnn_state = rnn_state
                while not done:
                    a, v, rnn_state = sess.run([self.local_net.a, self.local_net.v, self.local_net.state_out], {
                        self.local_net.inputs : [ob],
                        self.local_net.state_in[0]: rnn_state[0],
                        self.local_net.state_in[1]: rnn_state[1]
                    })
                    ob_, reward, done, info = self.env.step(a)
                    obs_buffer.append(ob)
                    values_buffer.append(v[0, 0])
                    reward_buffer.append(reward)
                    actions_buffer.append(a)
                    episode_values += v[0, 0]
                    episode_reward += reward
                    ob = ob_
                    total_steps += 1
                    episode_step_count += 1

                    if len(obs_buffer) == 30 and not done and episode_count != max_episode_length - 1:
                        v1 = sess.run(self.local_net.v, {self.local_net.inputs: [ob], self.local_net.state_in[0]: rnn_state[0], self.local_net.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(obs_buffer, reward_buffer, actions_buffer, values_buffer, sess, gamma, v1[0, 0])
                        obs_buffer = []
                        reward_buffer = []
                        values_buffer = []
                        actions_buffer = []
                        sess.run(self.update_local_ops)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(episode_values / episode_step_count)

                if len(obs_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(obs_buffer, reward_buffer, actions_buffer, values_buffer,sess, gamma, 0.0)

                if episode_count % 5 == 0 and episode_count != 0:
                    # not using np.mean would be faster
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
