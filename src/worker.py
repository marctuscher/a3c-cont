from src.networks.ac_network import AC_Network
from src.utils import update_target_graph
import tensorflow as tf
import numpy as np

class Worker():

    def __init__(self, env, worker_id, trainer, model_path, global_episodes, entropy_beta):
        self.env = env
        self.name = "worker_"+ str(worker_id)
        self.worker_id = worker_id
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.summary_writer = tf.summary.FileWriter("logs/train_" + str(self.worker_id))
        self.local_net = AC_Network(self.env, self.name, self.trainer, entropy_beta)

        self.update_local_ops = update_target_graph('global', self.name)

    def train(self, observations, rewards, actions, sess, gamma, bootstrap_value):
        self.summary_writer.add_graph(sess.graph)
        v_target = []
        v_s_ = bootstrap_value
        for r in reversed(rewards):
            v_s_ = r + gamma * v_s_
            v_target.append(v_s_)
        v_target.reverse()

        feed_dict = {self.local_net.target_v:v_target,
            self.local_net.inputs:observations,
            self.local_net.actions:actions,
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
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                obs_buffer = []
                reward_buffer = []
                actions_buffer = []

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
                    ob_, reward, done, info = self.env.step(a[0])
                    obs_buffer.append(ob)
                    reward_buffer.append(reward)
                    actions_buffer.append(a[0])

                    episode_values += v[0,0]
                    episode_reward += reward
                    ob = ob_
                    total_steps += 1
                    episode_step_count += 1

                    if len(obs_buffer) == 30 and not done and episode_count != max_episode_length - 1:
                        v_target = sess.run(self.local_net.v, {
                            self.local_net.inputs: [ob],
                            self.local_net.state_in[0]: rnn_state[0],
                            self.local_net.state_in[1]: rnn_state[1]
                        })
                        v_l, p_l, e_l, g_n, v_n = self.train(obs_buffer, reward_buffer, actions_buffer, sess, gamma, v_target[0, 0])
                        obs_buffer = []
                        reward_buffer = []
                        actions_buffer = []
                        sess.run(self.update_local_ops)
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(episode_values / episode_step_count)

                if len(obs_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(obs_buffer, reward_buffer, actions_buffer, sess, gamma, 0.0)

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
