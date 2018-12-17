import numpy as np
from src.networks.ac_network import AC_Network
import tensorflow as tf
import gym



with tf.device("/cpu:0"):
    test_model_path = 'net/a3c.ckpt'
    env = gym.make("Reacher-v2")

    class ReplayGuy():

        def __init__(self):
            self.env = env.unwrapped
            self.local_net = AC_Network(env, 'global', None, None)
            self.sess = tf.Session()
            tf.global_variables_initializer().run(session=self.sess)
            self.local_net.load_ckpt(self.sess)
 


        def play(self):

            ob = self.env.reset()
            rnn_state = self.local_net.state_init
            while True:
                self.env.render()
                a, v, rnn_state = self.sess.run([self.local_net.a, self.local_net.v, self.local_net.state_out], {
                        self.local_net.inputs : [ob],
                        self.local_net.state_in[0]: rnn_state[0],
                        self.local_net.state_in[1]: rnn_state[1]
                    })
                ob, reward, done, info = self.env.step(a)
                if done:
                    rnn_state = self.local_net.state_init
                    ob = env.reset()
                    

    guy = ReplayGuy()

    guy.play()
