import numpy as np
import os
import tensorflow as tf
import gym
import multiprocessing
from src.networks.ac_network import AC_Network
from src.worker import Worker
import threading
from time import sleep
import shutil
from gym.envs.box2d import LunarLanderContinuous

max_global_steps = 1500000
max_episode_length = 20
gamma = .99

entropy_beta = 0.005
model_path = './net/a3c.ckpt'
output_graph = True
resume = True
graph_dir = './graph_log'
env = "Reacher-v2"
environment = gym.make(env)
tf.reset_default_graph()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_net = AC_Network(env, 'global',model_path, None, None)
    num_workers = multiprocessing.cpu_count()
    workers = []
    for i in range(num_workers):
        workers.append(Worker(env, i, trainer, model_path, global_episodes, max_global_steps,entropy_beta))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    try:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        
        if resume == True:
            try:
                print('Loading Model...')
                saver.restore(sess,model_path)
                print('Model loaded... \nContinue Training...')
            except:
                print('Couldnt find existing Model... \nStarting New Training-Process...')

        if output_graph:
            if os.path.exists(graph_dir):
                shutil.rmtree(graph_dir)
            tf.summary.FileWriter(graph_dir, sess.graph)  

        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.1)
            worker_threads.append(t)
        coord.join(worker_threads)
    except:
        print("\nManual Training Exit - Try to save model")
    master_net.save_ckpt(sess,saver)
