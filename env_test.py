import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim




env = gym.make('Festium-v2')
env.reset()



while True:
     env.render()
     obs , rew , done, info = env.step(env.action_space.sample())
     #print(obs[0]),print()