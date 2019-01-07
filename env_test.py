import gym
import matplotlib.pyplot as plt


env = gym.make('Festium-v3')
for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
            
plt.imshow(observation[:,:,0], cmap='gray')
plt.show()