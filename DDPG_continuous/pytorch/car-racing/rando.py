import gym
import random
import cv2
env = gym.make('CarRacing-v1')
observation = env.reset()

# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))
print(env.observation_space.shape)
print(env.action_space.shape)

for _ in range(100):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  #print(action)
  observation, reward, done, info = env.step(action)
  #print(observation)

  # cv2.imwrite('color_img.jpg', observation)
  # cv2.imshow("image", observation)
  # cv2.waitKey()

  if done:
    observation = env.reset()
env.close()