import gym
import random
env = gym.make('Pendulum-v1')
observation = env.reset()

# Observation and action space 
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))
print(env.observation_space.shape)
print(env.action_space.shape)

for _ in range(10):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  print(action)
  observation, reward, done, info = env.step(action)
  print(observation)

  if done:
    observation = env.reset()
env.close()