import gym
import numpy as np
from ddpg_tf2 import Agent
import time

env = gym.make('Pendulum-v1')
agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
n_steps = 0
while n_steps <= agent.batch_size:
    observation = env.reset()
    action = env.action_space.sample()
    observation_, reward, done, info = env.step(action)
    agent.remember(observation, action, reward, observation_, done)
    n_steps += 1
agent.load_models()

observation = env.reset()
evaluate = False
done = False
score = 0
while not done:
    action = agent.choose_action(observation, evaluate)
    observation_, reward, done, info = env.step(action)
    score += reward
    agent.remember(observation, action, reward, observation_, done)
    observation = observation_
    env.render()
    time.sleep(0.01)

print('score %.2f' % score)
