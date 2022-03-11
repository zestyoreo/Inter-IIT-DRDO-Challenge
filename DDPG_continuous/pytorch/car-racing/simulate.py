from ddpg_torch import Agent
import gym
import numpy as np
import time
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=2)

agent.load_models()
np.random.seed(0)

score_history = []
obs = env.reset()
done = False
score = 0
while not done:
    act = agent.choose_action(obs)
    new_state, reward, done, info = env.step(act)
    score += reward
    obs = new_state
    env.render()
    time.sleep(0.01)

print('score %.2f' % score)
