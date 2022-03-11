from ddpg_torch_with_cnn import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('CarRacing-v1')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[96,96,3], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=3)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        print(act)
        print(type(act))
        print(act.shape)
        act = act.reshape((3))
        print(act.shape)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'CarRacing-v0-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)
