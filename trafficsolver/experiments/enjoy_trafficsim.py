"""
enjoy_trafficsim.py
Romain, Clément et Loïc
28/09/2019
"""

# Imports
import pickle
import torch
import time

import matplotlib.pyplot as plt
import numpy as np

from algorithm.dqn import DQN
from algorithm.mlp import MLP
from environments.trafficsim import TrafficSim

env = TrafficSim(3, 6, 6)

agent = DQN(env, MLP)

agent.load('network.pth')

obs = env.reset()
returns = 0
for i in range(10000):
    action = agent.predict(torch.FloatTensor(obs))
    obs, rew, done, info = env.step(action)
    env.render()
    time.sleep(1)
    returns += rew
    if done:
        obs = env.reset()
        print("Episode score: ", returns)
        returns = 0
