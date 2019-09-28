"""
train_trafficsim.py
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

nb_steps = 20000

agent = DQN(env,
            MLP,
            replay_start_size=1000,
            replay_buffer_size=50000,
            gamma=0.99,
            update_target_frequency=500,
            minibatch_size=32,
            learning_rate=1e-3,
            initial_exploration_rate=1,
            final_exploration_rate=0.02,
            final_exploration_step=5000,
            adam_epsilon=1e-8,
            update_frequency=4,
            logging=True)


agent.learn(timesteps=nb_steps, verbose=True)
agent.save()

logdata = pickle.load(open("log_data.pkl", 'rb'))
scores = np.array(logdata['Episode_score'])
plt.plot(scores[:, 1], scores[:, 0])
plt.show()
