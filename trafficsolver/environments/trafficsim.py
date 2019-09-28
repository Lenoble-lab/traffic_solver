"""
trafficsim.py
Romain, Clément et Loïc
28/09/2019
"""

# Imports
import numpy as np
import environments.rendering as rendering
import copy


class TrafficSim:

    def __init__(self):

        class ActionSpace:
            def __init__(self, n_actions):
                self.n = n_actions

        class ObservationSpace:
            def __init__(self, n_features):
                self.shape = [n_features]

        self.action_space = ActionSpace(2)
        self.observation_space = ObservationSpace(4)

        self.viewer = None

    def step(self, action):

        return state, reward, done, {}

    def reset(self):

        return state

    def seed(self, seed):
        return

    def render(self):
        return self.viewer.render(return_rgb_array = False)
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
