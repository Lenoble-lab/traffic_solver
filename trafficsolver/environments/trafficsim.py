"""
trafficsim.py
Romain, Clément et Loïc
28/09/2019
"""

# Imports
import numpy as np
import environments.rendering as rendering
import copy


class NetworkStructure:

    def __init__(self, n_inter, h_length, v_length):

        self.n_inter = n_inter
        self.h_length = h_length
        self.v_length = v_length

        self.h_lights = np.zeros(self.n_inter)
        self.v_lights = np.zeros(self.n_inter)

        self.h_cars = np.zeros((self.n_inter, self.h_length))
        self.v_cars = np.zeros((self.n_inter, self.v_length))


class TrafficSim:

    def __init__(self, network):

        assert isinstance(network, NetworkStructure)
        self.network = network

        class ActionSpace:
            def __init__(self, n_actions):
                self.n = n_actions

        class ObservationSpace:
            def __init__(self, n_features):
                self.shape = [n_features]

        self.action_space = ActionSpace(4 * self.network.n_inter)
        self.observation_space = ObservationSpace(self.network.n_inter * (self.network.h_length + self.network.v_length + 2))

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
