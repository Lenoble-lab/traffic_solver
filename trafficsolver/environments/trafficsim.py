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

    def __init__(self, n_inter, h_length, v_length):

        self.n_inter = n_inter
        self.h_length = h_length
        self.v_length = v_length
        self.network = NetworkStructure(n_inter, h_length, v_length)

        class ActionSpace:
            def __init__(self, n_actions):
                self.n = n_actions

        class ObservationSpace:
            def __init__(self, n_features):
                self.shape = [n_features]

        self.action_space = ActionSpace(4 ** self.network.n_inter)
        self.observation_space = ObservationSpace(self.network.n_inter * (self.network.h_length + self.network.v_length + 2))

        self.MAX_TIMESTEPS = 50
        self.current_timestep = 0

        self.TRAFFIC_INTENSITY = 0.2

        self.viewer = None

    def step(self, action):

        done = False
        self.current_timestep += 1
        reward = 0

        # check if time is up
        if self.current_timestep >= self.MAX_TIMESTEPS:
            done = True

        # change traffic lights according to action
        for i in range(self.network.n_inter):
            current_action = action % 4
            if current_action == 0:
                self.network.h_lights[i] = 0
                self.network.v_lights[i] = 0
            elif current_action == 1:
                self.network.h_lights[i] = 1
                self.network.v_lights[i] = 0
            elif current_action == 2:
                self.network.h_lights[i] = 0
                self.network.v_lights[i] = 1
            else:
                self.network.h_lights[i] = 1
                self.network.v_lights[i] = 1
            action /= 4

        # move cars
        for i in reversed(range(self.network.n_inter)):

            # move horizontal cars
            for j in reversed(range(self.network.h_length)):
                # last car in interval
                if j == (self.network.h_length - 1) and self.network.h_cars[i, j] == 1:
                    self.network.h_cars[i, j] = 0
                    # reward if car leaves network
                    if i == (self.network.n_inter - 1):
                        reward += 1
                    # if not last interval, go to next
                    if i < (self.network.n_inter - 1):
                        self.network.h_cars[i+1, 0] = 1
                # car before last moves only if light is green
                elif j == (self.network.h_length - 2) and self.network.h_lights[i] == 1:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1
                else:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1

            # move vertical cars
            for j in reversed(range(self.network.v_length)):
                # last car in interval
                if j == (self.network.v_length - 1) and self.network.v_cars[i, j] == 1:
                    self.network.v_cars[i, j] = 0
                    # reward if car leaves network
                    reward += 1
                # car before last moves only if light is green
                elif j == (self.network.v_length - 2) and self.network.v_lights[i] == 1:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1
                else:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1

        # look for collisions
        for i in range(self.network.n_inter):
            if self.network.h_cars[i, (self.network.h_length - 1)] == 1 and self.network.v_cars[i, (self.network.v_length - 1)] == 1:
                done = True

        # generate random cars coming in the network
        if np.random.uniform() < self.TRAFFIC_INTENSITY and self.network.h_cars[0, 0] == 0:
            self.network.h_cars[0, 0] = 1

        for i in range(self.network.n_inter):
            if np.random.uniform() < self.TRAFFIC_INTENSITY and self.network.v_cars[i, 0] == 0:
                self.network.v_cars[i, 0] = 1

        # final state
        state = np.concatenate(self.network.h_cars,
                               self.network.v_cars,
                               self.network.h_lights,
                               self.network.h_lights)

        return state, reward, done, {}

    def reset(self):

        self.current_timestep = 0
        self.network = NetworkStructure(self.n_inter, self.h_length, self.v_length)
        state = np.concatenate(self.network.h_cars,
                               self.network.v_cars,
                               self.network.h_lights,
                               self.network.h_lights)

        return state

    def seed(self, seed):
        return

    def render(self):
        return self.viewer.render(return_rgb_array = False)
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
