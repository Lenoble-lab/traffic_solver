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
    
        self.h_time_spend = [0] * n_inter
        self.v_time_spend = [0] * n_inter

        self.decrease_reward = 0.01

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
                        if self.network.h_cars[i+1, 0] == 0:
                            self.network.h_cars[i+1, 0] = 1
                        else:
                            self.network.h_cars[i, j] = 1
                # car before last moves only if light is green
                elif j == (self.network.h_length - 2) and self.network.h_lights[i] == 1 and self.network.h_cars[i, j] == 1 and self.network.h_cars[i, j+1] == 0:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1
                elif j < (self.network.h_length - 2) and self.network.h_cars[i, j] == 1 and self.network.h_cars[i, j+1] == 0:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1
                #time spend in front of the light
                elif j == (self.network.h_length - 2) and self.network.h_lights[i] == 0  and self.network.h_cars[i, j] == 1: 
                    self.h_time_spend[i] += 1

                #re-start time
                elif j == (self.network.h_length - 2) and self.network.h_cars[i, j] == 0:
                     reward -= self.decrease_reward * self.h_time_spend[i]
                     self.h_time_spend[i] = 0

            # move vertical cars
            for j in reversed(range(self.network.v_length)):
                # last car in interval
                if j == (self.network.v_length - 1) and self.network.v_cars[i, j] == 1:
                    self.network.v_cars[i, j] = 0
                    # reward if car leaves network
                    reward += 1
                # car before last moves only if light is green
                elif j == (self.network.v_length - 2) and self.network.v_lights[i] == 1 and self.network.v_cars[i, j] == 1 and self.network.v_cars[i, j+1] == 0:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1
                elif j < (self.network.v_length - 2) and self.network.v_cars[i, j] == 1 and self.network.v_cars[i, j+1] == 0:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1
                #time spend in front of th light (for the reward)
                elif j == (self.network.v_length - 2) and self.network.v_lights[i] == 0  and self.network.v_cars[i, j] == 1: 
                    self.v_time_spend[i] += 1
                #restart time if there is no one
                elif j == (self.network.h_length - 2) and self.network.v_cars[i, j] == 0:
                     reward -= self.decrease_reward * self.v_time_spend[i]
                     self.v_time_spend[i] = 0

        # look for collisions
        for i in range(self.network.n_inter):
            if self.network.h_cars[i, (self.network.h_length - 1)] == 1 and self.network.v_cars[i, (self.network.v_length - 1)] == 1:
                done = True

        # generate random cars coming in the network
        if np.random.uniform() < 2 * self.TRAFFIC_INTENSITY and self.network.h_cars[0, 0] == 0:
            self.network.h_cars[0, 0] = 1

        for i in range(self.network.n_inter):
            if np.random.uniform() < self.TRAFFIC_INTENSITY and self.network.v_cars[i, 0] == 0:
                self.network.v_cars[i, 0] = 1

        # final state
        state = np.concatenate((self.network.h_cars.reshape(self.network.n_inter * self.network.h_length),
                               self.network.v_cars.reshape(self.network.n_inter * self.network.v_length),
                               self.network.h_lights,
                               self.network.h_lights))

        return state, reward, done, {}

    def reset(self):

        self.current_timestep = 0
        self.network = NetworkStructure(self.n_inter, self.h_length, self.v_length)
        state = np.concatenate((self.network.h_cars.reshape(self.network.n_inter * self.network.h_length),
                               self.network.v_cars.reshape(self.network.n_inter * self.network.v_length),
                               self.network.h_lights,
                               self.network.h_lights))

        return state

    def seed(self, seed):
        return

    def render(self):
       
        road_length = 80 * self.network.h_length
        road_height = 80 * (self.network.v_length-4)
        screen_width = 1500
        screen_height = 800
        carwidth = 30
        carheight = 30

        traffic_light_radius = 10

        l, r, t, b = -carwidth/2, carwidth/2, carheight/2, -carheight/2

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

        # flux horizontaux
        for s in range(self.n_inter):
            for i in range(self.h_length):
                if self.network.h_cars[s][i] == 1:

                    car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    car.set_color(1, 0, 0)

                    cartrans = rendering.Transform()
                    car.add_attr(cartrans)
                    cartrans.set_translation(20 + 80*i + s*road_length, 300+road_height)

                    self.viewer.add_onetime(car)

        # flux verticaux
        for s in range(self.n_inter):
            for i in range(self.v_length):
                if self.network.v_cars[s][i] == 1:
                    car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    car.set_color(0, 0.3*s, 0.2*s)

                    cartrans = rendering.Transform()
                    car.add_attr(cartrans)
                    cartrans.set_translation(260+road_length*s, 60 + 80*i)

                    self.viewer.add_onetime(car)
        
        # traffic light
        for s in range(self.n_inter):
            light1 = rendering.make_circle(traffic_light_radius)
            if self.network.h_lights[s] == 0:
                light1.set_color(1, 0, 0)
            else:
                light1.set_color(0, 1, 0)
            lighttrans = rendering.Transform()
            light1.add_attr(lighttrans)
            lighttrans.set_translation(220+s*road_length, 350+road_height)
            self.viewer.add_onetime(light1)

            light2 = rendering.make_circle(traffic_light_radius)
            if self.network.v_lights[s] == 0:
                light2.set_color(1, 0, 0)
            else:
                light2.set_color(0, 1, 0)
            lighttrans = rendering.Transform()
            light2.add_attr(lighttrans)
            lighttrans.set_translation(300+s*road_length, 250+road_height)
            self.viewer.add_onetime(light2)

        # dessiner les lignes horizontales
        top_line = rendering.Line((0, 325+road_height), (screen_width, 325+road_height))
        top_line.set_color(0, 0, 0)
        self.viewer.add_onetime(top_line)

        bottom_line = rendering.Line((0, 275+road_height), (screen_width, 275+road_height))
        bottom_line.set_color(0, 0, 0)
        self.viewer.add_onetime(bottom_line)
        
        # dessiner les colonnes
        for i in range(self.n_inter):

            left_line = rendering.Line((235+i*road_length, screen_height), (235+i*road_length, 0))
            left_line.set_color(0, 0, 0)
            self.viewer.add_onetime(left_line)

            right_line = rendering.Line((285+i*road_length, screen_height), (285+i*road_length, 0))
            right_line.set_color(0, 0, 0)
            self.viewer.add_onetime(right_line)

        return self.viewer.render(return_rgb_array=False)
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
