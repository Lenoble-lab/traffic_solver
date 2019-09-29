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

        self.time_reward = 0.1
        self.v_time_spend = [0] * n_inter
        self.h_time_spend = [0] * n_inter

        self.std_reward = 1.0

        self.out_reward = 10.0

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

        self.TRAFFIC_INTENSITY = 0.8

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
                # car at intersection i
                if j == (self.network.h_length - 1) and self.network.h_cars[i, j] == 1:
                    # if last interval, leave network
                    if i == self.network.n_inter - 1:
                        self.network.h_cars[i, j] = 0
                        reward += self.out_reward
                    # if not last interval, choose random direction
                    else:
                        # choose random direction depending on traffic density in next horizontal interval
                        leave = (np.random.uniform() < np.mean(self.network.h_cars[i+1, :]))
                        if leave:
                            self.network.h_cars[i, j] = 0
                            reward += 1
                        elif not leave and self.network.h_cars[i+1, 0] == 0:
                            self.network.h_cars[i, j] = 0
                            self.network.h_cars[i+1, 0] = 1
                # car before last moves only if light is green
                elif j == (self.network.h_length - 2) and self.network.h_lights[i] == 1 and self.network.h_cars[i, j] == 1 and self.network.h_cars[i, j+1] == 0:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1
                # other cars
                elif j < (self.network.h_length - 2) and self.network.h_cars[i, j] == 1 and self.network.h_cars[i, j+1] == 0:
                    self.network.h_cars[i, j] = 0
                    self.network.h_cars[i, j+1] = 1

            # move vertical cars
            for j in reversed(range(self.network.v_length)):
                # car at intersection i
                if j == (self.network.v_length - 1) and self.network.v_cars[i, j] == 1:
                    # if last interval, leave network
                    if i == self.network.n_inter - 1:
                        self.network.v_cars[i, j] = 0
                        reward += self.out_reward
                    # choose random direction depending in traffic density in next horizontal interval
                    else:
                        leave = (np.random.uniform() < np.mean(self.network.h_cars[i + 1, :]))
                        if leave:
                            self.network.v_cars[i, j] = 0
                            reward += 1
                        elif not leave and self.network.h_cars[i+1, 0] == 0:
                            self.network.v_cars[i, j] = 0
                            self.network.h_cars[i+1, 0] = 1
                # car before last moves only if light is green
                elif j == (self.network.v_length - 2) and self.network.v_lights[i] == 1 and self.network.v_cars[i, j] == 1 and self.network.v_cars[i, j+1] == 0:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1
                # other cars
                elif j < (self.network.v_length - 2) and self.network.v_cars[i, j] == 1 and self.network.v_cars[i, j+1] == 0:
                    self.network.v_cars[i, j] = 0
                    self.network.v_cars[i, j+1] = 1

                """
                # time spend in front of th light (for negative reward)
                elif j == (self.network.v_length - 2) and self.network.v_lights[i] == 0 and self.network.v_cars[i, j] == 1:
                    self.v_time_spend[i] += 1

                    if self.v_time_spend[i] == 5:
                        reward -= self.time_reward * self.v_time_spend[i]
                        self.v_time_spend[i] = 0
                # restart time if there is no one
                elif j == (self.network.h_length - 2) and self.network.v_cars[i, j] == 0:
                    reward -= self.time_reward * self.v_time_spend[i]
                    self.v_time_spend[i] = 0
                """

        # use standard deviation of traffic density in the different intervals for reward
        reward -= self.time_reward * np.std(np.concatenate((np.mean(self.network.h_cars, axis=1), np.mean(self.network.v_cars, axis=1))))

        # look for collisions
        for i in range(self.network.n_inter):
            if self.network.h_cars[i, (self.network.h_length - 1)] == 1 and self.network.v_cars[i, (self.network.v_length - 1)] == 1:
                done = True

        # generate random cars coming in the network
        if np.random.uniform() < self.TRAFFIC_INTENSITY and self.network.h_cars[0, 0] == 0:
            self.network.h_cars[0, 0] = 1

        for i in range(self.network.n_inter):
            if np.random.uniform() < (self.TRAFFIC_INTENSITY / 4) and self.network.v_cars[i, 0] == 0:
                self.network.v_cars[i, 0] = 1

        # final state
        state = np.concatenate((self.network.h_cars.reshape(self.network.n_inter * self.network.h_length),
                               self.network.v_cars.reshape(self.network.n_inter * self.network.v_length),
                               self.network.h_lights,
                               self.network.v_lights))

        return state, reward, done, {}

    def reset(self):

        self.current_timestep = 0
        self.network = NetworkStructure(self.n_inter, self.h_length, self.v_length)
        state = np.concatenate((self.network.h_cars.reshape(self.network.n_inter * self.network.h_length),
                               self.network.v_cars.reshape(self.network.n_inter * self.network.v_length),
                               self.network.h_lights,
                               self.network.v_lights))

        return state

    def seed(self, seed):
        return

    def render(self):
        scale = 0.5
        road_length = 80 * self.network.h_length
        road_height = 80 * self.network.v_length
        screen_width = 1500
        screen_height = 800
        carwidth = int(scale *30)
        carheight = int(scale*30)

        traffic_light_radius = int(scale * 10)

        l, r, t, b = -carwidth//2, carwidth//2, carheight//2, -carheight//2

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
                    cartrans.set_translation(int(scale*(45 + 80*(i+1) + s*road_length)), int(scale*(road_height-20)))

                    self.viewer.add_onetime(car)

        # flux verticaux
        for s in range(self.n_inter):
            for i in range(self.v_length):
                if self.network.v_cars[s][i] == 1:
                    car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    car.set_color(0, 0.3*s, 0.2*s)

                    cartrans = rendering.Transform()
                    car.add_attr(cartrans)
                    cartrans.set_translation(int(scale*(45+road_length+road_length*s)), int(scale*(60 + 80*i)))

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
            lighttrans.set_translation(int(scale*((s+1)*road_length)), int(scale*(25+road_height)))
            self.viewer.add_onetime(light1)

            light2 = rendering.make_circle(traffic_light_radius)
            if self.network.v_lights[s] == 0:
                light2.set_color(1, 0, 0)
            else:
                light2.set_color(0, 1, 0)
            lighttrans = rendering.Transform()
            light2.add_attr(lighttrans)
            lighttrans.set_translation(int(scale*(90+(s+1)*road_length)), int(scale*(road_height-65)))
            self.viewer.add_onetime(light2)

        # dessiner les lignes horizontales
        top_line = rendering.Line((0, int(scale*(5+road_height))), (screen_width, int(scale*(5+road_height))))
        top_line.set_color(0, 0, 0)
        self.viewer.add_onetime(top_line)

        bottom_line = rendering.Line((0, int(scale*(road_height-45))), (screen_width, int(scale*(road_height-45))))
        bottom_line.set_color(0, 0, 0)
        self.viewer.add_onetime(bottom_line)
        
        # dessiner les colonnes
        for s in range(self.n_inter):

            left_line = rendering.Line((int(scale*((20+(s+1)*road_length))), screen_height), (int(scale*(20+(s+1)*road_length)), 0))
            left_line.set_color(0, 0, 0)
            self.viewer.add_onetime(left_line)

            right_line = rendering.Line((int(scale*((70+(s+1)*road_length))), screen_height), (int(scale*(70+(s+1)*road_length)), 0))
            right_line.set_color(0, 0, 0)
            self.viewer.add_onetime(right_line)

        return self.viewer.render(return_rgb_array=False)
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
