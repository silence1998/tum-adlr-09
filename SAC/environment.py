
import math
import numpy as np

import gym
from gym import spaces
import pygame
import parameters

from collections import deque

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": parameters.env_parameters["render_fps"]}  # fps was 4

    def __init__(self, render_mode=None, object_radius=100, num_obstacles=5, window_size=1024):
        self.radius = object_radius  # The size of the radius of the elements, defined in the training and plotting scripts
        self.window_size = window_size  # The size of the PyGame window
        self.num_obstacles = num_obstacles
        self.total_step = 0

        ### REWARD PARAMETERS ###
        self.reward_parameters = parameters.reward_parameters
        self.env_parameters = parameters.env_parameters

        self._agent_location = None
        self._target_location = None
        self._obstacle_locations = None
        self._obstacle_velocities = None
        if parameters.reward_parameters['checkpoints']:
            self.checkpoint_reward_given = [False] * (self.reward_parameters['checkpoint_number'] + 1)
        if parameters.reward_parameters['history']:
            self._agent_location_history = deque(maxlen=parameters.reward_parameters['history_size'])

        # Observations are dictionaries with the agent's, obstacles' and the target's location.

        elements = {"agent": spaces.Box(low=np.array([self.radius, self.radius, -1, -1]),  # x, y, vx, vy
                                            high=np.array([self.window_size - self.radius, self.window_size - self.radius, 1, 1]),
                                            shape=(4,),  # x, y, vx, vy
                                            dtype=np.float32),
                    # "agent": spaces.Box(self.radius, self.window_size - self.radius, shape=(2,), dtype=np.float32),
                    "target": spaces.Box(self.radius, self.window_size - self.radius, shape=(2,), dtype=np.float32)}
        for idx_obstacle in range(self.num_obstacles):

            elements.update({"obstacle_{0}".format(idx_obstacle):
                                 spaces.Box(low=np.array([self.radius, self.radius, -1, -1]),  # x, y, vx, vy
                                            high=np.array([self.window_size - self.radius, self.window_size - self.radius, 1, 1]),
                                            shape=(4,),  # x, y, vx, vy
                                            dtype=np.float32)})

        self.observation_space = spaces.Dict(elements)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None



    def _get_obs(self):
        elements = {"agent": np.concatenate((self._agent_location, self._agent_velocity)),  # TODO check type
                    "target": self._target_location}
        for idx_obstacle in range(self.num_obstacles):
            elements.update({"obstacle_{0}".format(idx_obstacle):
                np.concatenate((self._obstacle_locations[str(idx_obstacle)],
                self._obstacle_velocities[str(idx_obstacle)]))
            })
        return elements


    def _get_info(self):
        distances = {"distance_to_target":
                         self.euclidean_norm(self._agent_location - self._target_location)}
        for idx_obstacle in range(self.num_obstacles):
            distances.update({"distance_to_obstacle_{0}".format(idx_obstacle):
                                  self.euclidean_norm(self._agent_location - self._obstacle_locations[str(idx_obstacle)])})
        return distances



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random

        super().reset(seed=seed)

        self.total_step = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.random(size=(2,), dtype=np.float32) * (self.window_size - 2 * self.radius)
        self._agent_velocity = self.np_random.random(size=(2,), dtype=np.float32) * 2 - 1

        if parameters.reward_parameters['history']:
            self._agent_location_history.extend(self._agent_location)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while self.euclidean_norm(self._target_location - self._agent_location) < self.radius * 2:
            self._target_location = self.np_random.random(size=(2,), dtype=np.float32) * (self.window_size - 2 * self.radius)



        # We will sample the obstacle's location randomly until it does not coincide
        # with the agent's/target's/other obstacles location
        self._obstacle_locations = {}  # TODO: how to check if this works
        for idx_obstacle in range(self.num_obstacles):
            self._obstacle_locations.update({"{0}".format(idx_obstacle): self._agent_location})
            while (self.euclidean_norm(self._obstacle_locations[str(idx_obstacle)]
                                       - self._agent_location) < self.radius * 2) \
                    or (self.euclidean_norm(self._obstacle_locations[str(idx_obstacle)]
                                       - self._target_location) < self.radius * 2):  # Colliding with agent or target
                _obstacle_locations_array = np.array(list(self._obstacle_locations.values()))
                random_location = self.np_random.random(size=(2,), dtype=np.float32) * (self.window_size - 2 * self.radius)
                random_location_rep = np.array(
                     [random_location for i in range(len(_obstacle_locations_array))])
                #print(random_location_rep)
                #print(_obstacle_locations_array)
                distances = np.array(self.elementwise_euclidean_norm(_obstacle_locations_array, random_location_rep))
                collision = distances - 2 * self.radius
                if (collision < 0).any():  # Colliding with other obstacle
                    continue
                self._obstacle_locations[str(idx_obstacle)] = np.array(random_location_rep[0])
                #print(random_location_rep[0])
        assert len(self._obstacle_locations) == self.num_obstacles

        self._obstacle_velocities = {}
        for idx_obstacle in range(0, int(np.ceil(self.num_obstacles/2))):
            self._obstacle_velocities.update({"{0}".format(idx_obstacle): np.array([0., 0.], dtype=np.float32)})
        for idx_obstacle in range(int(np.ceil(self.num_obstacles/2)), self.num_obstacles):
            self._obstacle_velocities.update({"{0}".format(idx_obstacle):
                                                  (
                                                          self.np_random.random(size=(2,),
                                                                                dtype=np.float32) * 2 - 1)})  # between [-1, 1]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action_step):
        global penalty_distance_collision


        ### ENVIRONMENT UPDATE ###
        action_step = (self.env_parameters["delta_T"] *
                       self.env_parameters["action_step_scaling"] *
                      action_step)  # a float value from [-1, 1] otherwise problem in entropy

        previous_position = self._agent_location
        if parameters.reward_parameters['history']:
            self._agent_location_history.extend(self._agent_location)
        self._agent_location = self._agent_location + action_step
        self._agent_velocity = self._agent_location - previous_position #/ self.env_parameters["delta_T"] # TODO: correct? so that we keep the velocity in [-1,1 interval]

        for idx_obstacle in range(self.num_obstacles):
            self._obstacle_locations.update({"{0}".format(idx_obstacle):
                                                 self._obstacle_locations["{0}".format(idx_obstacle)] +
                                                 self.env_parameters["delta_T"] *
                                                 self._obstacle_velocities["{0}".format(idx_obstacle)]})

        self._max_distance = math.sqrt(2) * self.window_size

        ## if too many steps

        if self.total_step > self.reward_parameters['total_step_limit']:
            terminated = True  # agent is out of bounds but did not collide with obstacle
            reward = self.reward_parameters['reward_reach_limit']  # collision with wall or obstacles
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info



        ### COLLISION SUPER SPARSE REWARD ###
        # Check for obstacle collision
        terminated = False
        for idx_obstacle in range(self.num_obstacles):
            terminated = self.euclidean_norm(self._obstacle_locations[str(idx_obstacle)] -
                                             self._agent_location) < self.radius * 2
            if terminated:
                break
        # Check if the agent is out of bounds
        if self._agent_location[0] < self.radius or \
            self._agent_location[1] < self.radius or \
            self._agent_location[0] > self.window_size - self.radius or \
            self._agent_location[1] > self.window_size - self.radius or \
            terminated:
            terminated = True  # agent is out of bounds but did not collide with obstacle

            reward = self.reward_parameters['collision_value']  # collision with wall or obstacles

            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info
        reward = 0
        ### TARGET SUPER SPARSE REWARD ###
        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)  # target reached
        terminated = self.euclidean_norm(self._target_location -
                                             self._agent_location) < self.radius * 2
        if terminated:
            reward = self.reward_parameters['target_value']  # sparse target reward

        ## OTHER REWARDS ### -> SKILLS
        else:
            ### Distances
            # Distance to target
            if self.reward_parameters['obstacle_avoidance'] \
                    or self.reward_parameters['target_seeking']:
                previous_distance_to_target = math.sqrt((previous_position[0] - self._target_location[0]) ** 2
                                                        + (previous_position[1] - self._target_location[1]) ** 2)

                distance_to_target = math.sqrt((self._agent_location[0] - self._target_location[0]) ** 2
                                               + (self._agent_location[1] - self._target_location[1]) ** 2)

                # Distances to obstacles
                previous_distances_to_obstacles = np.array([])
                distances_to_obstacles = np.array([])
                for idx_obstacle in self._obstacle_locations:
                    previous_distance_to_obstacle = math.sqrt(
                        (previous_position[0] - self._obstacle_locations[str(idx_obstacle)][0]) ** 2
                        + (previous_position[1] - self._obstacle_locations[str(idx_obstacle)][1]) ** 2)
                    distance_to_obstacle = math.sqrt(
                        (self._agent_location[0] - self._obstacle_locations[str(idx_obstacle)][0]) ** 2
                        + (self._agent_location[1] - self._obstacle_locations[str(idx_obstacle)][1]) ** 2)
                    np.append(previous_distances_to_obstacles, previous_distance_to_obstacle)
                    np.append(distances_to_obstacles, distance_to_obstacle)

                # Distance to the closest wall
                previous_distance_to_wall = np.amin(np.vstack((previous_position + self.radius + 1,
                                                               self.window_size - (self.radius + previous_position))))
                # get the distance to the closest wall in the previous step
                distance_to_wall = np.amin(np.vstack((self._agent_location + self.radius + 1,
                                                      self.window_size - (self.radius + previous_position))))
                # get the distance to the closest wall in the current step

                ### Distance differences
                # Difference to target # TODO: make use of this?
                diff_distance_to_target = np.abs(previous_distance_to_target - distance_to_target)

                # Difference to obstacles # TODO: make this a parameter, is this a good idea?
                distances_to_obstacles[distances_to_obstacles > 0.3 * self.window_size] = 0
                # set distances to obstacles > 5 to 0

                previous_distances_to_obstacles[distances_to_obstacles == 0] = 0
                # set previous distances to obstacles 0 with the same indices as distances to obstacles

                diff_obstacle_distances = np.abs(
                    previous_distances_to_obstacles - distances_to_obstacles)  # TODO: make use of this

                # Difference to wall
                diff_distance_to_wall = np.abs(distance_to_wall - previous_distance_to_wall)  # TODO: make use of this

            reward = 0  # TODO: remove this??

            ### COLLISION PREDICTION PENALTY###
            if self.reward_parameters['collision_prediction']:
                self._obstacle_locations_pred = self._obstacle_locations  # TODO: how to check if this works
                for idx_obstacle in range(self.num_obstacles):
                    self._obstacle_locations_pred.update({"{0}".format(idx_obstacle):
                                                              self._obstacle_locations["{0}".format(idx_obstacle)] +
                                                              self.env_parameters["delta_T"] *
                                                              self._obstacle_velocities["{0}".format(idx_obstacle)]})

                    _obstacle_locations_array = np.array(list(self._obstacle_locations_pred.values()))
                    _agent_location_rep = np.array([self._agent_location for i in range(len(_obstacle_locations_array))])
                    distances = np.array(self.elementwise_euclidean_norm(_obstacle_locations_array, _agent_location_rep))
                    collision = distances - 2 * self.radius
                    #avoidance = distances - 3 * self. # TODO avoidance SR
                    #if avoidance...
                    #    rewards
                    if (collision < 0).any():
                        reward += self.reward_parameters['collision_prediction_penalty']  # collision with wall or obstacles

            ### DENSE REWARDS ###
            # Reward for avoiding obstacles
            if self.reward_parameters['obstacle_avoidance']:
                min_collision_distance = np.min(np.append(distances_to_obstacles, [distance_to_wall]))
                penalty_distance_collision = np.max(np.array([1.0 - min_collision_distance / self._max_distance, 0.0]))
                reward += self.reward_parameters['obstacle_distance_weight'] * penalty_distance_collision

            if self.reward_parameters['target_seeking']:
                reward += self.reward_parameters['target_distance_weight'] * distance_to_target / self._max_distance

            ### SUB-SPARSE REWARDS ###
            # Distance checkpoint rewards
            if self.reward_parameters['checkpoints']:
                distance_to_target = math.sqrt((self._agent_location[0] - self._target_location[0]) ** 2
                                               + (self._agent_location[1] - self._target_location[1]) ** 2)
                for i in np.flip(range(1, self.reward_parameters['checkpoint_number'] + 1)):
                    if not self.checkpoint_reward_given[i]:
                        if (distance_to_target < i * self.reward_parameters['checkpoint_distance_proportion'] * self.window_size):
                            self.checkpoint_reward_given[i] = True
                            reward += self.reward_parameters['checkpoint_value']  # checkpoint reward

            # Time penalty
            if self.reward_parameters['time']:
                reward += self.reward_parameters['time_penalty']

            if parameters.reward_parameters['history']:
                # Waiting penalty
                last_x_positions = list(self._agent_location_history)
                last_x_positions = last_x_positions[-self.reward_parameters['max_waiting_steps']:]
                if self.reward_parameters['waiting']:
                    if last_x_positions.count(last_x_positions[-1]) == len(last_x_positions):  # Checks if all positions are equal
                        reward += self.reward_parameters['waiting_penalty']

                # Waiting reward # TODO: history??
                last_x_positions = list(self._agent_location_history)
                last_x_positions = last_x_positions[-self.reward_parameters['waiting_step_number_to_check']:]
                if self.reward_parameters['waiting']:
                    if last_x_positions.count(last_x_positions[-1]) == len(last_x_positions):  # Checks if all positions are equal
                        reward += self.reward_parameters['waiting_value']

                # Consistency reward # TODO: history??
                if self.reward_parameters['consistency']:
                    last_x_steps = self._agent_location_history[-self.reward_parameters['consistency_step_number_to_check']:]
                    for i in np.flip((range(1, self.reward_parameters['consistency_step_number_to_check'] + 1))):  # csn,...,1
                        last_x_steps.append(last_x_positions[-1] - last_x_positions[i - 1])
                        if last_x_steps.count(last_x_steps[0]) == len(last_x_steps):  # Checks if all directions are equal
                            reward += self.reward_parameters['consistency_value']

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        # self.window_size = 1024
        pix_square_size = self.radius
        agent_size = pix_square_size
        target_size = pix_square_size
        object_radius = pix_square_size

        # First we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # blue
            self._agent_location,
            agent_size,
        )
        # Now we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),  # red
            self._target_location,
            target_size,
        )
        # Now we draw the obstacles
        for idx_obstacle in range(self.num_obstacles):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),  # black
                self._obstacle_locations[str(idx_obstacle)],
                object_radius,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame_for_gif(self):  # TODO: This is not working, after switching to continious adjust to the above render frame function
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Now we draw the obstacles
        for idx_obstacle in range(self.num_obstacles):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * self._obstacle_locations[str(idx_obstacle)],
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    @staticmethod
    def euclidean_norm(vector):
        """Calculates the Euclidean norm of a given vector."""
        norm = 0
        for i in range(len(vector)):
            norm += vector[i] ** 2
        return math.sqrt(norm)

    @staticmethod
    def elementwise_euclidean_norm(vec1, vec2):
        """Calculates the element-wise Euclidean norm of two given 2D vectors."""
        elementwise_norm = []
        for i in range(len(vec1)):
            elementwise_norm.append(math.sqrt((vec1[i][0] - vec2[i][0]) ** 2 + (vec1[i][1] - vec2[i][1]) ** 2))
        return elementwise_norm