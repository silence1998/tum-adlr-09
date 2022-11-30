
import math
import numpy as np

import gym
from gym import spaces
import pygame

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_obstacles=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.num_obstacles = num_obstacles

        self._agent_location = None
        self._target_location = None
        self._obstacle_locations = None
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        elements = {"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)}
        for idx_obstacle in range(self.num_obstacles):
            elements.update({"obstacle_{0}".format(idx_obstacle): spaces.Box(0, size - 1, shape=(2,), dtype=int)})
        self.observation_space = spaces.Dict(elements)

        ### REWARD PARAMETERS ###
        self.reward_parameters = {
            # 'field_of_view': 5,  # see min_collision_distance
            # 'collision_weight': 0.3,
            # 'time_weight': 1,
            # the above are not used in the current version which is sparse reward based

            'action_step_scaling': 1,  # 1 step -> "2" grids of movement reach in x and y directions
            ### DENSE REWARDS ###
            'obstacle_avoidance': False,
            'obstacle_distance_weight': -1,
            'target_seeking': True,
            'target_distance_weight': 1,

            ### SPARSE REWARDS ###
            'target_value': 1,
            'collision_value': -1,

            ### SUB-SPARSE REWARDS ###
            'checkpoints': True,  # if true, use checkpoints rewards
            'checkpoint_distance_proportion': 0.0,
            'checkpoint_number': 5,  # make sure checkpoint_distance_proportion * "checkpoint_number" <= 1
            'checkpoint_value': 0.0,  # make sure checkpoint_value * checkpoint_number < 1

            'time': True,  # if true, use time rewards
            'time_penalty': -0.01,  # == penalty of -1 for "100" action steps

            'history_size': 20,  # size of history to check for waiting and consistency

            'waiting': False,  # if true, use waiting rewards # TODO: implement action history in step()
            'waiting_value': 0.0,  # make sure waiting_value < 1
            'max_waiting_steps': 20,  # make sure < history_size, punishment for waiting too long
            # threshold

            'consistency': False,  # if true, use consistency rewards # TODO: implement action history in step()
            'consistency_step_number': 5,  # make sure consistency_step_number < history_size
            'consistency_value': 0.0,  # make sure consistency_value * consistency_step_number < 1
            # threshold

            # fast moving etc. for sub actions for sparse rewards
        }

        # TODO action space should be continuous now its bounded in [-3, 3]
        self.action_space = spaces.Discrete(4)  # Continuous 3 see gym examples
        #Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32) - Box(3,) x,y velocity
        #no polar coord as its already encoded

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            # TODO: a normalized direction vector and a scalar amount of velocity [-1,1]
            # if time, dynamics
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

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
        elements = {"agent": self._agent_location, "target": self._target_location}
        for idx_obstacle in range(self.num_obstacles):
            elements.update({"obstacle_{0}".format(idx_obstacle): self._obstacle_locations[str(idx_obstacle)]})
        return elements

    def _get_info(self):
        distances = {"distance_to_target":
                         np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        # ord=1: max(sum(abs(x), axis=0))
        for idx_obstacle in range(self.num_obstacles):
            distances.update({"distance_to_obstacle_{0}".format(idx_obstacle):
                                  np.linalg.norm(self._agent_location - self._obstacle_locations[str(idx_obstacle)],
                                                 ord=1)})
        return distances

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # We will sample the obstacle's location randomly until it does not coincide with the agent's/target's location
        self._obstacle_locations = {}
        for idx_obstacle in range(self.num_obstacles):
            self._obstacle_locations.update({"{0}".format(idx_obstacle): self._agent_location})
            while np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._agent_location) \
                    or np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._target_location):
                random_location = np.array(self.np_random.integers(0, self.size, size=2, dtype=int))
                _obstacle_locations_array = np.array(self._obstacle_locations.values())
                if (idx_obstacle != 0) & (np.array_equal(random_location, _obstacle_locations_array.any())):
                    print(_obstacle_locations_array)
                    print("\n \n \n \n Collision in random object generation!!! \n \n \n \n")
                    continue
                self._obstacle_locations[str(idx_obstacle)] = random_location
                assert not np.array_equal(random_location, _obstacle_locations_array.any())
        assert len(self._obstacle_locations) == self.num_obstacles

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action_step):
        global penalty_distance_collision
        action_step = np.round(
            self.reward_parameters[
                "action_step_scaling"] * action_step)  # scale action to e.g. [-2, 2] -> reach is 5x5 grid
        previous_position = self._agent_location
        self._agent_location = self._agent_location + action_step
        self._max_distance = math.sqrt(2) * self.size

        ### COLLISION SPARSE REWARD ###
        # Check for obstacle collision
        terminated = False
        for idx_obstacle in range(self.num_obstacles):
            terminated = np.array_equal(self._agent_location, self._obstacle_locations[str(idx_obstacle)])
            if terminated:
                break
        # Check if the agent is out of bounds
        if self._agent_location[0] < 0 or self._agent_location[1] < 0 or \
                self._agent_location[0] > self.size - 1 or self._agent_location[1] > self.size - 1 or \
                terminated:
            terminated = True  # agent is out of bounds but did not collide with obstacle

            reward = self.reward_parameters['collision_value']  # collision with wall or obstacles

            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info

        ### TARGET SPARSE REWARD ###
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)  # target reached
        if terminated:
            reward = self.reward_parameters['target_value']  # sparse target reward

        ### OTHER REWARDS ###
        else:
            ### Distances
            # Distance to target

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
            previous_distance_to_wall = np.amin(np.vstack((previous_position + 1, self.size - previous_position)))
            # get the distance to the closest wall in the previous step
            distance_to_wall = np.amin(np.vstack((self._agent_location + 1, self.size - self._agent_location)))
            # get the distance to the closest wall in the current step # TODO: check if this is correct (@Mo)

            ### Distance differences
            # Difference to target
            diff_distance_to_target = np.abs(previous_distance_to_target - distance_to_target)

            # Difference to obstacles TODO: make this a parameter, is this a good idea?
            distances_to_obstacles[distances_to_obstacles > 0.3 * self.size] = 0  # set distances to obstacles > 5 to 0
            previous_distances_to_obstacles[distances_to_obstacles == 0] = 0
            # set previous distances to obstacles 0 with the same indices as distances to obstacles
            diff_obstacle_distances = np.abs(
                previous_distances_to_obstacles - distances_to_obstacles)  # TODO: make use of this

            # Difference to wall
            diff_distance_to_wall = np.abs(distance_to_wall - previous_distance_to_wall)  # TODO: make use of this

            reward = 0

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
                checkpoint_reward_given = [False] * (self.reward_parameters['checkpoint_number'] + 1)
                for i in np.invert(range(1, self.reward_parameters['checkpoint_number'] + 1)):
                    if (distance_to_target < i * self.reward_parameters['checkpoint_distance_proportion'] * self.size) \
                            and not checkpoint_reward_given[i]:
                        checkpoint_reward_given[i] = True
                        reward += self.reward_parameters['checkpoint_value']  # checkpoint reward

            # Time penalty
            if self.reward_parameters['time']:
                reward += self.reward_parameters['time_penalty']  # time penalty

            # last_x_positions = self._agent_location_history[-self.reward_parameters['history_size']:]
            # # Waiting reward # TODO: add step history to check if the agent is waiting
            # if self.reward_parameters['waiting']:
            #     if last_x_positions.count(last_x_positions[0]) == len(last_x_positions):  # Checks if all positions are equal
            #         reward += self.reward_parameters['waiting_value']
            #
            # # Consistency reward # TODO: add step history to check if the agent is waiting
            # if self.reward_parameters['consistency']:
            #     last_x_steps = []
            #     for i in np.invert((range(1, self.reward_parameters['consistency_step_number'] + 1))):  # csn,...,1
            #         last_x_steps.append(last_x_positions[i] - last_x_positions[i - 1])
            #         if last_x_steps.count(last_x_steps[0]) == len(last_x_steps):  # Checks if all directions are equal
            #             reward += self.reward_parameters['consistency_value']

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


