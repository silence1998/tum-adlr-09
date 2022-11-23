import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import gym
from gym import spaces
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Used Sources:
https://www.gymlibrary.dev/content/environment_creation/
https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
"""


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, reward_parameters=None, num_obstacles=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.reward_parameters = reward_parameters
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

        # TODO action space should be continuous and bounded in [-3, 3]
        self.action_space = spaces.Discrete(4) # Continuous 3 see gym examlpes
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
            if idx_obstacle == 0:
                while np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._agent_location) \
                        or np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._target_location):
                    self._obstacle_locations[str(idx_obstacle)] = self.np_random.integers(
                        0, self.size, size=2, dtype=int
                    )
                continue
            while np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._agent_location) \
                    or np.array_equal(self._obstacle_locations[str(idx_obstacle)], self._target_location) \
                    or np.array_equal(self._obstacle_locations[str(idx_obstacle)],
                                      self._obstacle_locations[str(idx_obstacle - 1)]):
                self._obstacle_locations[str(idx_obstacle)] = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
            # TODO: check all obstacles pairwise for every additional obstacle
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
                checkpoint_reward_given = [False] * (reward_parameters['checkpoint_number'] + 1)
                for i in np.invert(range(1, reward_parameters['checkpoint_number'] + 1)):
                    if (distance_to_target < i * reward_parameters['checkpoint_distance_proportion'] * self.size) \
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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):  # a memory buffer to store transitions

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=16, fc2_dims=16,
                 name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=16, fc2_dims=16,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=16,
                 fc2_dims=16, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)  # log_std

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=2)  # TODO: decaying sigma

        return mu, sigma

    def sample_normal(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action_sample = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action_sample.pow(2) + self.reparam_noise)  # lower bound for probabilities
        log_probs = log_probs.sum(1, keepdim=True)

        return action_sample, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


def optimize_model():
    if len(memory) < hyper_parameters["batch_size"]:  # if memory is not full enough to start training, return
        return
    transitions = memory.sample(hyper_parameters["batch_size"])  # sample a batch of transitions from memory
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    value = valueNet(state_batch).view(-1)  # infer size of batch
    value_ = torch.zeros(hyper_parameters["batch_size"], device=device)
    value_[non_final_mask] = target_valueNet(non_final_next_states).view(-1)

    actions, log_probs = actorNet.sample_normal(state_batch, reparametrize=False)
    log_probs = log_probs.view(-1)
    q1_new_policy = criticNet_1.forward(state_batch, actions)
    q2_new_policy = criticNet_2.forward(state_batch, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    valueNet.optimizer.zero_grad()
    value_target = critic_value - log_probs
    value_loss = 0.5 * F.mse_loss(value, value_target)
    value_loss.backward(retain_graph=True)
    valueNet.optimizer.step()

    actions, log_probs = actorNet.sample_normal(state_batch, reparametrize=True)
    log_probs = log_probs.view(-1)
    q1_new_policy = criticNet_1.forward(state_batch, actions)
    q2_new_policy = criticNet_2.forward(state_batch, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    actor_loss = log_probs - critic_value
    actor_loss = torch.mean(actor_loss)
    print(actor_loss)
    actorNet.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actorNet.optimizer.step()

    criticNet_1.optimizer.zero_grad()
    criticNet_2.optimizer.zero_grad()
    q_hat = reward_batch + hyper_parameters["gamma"] * value_
    q1_old_policy = criticNet_1.forward(state_batch, action_batch).view(-1)
    q2_old_policy = criticNet_2.forward(state_batch, action_batch).view(-1)
    critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)  # line 13 in s.u. pseudocode
    critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    criticNet_1.optimizer.step()  # line 13 in s.u. pseudocode
    criticNet_2.optimizer.step()
    # TODO: check tau

def select_action(state):
    # state = torch.Tensor([state]).to(actorNet.device)
    actions, _ = actorNet.sample_normal(state, reparametrize=False)

    return actions.cpu().detach().numpy()[0]


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take X episode averages and plot them too
    avg_every_X_episodes = 25
    if len(durations_t) >= avg_every_X_episodes:
        means = durations_t.unfold(0, avg_every_X_episodes, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(avg_every_X_episodes - 1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


# initialize hyper-parameters

num_obstacles = 5
hyper_parameters = {
    'input_dims': 4 + num_obstacles * 2,  # original position of actor, obstacle and target position
    'batch_size': 256,
    'gamma': 0.999,  # discount factor
    'target_update': 10,  # update target network every 10 episodes TODO: UNUSED?
    'alpha': 0.0003,  # learning rate for actor
    'beta': 0.0003,  # learning rate for critic
    'tau': 0.005,  # target network soft update parameter (parameters = tau*parameters + (1-tau)*new_parameters)
    'num_episodes': 200,
    'env_size': 20  # size of the environment
}

reward_parameters = {
    # 'field_of_view': 5,  # see min_collision_distance
    # 'collision_weight': 0.3,
    # 'time_weight': 1,
    # the above are not used in the current version which is sparse reward based

    'action_step_scaling': 2,  # 1 step -> "2" grids of movement reach in x and y directions
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
    'checkpoint_distance_proportion': 0.2,
    'checkpoint_number': 5,  # make sure checkpoint_distance_proportion * "checkpoint_number" <= 1
    'checkpoint_value': 0.1,  # make sure checkpoint_value * checkpoint_number < 1

    'time': True,  # if true, use time rewards
    'time_penalty': -0.01,  # == penalty of -1 for "100" action steps

    'history_size': 20,  # size of history to check for waiting and consistency

    'waiting': False,  # if true, use waiting rewards # TODO: implement action history in step()
    'waiting_value': 0.1,  # make sure waiting_value < 1
    'max_waiting_steps': 20,  # make sure < history_size, punishment for waiting too long
    # threshold

    'consistency': False,  # if true, use consistency rewards # TODO: implement action history in step()
    'consistency_step_number': 5,  # make sure consistency_step_number < history_size
    'consistency_value': 0.1,  # make sure consistency_value * consistency_step_number < 1
    # threshold

    # fast moving etc. for sub actions for sparse rewards
}

env = GridWorldEnv(render_mode=None, size=hyper_parameters['env_size'], reward_parameters=reward_parameters)

# initialize NN
n_actions = 2  # velocity in 2 directions
actorNet = ActorNetwork(hyper_parameters["alpha"], hyper_parameters["input_dims"], n_actions=n_actions,
                        name='actor', max_action=[1, 1])  # TODO max_action value and min_action value
criticNet_1 = CriticNetwork(hyper_parameters["beta"], hyper_parameters["input_dims"], n_actions=n_actions,
                            name='critic_1')
criticNet_2 = CriticNetwork(hyper_parameters["beta"], hyper_parameters["input_dims"], n_actions=n_actions,
                            name='critic_2')
valueNet = ValueNetwork(hyper_parameters["beta"], hyper_parameters["input_dims"], name='value')
target_valueNet = ValueNetwork(hyper_parameters["beta"], hyper_parameters["input_dims"], name='target_value')

memory = ReplayMemory(10000)  # replay buffer size

steps_done = 0

episode_durations = []
for i_episode in range(hyper_parameters["num_episodes"]):
    # Initialize the environment and state
    env.reset()
    obs = env._get_obs()

    obs_values = [obs["agent"], obs["target"]]
    for idx_obstacle in range(num_obstacles):
        obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
    state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

    state = state.view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action)
        reward = torch.tensor([reward], dtype=torch.float, device=device)

        # Observe new state
        obs = env._get_obs()
        if not done:
            obs_values = [obs["agent"], obs["target"]]
            for idx_obstacle in range(num_obstacles):
                obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
            next_state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

            next_state = next_state.view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        action = np.array([action])
        action = torch.tensor(action, dtype=torch.float).to(actorNet.device)
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, using tau
    target_value_params = target_valueNet.named_parameters()
    value_params = valueNet.named_parameters()

    target_value_state_dict = dict(target_value_params)
    value_state_dict = dict(value_params)

    for name in value_state_dict:
        value_state_dict[name] = hyper_parameters["tau"] * value_state_dict[name].clone() + \
                                 (1 - hyper_parameters["tau"]) * target_value_state_dict[name].clone()
    target_valueNet.load_state_dict(value_state_dict)

print('Complete')

env.render_mode = "human"

# env=GridWorldEnv(render_mode="human")
i = 0
while i < 3:  # run plot for 3 episodes to see what it learned
    i += 1
    env.reset()
    obs = env._get_obs()

    obs_values = [obs["agent"], obs["target"]]
    for idx_obstacle in range(num_obstacles):
        obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
    state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

    state = state.view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action)

        action_ = torch.tensor(action, dtype=torch.float, device=device)
        action_ = action_.view(1, 2)
        mu, sigma = actorNet(state)
        print(actorNet(state))
        print(criticNet_1(state, action_))
        print(criticNet_2(state, action_))
        print(target_valueNet(state))

        reward = torch.tensor([reward], device=device)
        env._render_frame()
        # Observe new state
        obs = env._get_obs()
        if not done:
            obs_values = [obs["agent"], obs["target"]]
            for idx_obstacle in range(num_obstacles):
                obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
            next_state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

            next_state = next_state.view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        if done:
            break



with open('model/hyper_parameters.txt', 'w') as file:
    file.write(json.dumps(hyper_parameters))  # use `json.loads` to do the reverse
with open('model/reward_parameters.txt', 'w') as file:
    file.write(json.dumps(reward_parameters))  # use `json.loads` to do the reverse

torch.save(actorNet.state_dict(), "model/actor.pt")
torch.save(criticNet_1.state_dict(), "model/criticNet_1.pt")
torch.save(criticNet_2.state_dict(), "model/criticNet_2.pt")
torch.save(target_valueNet.state_dict(), "model/target_valueNet.pt")
