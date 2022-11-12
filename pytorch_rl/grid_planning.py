import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import gym
from gym import spaces
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # TODO actionspace shoubled be continuous and bounded in [-3, 3]
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

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

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action_step):
        action_step=np.round(2*action_step)
        original_position = self._agent_location
        self._agent_location = self._agent_location + action_step

        if self._agent_location[0] < 0 or self._agent_location[1] < 0 or \
                self._agent_location[0] > self.size - 1 or self._agent_location[1] > self.size - 1:
            terminated = True  # collide with wall
            reward = -5
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            reward = 10
        else:
            original_distance = math.sqrt((original_position[0] - self._target_location[0]) ** 2 + (
                        original_position[1] - self._target_location[1]) ** 2)
            distance = math.sqrt((self._agent_location[0] - self._target_location[0]) ** 2 + (
                        self._agent_location[1] - self._target_location[1]) ** 2)
            reward = (original_distance - distance) - 1
            # 0.4 leads the agent to not learn the goal fast enough, -1 is to avoid the agent to stay at the same place and accumulates over time

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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims)
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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
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
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=0.5)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action_sample = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action_sample.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action_sample, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# initialize hyperparameters
env = GridWorldEnv(render_mode=None, size=20)
input_dims=4
BATCH_SIZE = 1024
GAMMA = 0.999
TARGET_UPDATE = 10
alpha=0.0003
beta=0.0003
tau=0.005


# initialize NN
n_actions = 2 # velocity in 2 direction
actorNet = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                        name='actor', max_action=[1,1]) # TODO max_action value and min_action value
criticNet_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                            name='critic_1')
criticNet_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                            name='critic_2')
valueNet = ValueNetwork(beta, input_dims, name='value')
target_valueNet = ValueNetwork(beta, input_dims, name='target_value')

memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    #state = torch.Tensor([state]).to(actorNet.device)
    actions, _ = actorNet.sample_normal(state, reparameterize=False)

    return actions.cpu().detach().numpy()[0]


episode_durations = []


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
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

    value = valueNet(state_batch).view(-1)
    value_ = torch.zeros(BATCH_SIZE, device=device)
    value_[non_final_mask] = target_valueNet(non_final_next_states).view(-1)

    actions, log_probs = actorNet.sample_normal(state_batch, reparameterize=False)
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

    actions, log_probs = actorNet.sample_normal(state_batch, reparameterize=True)
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
    q_hat = reward_batch + GAMMA * value_
    q1_old_policy = criticNet_1.forward(state_batch, action_batch).view(-1)
    q2_old_policy = criticNet_2.forward(state_batch, action_batch).view(-1)
    critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
    critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    criticNet_1.optimizer.step()
    criticNet_2.optimizer.step()

num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    obs = env._get_obs()
    state = torch.tensor([obs["agent"], obs["target"]], dtype=torch.float, device=device)
    state = state.view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        obs = env._get_obs()
        if not done:
            next_state = torch.tensor([obs["agent"], obs["target"]], dtype=torch.float, device=device)
            next_state = next_state.view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        action=[action]
        action=torch.tensor(action, dtype=torch.float).to(actorNet.device)
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
        value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                 (1 - tau) * target_value_state_dict[name].clone()
    target_valueNet.load_state_dict(value_state_dict)

print('Complete')

env.render_mode = "human"

# env=GridWorldEnv(render_mode="human")
i = 0
while i < 3:
    i += 1
    env.reset()
    obs = env._get_obs()
    state = torch.tensor([obs["agent"], obs["target"]], dtype=torch.float, device=device)
    state = state.view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _, _ = env.step(action)

        action_=torch.tensor(action, dtype=torch.float, device=device)
        action_=action_.view(1,2)
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
            next_state = torch.tensor([obs["agent"], obs["target"]], dtype=torch.float, device=device)
            next_state = next_state.view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        if done:
            break
