
import os
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

torch.manual_seed(3407)

class ReplayMemory(object):  # a memory buffer to store transitions

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
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
        self.fc2 = nn.ModuleList([nn.Linear(self.fc1_dims, self.fc2_dims) for i in range(6)])
        self.q = nn.ModuleList([nn.Linear(self.fc2_dims, 1) for i in range(6)])

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action, task: int):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2[task](action_value)
        action_value = F.relu(action_value)

        q = self.q[task](action_value)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.ModuleList([nn.Linear(self.fc1_dims, self.fc2_dims) for i in range(6)])
        self.v = nn.ModuleList([nn.Linear(self.fc2_dims, 1) for i in range(6)])

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, task):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2[task](state_value)
        state_value = F.relu(state_value)

        v = self.v[task](state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
                 fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac', sigma=2.0):
        super(ActorNetwork, self).__init__()
        self.max_sigma = sigma
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.zip')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.ModuleList([nn.Linear(self.fc1_dims, self.fc2_dims) for i in range(6)])
        self.mu = nn.ModuleList([nn.Linear(self.fc2_dims, self.n_actions) for i in range(6)])
        self.sigma = nn.ModuleList([nn.Linear(self.fc2_dims, self.n_actions) for i in range(6)])

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, task):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2[task](prob)
        prob = F.relu(prob)

        mu = self.mu[task](prob)
        sigma = self.sigma[task](prob)  # log_std

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=self.max_sigma)  # TODO: decaying sigma

        return mu, sigma

    def sample_normal(self, state, task, reparametrize=True):
        # SpinningUP SAC PC: line 12 -> big () right term
        # we set alpha from PC = 1 -> to modify the variance of the distribution (entropy) # line 12 -> big () right terms coeffiecient
        mu, sigma = self.forward(state, task)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            actions = probabilities.rsample()   # for entropy regularization
        else:
            actions = probabilities.sample()  # use the generated action

        action_sample = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)  # [-1,1] normalization
        log_probs = probabilities.log_prob(actions)  # log_prob of the generated action
        log_probs -= torch.log(1 - action_sample.pow(2) + self.reparam_noise)  # lower bound for probabilities  #
        log_probs = log_probs.sum(1, keepdim=True)  # sum over all actions

        return action_sample, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))




