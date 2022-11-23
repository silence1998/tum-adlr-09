import numpy as np
import json
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn.functional as F

from model import *
from environment import *

from

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Used Sources:
https://www.gymlibrary.dev/content/environment_creation/
https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
"""

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

def select_action(state, actorNet):
    # state = torch.Tensor([state]).to(actorNet.device)
    actions, _ = actorNet.sample_normal(state, reparametrize=False)

    return actions.cpu().detach().numpy()[0]

def select_action_A_star(state):
    size = env.size
    grid = np.zeros((size, size))
    grid[state[2], state[3]] = 1
    # Start position
    StartNode = (state[0], state[1])
    # Goal position
    EndNode = (state[4], state[5])
    path = A_star.algorithm.algorithm(grid, StartNode, EndNode)
    if path == None:
        print("error: doesn't find a path")
        return None
    path = np.array(path)
    actions = np.zeros(((len(path) - 1), 2))
    for i in range(len(path) - 1):
        actions[i, :] = path[i + 1] - path[i]
    return actions


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

env_parameters = {
    'num_obstacles': 5,
    'env_size': 20  # size of the environment
}
env = GridWorldEnv(render_mode=None, size=env_parameters['env_size'])

hyper_parameters = {
    'input_dims': 4 + env_parameters['num_obstacles'] * 2,  # original position of actor, obstacle and target position
    'batch_size': 256,
    'gamma': 0.999,  # discount factor
    'target_update': 10,  # update target network every 10 episodes TODO: UNUSED?
    'alpha': 0.0003,  # learning rate for actor
    'beta': 0.0003,  # learning rate for critic
    'tau': 0.005,  # target network soft update parameter (parameters = tau*parameters + (1-tau)*new_parameters)
    'num_episodes': 20
}

def init_model():
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

    return actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory


if __name__ == "__main__":

    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model()

    steps_done = 0
    episode_durations = []
    for i_episode in range(hyper_parameters["num_episodes"]):
        # Initialize the environment and state
        env.reset()
        obs = env._get_obs()

        obs_values = [obs["agent"], obs["target"]]
        for idx_obstacle in range(env_parameters['num_obstacles']):
            obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
        state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

        state = state.view(1, -1)
        for t in count():
            # Select and perform an action
            action = select_action(state, actorNet)
            _, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float, device=device)

            # Observe new state
            obs = env._get_obs()
            if not done:
                obs_values = [obs["agent"], obs["target"]]
                for idx_obstacle in range(env_parameters['num_obstacles']):
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

    with open('model/reward_parameters.txt', 'w+') as file:
        file.write(json.dumps(env_parameters))  # use `json.loads` to do the reverse
    with open('model/hyper_parameters.txt', 'w+') as file:
        file.write(json.dumps(hyper_parameters))  # use `json.loads` to do the reverse
    with open('model/reward_parameters.txt', 'w+') as file:
        file.write(json.dumps(env.reward_parameters))  # use `json.loads` to do the reverse


    torch.save(actorNet.state_dict(), "model/actor.pt")
    torch.save(criticNet_1.state_dict(), "model/criticNet_1.pt")
    torch.save(criticNet_2.state_dict(), "model/criticNet_2.pt")
    torch.save(target_valueNet.state_dict(), "model/target_valueNet.pt")
