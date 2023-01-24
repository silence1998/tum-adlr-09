"""
Used Sources:
https://www.gymlibrary.dev/content/environment_creation/
https://github.com/Farama-Foundation/gym-examples/blob/main/gym_examples/envs/grid_world.py
#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from itertools import count

from collections import namedtuple
from collections import deque

import torch
import torch.nn.functional as F

from model import ActorNetwork, CriticNetwork, ValueNetwork, ReplayMemory
from environment import GridWorldEnv

import A_star.algorithm
import parameters

import wandb


def optimize_model(entropy_factor):  # SpinningUP SAC PC: lines 12-14
    if len(memory) < hyper_parameters["batch_size"]:  # if memory is not full enough to start training, return
        return
    if len(memory_success) < hyper_parameters["batch_size"] * 3 / 4:
        transitions = memory.sample(hyper_parameters["batch_size"])
        batch = Transition(*zip(*transitions))
    ### Sample a batch of transitions from memory
    else:
        transitions = memory.sample(round(hyper_parameters["batch_size"] / 4))  # SpinningUP SAC PC: line 11
        transitions_success = memory_success.sample(round(hyper_parameters["batch_size"] * 3 / 4))
        batch = Transition(*zip(*transitions, *transitions_success))
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays

    ### SpinningUP SAC PC: line 12
    # Compute a mask of non-final states and concatenate the batch elements -> (1-d)
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    if not len(memory) < hyper_parameters["batch_size"]:
        ### Calculate average sigma per batch
        mu, sigma = actorNet.forward(state_batch)
        global average_sigma_per_batch
        average_sigma_per_batch.append(
            np.mean(sigma.detach().cpu().numpy(), axis=0))  # mean of sigma of the current batch

    value = valueNet(state_batch).view(-1)  # infer size of batch
    value_ = torch.zeros(hyper_parameters["batch_size"], device=device)
    value_[non_final_mask] = target_valueNet(non_final_next_states).view(-1)

    # Compute the target Q value -> Q_phi_target_1/2
    actions, log_probs = actorNet.sample_normal(state_batch, reparametrize=False)
    log_probs = log_probs.view(-1)
    q1_new_policy = criticNet_1.forward(state_batch, actions)
    q2_new_policy = criticNet_2.forward(state_batch, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)  # rhs SpinningUP SAC PC: line 12 big ()

    valueNet.optimizer.zero_grad()
    value_target = critic_value - entropy_factor * log_probs
    value_loss = 0.5 * F.mse_loss(value, value_target)
    value_loss.backward(retain_graph=True)
    valueNet.optimizer.step()

    # Update the target value network
    actions, log_probs = actorNet.sample_normal(state_batch,
                                                reparametrize=True)  # line 14 big () right term a_tilde_theta(s)
    log_probs = log_probs.view(-1)
    q1_new_policy = criticNet_1.forward(state_batch, actions)
    q2_new_policy = criticNet_2.forward(state_batch, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    actor_loss = entropy_factor * log_probs - critic_value
    actor_loss = torch.mean(actor_loss)

    wandb.log({"actor_loss": actor_loss})

    # print(str(i_episode) + " - " + str(actor_loss))
    # print(str(i_episode) + "-actor_loss: " + str(actor_loss.detach().cpu().numpy()))
    # too slow to switch to cpu everytime

    actorNet.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actorNet.optimizer.step()

    criticNet_1.optimizer.zero_grad()
    criticNet_2.optimizer.zero_grad()
    q_hat = reward_batch + hyper_parameters["gamma"] * value_  # SpinningUP SAC PC: line 12 -> calc y (target)
    q1_old_policy = criticNet_1.forward(state_batch, action_batch).view(-1)
    q2_old_policy = criticNet_2.forward(state_batch, action_batch).view(-1)
    critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)  # line 13 in s.u. pseudocode
    critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

    # Update Q-functions (critic) by one step of gradient descent # SpinningUP SAC PC: line 13
    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    criticNet_1.optimizer.step()  # line 13 in s.u. pseudocode
    criticNet_2.optimizer.step()


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take X episode averages and plot them too
    avg_last_X_episodes = 10
    if len(durations_t) >= avg_last_X_episodes:
        means = durations_t.unfold(0, avg_last_X_episodes, 1).mean(1).view(-1)
        # takes the average of the last X episodes and adds to the averages list
        # starting from Xth episode as average value slice needs X episodes for mean as given
        means = torch.cat((torch.zeros(avg_last_X_episodes - 1), means))  # pad with zeros for the first X episodes
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_sigma():
    global average_sigma_per_batch
    plt.figure(2)
    plt.clf()
    sigma_t = torch.tensor(np.array(average_sigma_per_batch), dtype=torch.float)
    sigma_tx = sigma_t[:, 0]
    sigma_ty = sigma_t[:, 1]
    plt.title('Training...')
    plt.xlabel('Batch')
    plt.ylabel('Sigma')
    plt.plot(sigma_t.numpy())
    # Take X episode averages and plot them too
    avg_last_X_batches = 100
    if len(sigma_t) > avg_last_X_batches:
        means_x = sigma_tx.unfold(0, avg_last_X_batches, 1).mean(1).view(-1)
        means_x = torch.cat((torch.zeros(avg_last_X_batches - 1), means_x))  # pad with zeros for the first X episodes
        means_y = sigma_ty.unfold(0, avg_last_X_batches, 1).mean(1).view(-1)
        means_y = torch.cat((torch.zeros(avg_last_X_batches - 1), means_y))  # pad with zeros for the first X episodes
        plt.plot(means_x.numpy())
        plt.plot(means_y.numpy())
        wandb.log({"means_x": means_x.numpy()[-1]})
        wandb.log({"means_y": means_y.numpy()[-1]})
    plt.pause(0.001)  # pause a bit so that plots are updated


def select_action(state, actorNet):
    # state = torch.Tensor([state]).to(actorNet.device)
    actions, _ = actorNet.sample_normal(state, reparametrize=False)

    return actions.cpu().detach().numpy()[0]


def select_action_smooth(action_history):
    action_history_ = np.array(action_history)
    return np.mean(action_history_, axis=0)


### The following code is unused, maybe useful for future work vvv
def select_action_filter(state, actorNet):
    """
    erases the actions which are directed away from the goal
    """
    # state = torch.Tensor([state]).to(actorNet.device)
    # 0,1: agent position, 2,3: target position
    delta_x = state[0, 2] - state[0, 0]
    delta_y = state[0, 3] - state[0, 1]
    actions, _ = actorNet.sample_normal(state, reparametrize=False)
    while actions[0, 0] * delta_x < 0 or actions[0, 1] * delta_y < 0:
        actions, _ = actorNet.sample_normal(state, reparametrize=False)
    return actions.cpu().detach().numpy()[0]


def action_selection(state, actorNet):
    if feature_parameters['select_action_filter']:
        if len(episode_durations) < feature_parameters['select_action_filter_after_episode']:
            action = select_action(state, actorNet)
        else:
            action = select_action_filter(state, actorNet)
    else:
        action = select_action(state, actorNet)
    return action


### The above code is unused, maybe useful for future work ^^^

def select_action_A_star(state, window_size, object_size):  # TODO does this work correctly in continuous space?
    ratio = window_size / (2 * object_size)
    size = int(ratio) + 2
    grid = np.zeros((size, size))
    # state = np.matrix.round(state, decimals=0).astype(int)
    # print("state: " + str(state))
    # print("Rounded Location Object 0: [" + str(int(np.ceil(state[4]))) + "," + str(int(np.ceil(state[5]))) + "]")
    for i in range(env_parameters['num_obstacles']):
        grid[round(state[4 + 4 * i] / (2 * object_size)), round(state[5 + 4 * i] / (2 * object_size))] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)) + 1, round(state[5 + 4 * i] / (2 * object_size))] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)), round(state[5 + 4 * i] / (2 * object_size)) + 1] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)) + 1, round(state[5 + 4 * i] / (2 * object_size)) + 1] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)) - 1, round(state[5 + 4 * i] / (2 * object_size))] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)), round(state[5 + 4 * i] / (2 * object_size)) - 1] = 1
        grid[round(state[4 + 4 * i] / (2 * object_size)) - 1, round(state[5 + 4 * i] / (2 * object_size)) - 1] = 1

    # Start position
    StartNode = (round(state[0] / (2 * object_size)), round(state[1] / (2 * object_size)))  # agent position
    # Goal position
    EndNode = (round(state[2] / (2 * object_size)), round(state[3] / (2 * object_size)))  # target position
    path = A_star.algorithm.algorithm(grid, StartNode, EndNode)
    if path == None or StartNode == EndNode:
        # print("error: doesn't find a path")
        return None
    path = np.array(path)
    actions = np.zeros(((len(path) - 1), 2))
    for i in range(len(path) - 1):
        actions[i, :] = path[i + 1] - path[i]
    return actions[0]


def obstacle_sort(obs):
    """
    Sorts the obstacles in the environment by their distance to the agent.
    :return: A list of obstacle indices sorted by their distance to the agent.
    """
    distances = []
    obs_temp = obs.copy()  # copy the dict elements in the env
    for idx_obstacle in range(env_parameters["num_obstacles"]):
        distances.append(
            np.sqrt(np.sum(np.power((obs_temp["agent"] - obs_temp["obstacle_{0}".format(idx_obstacle)][0:2]), 2))))

    idx_obstacle_sorted = np.argsort(distances)  # min to max
    num_obstacles = range(env_parameters["num_obstacles"])
    for i, j in zip(num_obstacles, idx_obstacle_sorted):
        obs["obstacle_{0}".format(i)] = obs_temp["obstacle_{0}".format(j)]
    return obs


def save_models():
    if feature_parameters['pretrain']:
        model_path = "model_pretrain/"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            print("created folder : ", model_path)

    else:
        model_path = "model/"
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
            print("created folder : ", model_path)

    with open(model_path + 'env_parameters.txt', 'w+') as file:
        file.write(json.dumps(env_parameters))  # use `json.loads` to do the reverse
    with open(model_path + 'hyper_parameters.txt', 'w+') as file:
        file.write(json.dumps(hyper_parameters))  # use `json.loads` to do the reverse
    with open(model_path + 'reward_parameters.txt', 'w+') as file:
        file.write(json.dumps(env.reward_parameters))  # use `json.loads` to do the reverse
    with open(model_path + 'feature_parameters.txt', 'w+') as file:
        file.write(json.dumps(feature_parameters))  # use `json.loads` to do the reverse

    print("Saving models ...")
    torch.save(actorNet.state_dict(), model_path + "actor.pt")
    torch.save(criticNet_1.state_dict(), model_path + "criticNet_1.pt")
    torch.save(criticNet_2.state_dict(), model_path + "criticNet_2.pt")
    torch.save(target_valueNet.state_dict(), model_path + "target_valueNet.pt")
    print("Done")


def init_model(input_dims=parameters.hyper_parameters["input_dims"]):
    # initialize NN
    n_actions = 2  # velocity in 2 directions
    actorNet = ActorNetwork(hyper_parameters["alpha"], input_dims, n_actions=n_actions,
                            name='actor', max_action=[1, 1], sigma=2.0)  # TODO max_action value and min_action value
    criticNet_1 = CriticNetwork(hyper_parameters["beta"], input_dims, n_actions=n_actions,
                                name='critic_1')
    criticNet_2 = CriticNetwork(hyper_parameters["beta"], input_dims, n_actions=n_actions,
                                name='critic_2')
    valueNet = ValueNetwork(hyper_parameters["beta"], input_dims, name='value')
    target_valueNet = ValueNetwork(hyper_parameters["beta"], input_dims, name='target_value')

    memory = ReplayMemory(feature_parameters['maxsize_ReplayMemory'])  # replay buffer size

    return actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory


# initialize hyper-parameters
hyper_parameters = parameters.hyper_parameters
feature_parameters = parameters.feature_parameters
env_parameters = parameters.env_parameters
test_parameters = parameters.test_parameters

if __name__ == "__main__":

    # wandb.init(project="SAC", entity="tum-adlr-09")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    env = GridWorldEnv(render_mode=None,
                       object_size=env_parameters['object_size'],
                       num_obstacles=env_parameters['num_obstacles'],
                       window_size=env_parameters['window_size'])

    # wandb_dict = {}
    # wandb_dict.update(env_parameters)
    # wandb_dict.update(hyper_parameters)
    # wandb_dict.update(feature_parameters)
    # wandb_dict.update(test_parameters)
    # print("dict: " + str(wandb_dict))
    # wandb.config.update(wandb_dict)

    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model()
    memory_success = ReplayMemory(feature_parameters['maxsize_ReplayMemory'])  # replay buffer size
    # wandb.watch(actorNet)
    # wandb.watch(criticNet_1)
    # wandb.watch(criticNet_2)
    # wandb.watch(valueNet)
    # wandb.watch(target_valueNet)

    episode_durations = []
    average_sigma_per_batch = []
    if feature_parameters['apply_environment_seed']:
        seed = feature_parameters['seed_init_value']
        print("Testing random seed: " + str(torch.rand(2)))
    env.render_mode = "human"
    if feature_parameters['pretrain']:
        for i_episode in range(feature_parameters['num_episodes_pretrain']):
            # print("Pretrain episode: " + str(i_episode))

            # Initialize the environment and state
            if feature_parameters['apply_environment_seed']:
                env.reset(seed=seed)
                seed += 1
            else:
                env.reset()

            obs = env._get_obs()
            if feature_parameters['sort_obstacles']:
                obs = obstacle_sort(obs)

            obs_values = np.array([obs["agent"], obs["target"]])
            for idx_obstacle in range(env_parameters['num_obstacles']):
                obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
            obs_values = np.array(obs_values).reshape(-1)

            state = torch.tensor(obs_values, dtype=torch.float, device=device)
            state = state.view(1, -1)

            t = 0
            for t in count():
                # Select and perform an action
                action = select_action_A_star(obs_values, env.window_size, env.radius)

                if action is None:
                    # print("error: doesn't find a path")
                    break
                t += 1
                action = action  # / env.reward_parameters['action_step_scaling']
                _, reward, done, _, _ = env.step(action)
                reward = torch.tensor([reward], dtype=torch.float, device=device)

                # Observe new state
                obs = env._get_obs()
                if not done:
                    if feature_parameters['sort_obstacles']:
                        obs = obstacle_sort(obs)
                    obs_values = np.array([obs["agent"], obs["target"]])
                    for idx_obstacle in range(env_parameters['num_obstacles']):
                        obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
                    next_state = torch.tensor(np.array(obs_values).reshape(-1),
                                              dtype=torch.float,
                                              device=device)
                    next_state = next_state.view(1, -1)
                else:

                    next_state = None

                # Store the transition in memory
                action_torch = torch.tensor(np.array([action]), dtype=torch.float).to(actorNet.device)
                memory.push(state, action_torch, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(hyper_parameters['entropy_factor'])
                if done:
                    episode_durations.append(t + 1)
                    if feature_parameters['plot_durations']:
                        plot_durations()
                    if feature_parameters['plot_sigma']:
                        if not len(memory) < hyper_parameters["batch_size"]:
                            plot_sigma()
                    if reward > 0:
                        # print("success")
                        for j in range(t):
                            memory_success.memory.append(memory.memory[-1 - j])
                    break

            target_value_params = target_valueNet.named_parameters()
            value_params = valueNet.named_parameters()

            target_value_state_dict = dict(target_value_params)
            value_state_dict = dict(value_params)

            for name in value_state_dict:
                value_state_dict[name] = hyper_parameters['tau'] * value_state_dict[name].clone() + \
                                         (1 - hyper_parameters['tau']) * target_value_state_dict[name].clone()
            target_valueNet.load_state_dict(value_state_dict)

            if i_episode % 25 == 0:
                actorNet.save_checkpoint()
                criticNet_1.save_checkpoint()
                criticNet_2.save_checkpoint()
                valueNet.save_checkpoint()
                target_valueNet.save_checkpoint()

                with open('tmp/sac/i_episode_pretrain.txt', 'w+') as file:
                    file.write(json.dumps(i_episode))

        print('Pretrain complete')

    if feature_parameters['apply_environment_seed']:
        seed = 0  # feature_parameters['seed_init_value']
    action_history = deque(maxlen=feature_parameters['action_history_size'])

    for i_episode in range(hyper_parameters["num_episodes"]):  # SpinningUP SAC PC: line 10

        print("Normal training episode: " + str(i_episode))

        entropy_factor = hyper_parameters['entropy_factor'] + i_episode * (
                hyper_parameters['entropy_factor_final'] - hyper_parameters['entropy_factor']) / (
                                 hyper_parameters["num_episodes"] - 1)

        sigma_ = hyper_parameters['sigma_init'] + i_episode * (
                hyper_parameters['sigma_final'] - hyper_parameters['sigma_init']) / (
                         hyper_parameters["num_episodes"] - 1)

        actorNet.max_sigma = sigma_

        # Initialize the environment and state
        if feature_parameters['apply_environment_seed']:
            env.reset(seed=seed)
            seed += 1
        else:
            env.reset()
        obs = env._get_obs()
        if feature_parameters['sort_obstacles']:
            obs = obstacle_sort(obs)
        obs_values = np.array([obs["agent"], obs["target"]])
        for idx_obstacle in range(env_parameters['num_obstacles']):
            obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
        state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

        state = state.view(1, -1)
        for t in count():  # every step of the environment
            # Select and perform an action
            action = select_action(state, actorNet)
            if feature_parameters['action_smoothing']:
                action_history.extend([action])
                action = select_action_smooth(action_history)
            _, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float, device=device)

            # Observe new state
            obs = env._get_obs()
            if not done:
                if feature_parameters['sort_obstacles']:
                    obs = obstacle_sort(obs)
                obs_values = np.array([obs["agent"], obs["target"]])
                for idx_obstacle in range(env_parameters['num_obstacles']):
                    obs_values = np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
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
            optimize_model(entropy_factor)
            if done:
                episode_durations.append(t + 1)
                if feature_parameters['plot_durations']:
                    plot_durations()
                if feature_parameters['plot_sigma']:
                    if not len(memory) < hyper_parameters["batch_size"]:
                        plot_sigma()
                if reward > 0:
                    print("success")
                    for j in range(t):
                        memory_success.memory.append(memory.memory[-1 - j])
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

        if i_episode % 25 == 0:
            actorNet.save_checkpoint()
            criticNet_1.save_checkpoint()
            criticNet_2.save_checkpoint()
            valueNet.save_checkpoint()
            target_valueNet.save_checkpoint()
            with open('tmp/sac/i_episode.txt', 'w+') as file:
                file.write(json.dumps(i_episode))

    print('Normal training complete')

    save_models()

    print('Complete')
