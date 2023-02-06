import torch
import numpy as np
from itertools import count

from environment import GridWorldEnv
from training import init_model, select_action

from model import *
from training import *
import json

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = input("Select normal Model (m) OR \n \
    Model with pretrain (mp) OR \n \
    Tests with number (#): ")
    #m = "1"
    if m == "m":
        model_path = "model/"
    elif m == "mp":
        model_path = "model_pretrain/"
    elif m == "0":
        model_path = "test_models/Sa-t0/"
    elif m == "1":
        model_path = "test_models/Sa-t1/"
    elif m == "10":
        model_path = "test_models/Sa-t10/"
    elif m == "11":
        model_path = "test_models/Sa-t11/"
    elif m == "12":
        model_path = "test_models/Sa-t12/"
    elif m == "3":
        model_path = "test_models/Sa-t3/"
    elif m == "4":
        model_path = "test_models/Sa-t4/"
    elif m == "5":
        model_path = "test_models/Sa-t5/"
    elif m == "6":
        model_path = "test_models/Sa-t6/"
    elif m == "7":
        model_path = "test_models/Sa-t7/"
    elif m == "9":
        model_path = "test_models/Sa-t9/"

    # Load the model parameters
    with open(model_path + 'env_parameters.txt', 'r') as file:
        env_parameters = json.load(file)
    with open(model_path + 'hyper_parameters.txt', 'r') as file:
        hyper_parameters = json.load(file)
    with open(model_path + 'reward_parameters.txt', 'r') as file:
        reward_parameters = json.load(file)
    with open(model_path + 'feature_parameters.txt', 'r') as file:
        feature_parameters = json.load(file)



    # initialize environment
    env = GridWorldEnv(render_mode=None, object_size=env_parameters['object_size'], num_obstacles=env_parameters['num_obstacles'])
    # env.render_mode = "human"

    # initialize NN
    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model()
    seed = feature_parameters['seed_init_value']
    # Load model
    actorNet.load_state_dict(torch.load(model_path + "actor.pt", map_location=device))
    actorNet.max_sigma = hyper_parameters['sigma_final']
    criticNet_1.load_state_dict(torch.load(model_path + "criticNet_1.pt", map_location=device))
    criticNet_2.load_state_dict(torch.load(model_path + "criticNet_2.pt", map_location=device))
    target_valueNet.load_state_dict(torch.load(model_path + "target_valueNet.pt", map_location=device))

    init_seed = 0  # unseen envs
    #seed = feature_parameters['seed_init_value']  # for seen envs
    actual_reward = []
    issuccess_ = []
    actual_step = []
    i = 0
    seed = init_seed
    while i < 100:  # run plot for 100 episodes to see what it learned
        i += 1
        seed += 1
        env.reset(seed=seed)
        obs = env._get_obs()

        obs_values = np.append(obs["agent"], obs["target"])
        for idx_obstacle in range(env_parameters['num_obstacles']):
            np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
        state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

        state = state.view(1, -1)
        for t in count():
            # Select and perform an action
            action = select_action(state, actorNet, task)
            _, reward, done, _, _ = env.step(action)

            action_ = torch.tensor(action, dtype=torch.float, device=device)
            action_ = action_.view(1, 2)
            mu, sigma = actorNet(state)
            # print(actorNet(state))
            # print(criticNet_1(state, action_))
            # print(criticNet_2(state, action_))
            # print(target_valueNet(state))

            reward = torch.tensor([reward], device=device)
            env._render_frame()
            # Observe new state
            obs = env._get_obs()
            if not done:
                obs_values = np.append(obs["agent"], obs["target"])
                for idx_obstacle in range(env_parameters['num_obstacles']):
                    np.append(obs_values, obs["obstacle_{0}".format(idx_obstacle)])
                next_state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

                next_state = next_state.view(1, -1)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            if done:
                reward_ = reward.cpu().detach().numpy()[0]
                actual_reward.append(reward_)
                actual_step.append(t)
                if reward > 0:
                    issuccess_.append(1)
                else:
                    issuccess_.append(0)
                break
            elif t >= 500:
                issuccess_.append(0)
                actual_reward.append(0)
                actual_step.append(t)
                break

    # print(issuccess_)
    # print(actual_reward)
    print("accuracy=", np.sum(issuccess_) / len(issuccess_))
    print("mean_reward=", np.mean(actual_reward))
    print(actual_reward)
    print("mean_step=", np.mean(actual_step))
    print(actual_step)