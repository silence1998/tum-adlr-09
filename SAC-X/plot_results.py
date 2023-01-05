import torch
import numpy as np
from itertools import count
import pickle

from environment import GridWorldEnv
from training import init_model, select_action, Scheduler

from model import *
import json

from util_sacx.q_table import QTable
from util_sacx.policy import BoltzmannPolicy

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = "1" #input("Select normal Model (0) OR Model with pretrain (1): ")
    if m == "0":
        model_path = "model/"
    elif m == "1":
        model_path = "model_pretrain/"

    # Load the model parameters
    with open(model_path + 'env_parameters.txt', 'r') as file:
        env_parameters = json.load(file)
    with open(model_path + 'hyper_parameters.txt', 'r') as file:
        hyper_parameters = json.load(file)
    with open(model_path + 'reward_parameters.txt', 'r') as file:
        reward_parameters = json.load(file)
    with open(model_path + 'feature_parameters.txt', 'r') as file:
        feature_parameters = json.load(file)

    tasks = (0, 1, 2)
    sac_schedule = Scheduler(tasks)

    with open(model_path + "Q_task.pkl", "rb") as tf:
        sac_schedule.Q_task.store = pickle.load(tf)

    # initialize environment
    env = GridWorldEnv(render_mode=None, size=env_parameters['env_size'], num_obstacles=env_parameters['num_obstacles'])
    env.render_mode = "human"

    # initialize NN
    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model()
    seed = feature_parameters['seed_init_value']
    # Load model
    actorNet.load_state_dict(torch.load(model_path + "actor.pt", map_location=device))
    actorNet.max_sigma = hyper_parameters['sigma_final']
    criticNet_1.load_state_dict(torch.load(model_path + "criticNet_1.pt", map_location=device))
    criticNet_2.load_state_dict(torch.load(model_path + "criticNet_2.pt", map_location=device))
    target_valueNet.load_state_dict(torch.load(model_path + "target_valueNet.pt", map_location=device))

    # env=GridWorldEnv(render_mode="human")
    task = 1
    i = 0
    while i < 10:  # run plot for 10 episodes to see what it learned
        i += 1
        List_Tau = []
        if feature_parameters['apply_environment_seed']:
            env.reset(seed=seed)
            seed += 1
        else:
            env.reset()
        obs = env._get_obs()

        obs_values = [obs["agent"], obs["target"]]
        for idx_obstacle in range(env_parameters['num_obstacles']):
            obs_values.append(obs["obstacle_{0}".format(idx_obstacle)])
        state = torch.tensor(np.array(obs_values), dtype=torch.float, device=device)

        state = state.view(1, -1)
        for t in count():
            if t % sac_schedule.xi == 0:
                # task = random.choice(tasks) ## sac-u
                task = sac_schedule.schedule_task(List_Tau) ## sac-q
                List_Tau.append(task)
            # Select and perform an action
            action = select_action(state, actorNet, task)
            _, reward, done, _, _ = env.step(action)

            action_ = torch.tensor(action, dtype=torch.float, device=device)
            action_ = action_.view(1, 2)
            # mu, sigma = actorNet(state, task)
            print(task)
            print(actorNet(state, task))
            print(criticNet_1(state, action_, task))
            print(criticNet_2(state, action_, task))
            print(target_valueNet(state, task))

            reward = torch.tensor([reward[task]], device=device)
            print("reward=", reward)
            env._render_frame()
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
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state
            if done:
                break
