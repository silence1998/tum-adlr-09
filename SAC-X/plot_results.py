import torch
import numpy as np
from itertools import count
import pickle

from environment import GridWorldEnv
from training import init_model, select_action, obstacle_sort, select_action_smooth, Scheduler

from model import *
import json

from util_sacx.q_table import QTable
from util_sacx.policy import BoltzmannPolicy

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = "1"#input("Select normal Model (0) OR Model with pretrain (1): ")
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
    env = GridWorldEnv(render_mode=None,
                       object_size=env_parameters['object_size'],  # TODO: change back to env_size to radius objects
                       num_obstacles=env_parameters['num_obstacles'],
                       window_size=env_parameters['window_size'])
    env.render_mode = "human"

    # initialize NN
    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model(hyper_parameters["input_dims"])
    seed = feature_parameters['seed_init_value']
    # Load model
    actorNet.load_state_dict(torch.load(model_path + "actor.pt", map_location=device))
    # actorNet.max_sigma = hyper_parameters['sigma_final']
    criticNet_1.load_state_dict(torch.load(model_path + "criticNet_1.pt", map_location=device))
    criticNet_2.load_state_dict(torch.load(model_path + "criticNet_2.pt", map_location=device))
    target_valueNet.load_state_dict(torch.load(model_path + "target_valueNet.pt", map_location=device))


    if feature_parameters['apply_environment_seed']:
        seed = 0  # feature_parameters['seed_init_value']
    action_history = deque(maxlen=feature_parameters['action_history_size'])

    task = 0
    i_episode = 0
    while i_episode < 10:  # run plot for 10 episodes to see what it learned
        List_Tau = []
        if i_episode == 0 or i_episode == 1:
            entropy_factor = hyper_parameters['entropy_factor']
            sigma_ = hyper_parameters['sigma_init']
        else:
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
            if t % sac_schedule.xi == 0:
                fuzzy_state = sac_schedule.caluculate_fuzzy_distance(obs_values)
                # task = random.choice(tasks) ## sac-u
                task = sac_schedule.schedule_task(List_Tau, fuzzy_state)  ## sac-q
                print(task)
                List_Tau.append(task)

            action = select_action(state, actorNet, task)
            if feature_parameters['action_smoothing']:
                action_history.extend([action])
                action = select_action_smooth(action_history)
            _, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward[task]], dtype=torch.float, device=device)

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
                break

            # Store the transition in memory
            action = np.array([action])
            action = torch.tensor(action, dtype=torch.float).to(actorNet.device)
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state




