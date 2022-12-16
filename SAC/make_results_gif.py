import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import datetime
from PIL import Image

from itertools import count
from environment import GridWorldEnv
from training import init_model, select_action
from model import *
import json



sys.path.insert(0, '..')


if __name__ == '__main__':

    images = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "plots/"
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

    # initialize environment
    env = GridWorldEnv(render_mode=None, size=env_parameters['env_size'], num_obstacles=env_parameters['num_obstacles'])
    env.render_mode = "human"

    # initialize NN
    actorNet, criticNet_1, criticNet_2, valueNet, target_valueNet, memory = init_model()
    seed = feature_parameters['seed_init_value'] + 20
    # Load model
    actorNet.load_state_dict(torch.load(model_path + "actor.pt", map_location=device))
    actorNet.max_sigma = hyper_parameters['sigma_final']
    criticNet_1.load_state_dict(torch.load(model_path + "criticNet_1.pt", map_location=device))
    criticNet_2.load_state_dict(torch.load(model_path + "criticNet_2.pt", map_location=device))
    target_valueNet.load_state_dict(torch.load(model_path + "target_valueNet.pt", map_location=device))

    # env=GridWorldEnv(render_mode="human")
    i = 0
    while i < 10:  # run plot for X episodes to see what it learned
        i += 1
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
            # Select and perform an action
            action = select_action(state, actorNet)
            _, reward, done, _, _ = env.step(action)

            action_ = torch.tensor(action, dtype=torch.float, device=device)
            action_ = action_.view(1, 2)
            mu, sigma = actorNet(state)

            reward = torch.tensor([reward], device=device)

            rgb_array = env._render_frame_for_gif()
            print(rgb_array.shape)
            print(rgb_array)
            img = Image.fromarray(rgb_array)
            print(img)
            images.append(img)#.convert("P", dither=None))

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

    images_rev = images[::-1]
    images_full = images #+ images_rev

    t = datetime.datetime.now()
    time_path = ("_%s_%s_%s_%s-%s-%s" % (t.year, t.month, t.day, t.hour, t.minute, t.second))
    dir_path = data_path + "traj_gif" + time_path
    os.makedirs(dir_path, exist_ok=False)

    images_full[0].save(dir_path + "/traj_check.gif", format="GIF",
                        save_all=True, append_images=images_full[1:],
                        optimize=False, duration=300, loop=0)