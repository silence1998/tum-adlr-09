
import numpy as np
from itertools import count


from environment import GridWorldEnv
from training import hyper_parameters, select_action, env_parameters
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GridWorldEnv(render_mode=None, size=env_parameters['env_size'])

env.render_mode = "human"


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

# load model
torch.load("model/actor.pt", map_location=device)
torch.load("model/criticNet_1.pt", map_location=device)
torch.load("model/criticNet_2.pt", map_location=device)
torch.load("model/target_valueNet.pt", map_location=device)

memory = ReplayMemory(10000)  # replay buffer size

steps_done = 0

# env=GridWorldEnv(render_mode="human")
i = 0
while i < 3:  # run plot for 3 episodes to see what it learned
    i += 1
    env.reset()
    obs = env._get_obs()

    obs_values = [obs["agent"], obs["target"]]
    for idx_obstacle in range(env_parameters['num_obstacles']):
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
