env_parameters = {
    'num_obstacles': 5,
    'env_size': 10  # size of the environment in one dimension (environment is square)
}

hyper_parameters = {
    'input_dims': 4 + env_parameters['num_obstacles'] * 2,  # original position of actor, target and obstacle positions
    'batch_size': 512,
    'gamma': 0.999,  # discount factor
    'target_update': 10,  # update target network every 10 episodes TODO: UNUSED if code for now
    'alpha': 0.0003,  # learning rate for actor
    'beta': 0.0003,  # learning rate for critic
    'tau': 0.005,  # target network soft update parameter (parameters = tau*parameters + (1-tau)*new_parameters)
    'entropy_factor': 0.5,  # entropy factor
    'num_episodes': 100,  # set min 70 for tests as some parts of code starts after ~40 episodes
}

feature_parameters = {
    'pretrain': True,  # pretrain the model
    'num_episodes_pretrain': 100,  # set min 70 for tests as some parts of code starts after ~40 episodes

    'select_action_filter': False,  # filter actions to be directed towards target
    'select_action_filter_after_episode': 100,  # start filtering after this episode

    'sort_obstacles': False,  # sort obstacles by distance to target

    'apply_environment_seed': False,  # apply seed to environment to have comparable results
    'seed_init_value': 3407,

    'plot_durations': True,  # plot durations of episodes
    'plot_sigma': False  # plot sigma of actor
}

reward_parameters = {
    # 'field_of_view': 5,  # see min_collision_distance
    # 'collision_weight': 0.3,
    # 'time_weight': 1,
    # the above are not used in the current version which is sparse reward based

    'action_step_scaling': 1,  # 1 step -> "2" grids of movement reach in x and y directions
    ### DENSE REWARDS ###
    'obstacle_avoidance': False,
    'obstacle_distance_weight': -1,
    'target_seeking': False,
    'target_distance_weight': 1,

    ### SPARSE REWARDS ###
    'target_value': 1,
    'collision_value': -1,

    ### SUB-SPARSE REWARDS ###
    'checkpoints': False,  # if true, use checkpoints rewards
    'checkpoint_distance_proportion': 1.0,
    'checkpoint_number': 5,  # make sure checkpoint_distance_proportion * "checkpoint_number" <= 1
    'checkpoint_value': 1.0,  # make sure checkpoint_value * checkpoint_number < 1

    'time': False,  # if true, use time rewards
    'time_penalty': 1,  # == penalty of -1 for "100" action steps

    'history_size': 20,  # size of history to check for waiting and consistency

    'waiting': False,  # if true, use waiting rewards # TODO: implement action history in step()
    'waiting_value': 1.0,  # make sure waiting_value < 1
    'max_waiting_steps': 20,  # make sure < history_size, punishment for waiting too long
    # threshold

    'consistency': False,  # if true, use consistency rewards # TODO: implement action history in step()
    'consistency_step_number': 1,  # make sure consistency_step_number < history_size
    'consistency_value': 1.0,  # make sure consistency_value * consistency_step_number < 1
    # threshold

    # fast moving etc. for sub actions for sparse rewards
}
