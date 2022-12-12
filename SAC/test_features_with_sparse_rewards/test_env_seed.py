env_parameters = {
    'num_obstacles': 5,
    'env_size': 10 # size of the environment in one dimension (environment is square)
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
    'entropy_factor_final': 0.3,
    'num_episodes': 250,  # set min 70 for tests as some parts of code starts after ~40 episodes
    'sigma_init': 2.0,
    'sigma_final': 0.5
}

feature_parameters = {
    'pretrain': True,  # pretrain the model
    'num_episodes_pretrain': 500,  # set min 70 for tests as some parts of code starts after ~40 episodes

    'action_smoothing': True,
    'action_history_size': 3,  # number of actions to remember for the action history

    'select_action_filter': False,  # filter actions to be directed towards target # TODO: last test
    'select_action_filter_after_episode': 70,  # start filtering after this episode

    'sort_obstacles': True,  # sort obstacles by distance to target

    'apply_environment_seed': False,  # apply seed to environment to have comparable results
    'seed_init_value': 3407,

    'plot_durations': False,  # plot durations of episodes
    'plot_sigma': False  # plot sigma of actor
}

reward_parameters = {
    # 'field_of_view': 5,  # see min_collision_distance
    # 'collision_weight': 0.3,
    # 'time_weight': 1,
    # the above are not used in the current version which is sparse reward based

    'action_step_scaling': 1,  # 1 step -> "2" grids of movement reach in x and y directions
    ### DENSE REWARDS ### # TODO: check after midterm
    'obstacle_avoidance': False,
    'obstacle_distance_weight': -0.01,
    'target_seeking': False,
    'target_distance_weight': 0.01,

    ### SPARSE REWARDS ###
    'target_value': 10,
    'collision_value': -50,

    ### SUB-SPARSE REWARDS ###
    'checkpoints': False,  # if true, use checkpoints rewards
    'checkpoint_distance_proportion': 0.1,  # distance proportion to environment size in 1 dimension
    'checkpoint_number': 5,  # make sure checkpoint_distance_proportion * "checkpoint_number" <= 1
    'checkpoint_value': 0.1,  # make sure checkpoint_value * checkpoint_number < 1

    'time': False,  # if true, use time penalty
    'time_penalty': -0.01,  # 0.01 == penalty of -1 for "100" action steps

    # Rewards below depend on action history
    'history': False,  # if true, use history
    'history_size': 15,  # >= hyper_parameters['action_history_size'] above
    # size of history to check for waiting and consistency
    # can be chosen bigger to have longer checks for below features

    'waiting': False,  # if true, use waiting rewards
    'waiting_value': 0.01,  # make sure waiting_value < 1
    'waiting_step_number_to_check': 5,  # number of steps to check for waiting (in history)
    # make sure waiting_step_number_to_check < history_size
    'max_waiting_steps': 10,  # make sure < history_size, punishment for waiting too long
    'waiting_penalty': -0.05,
    # threshold

    'consistency': False,  # if true, use consistency rewards
    'consistency_step_number_to_check': 3,  # number of steps to check for consistency (in history)
    # make sure consistency_step_number_to_check < history_size
    'consistency_value': 0.01,  # make sure consistency_value * consistency_step_number < 1
    # threshold

    # fast moving etc. for sub actions for sparse rewards
}
