env_parameters = {
    'num_obstacles': 5,
    'object_size': 20,  # radius of every element in the environment
    'window_size': 512,  # use powers of 2 for better performance
    'action_step_scaling': 20,  # obstacles are between 0 and 1, this way we get a slower agent
    'obstacle_step_scaling': 5,
    'delta_T': 1,  # scaling velocities w/o changing the -1,1 internal for them in animation
    'render_fps': 24  # fps for rendering the environment
}

hyper_parameters = {
    'input_dims': 6 + env_parameters['num_obstacles'] * 4,  # original position of actor, target and obstacle positions and obstacle velocities
    'batch_size': 4096,
    'gamma': 0.999,  # discount factor
    'target_update': 10,  # update target network every 10 episodes TODO: UNUSED if code for now
    'alpha': 0.0003,  # learning rate for actor
    'beta': 0.0003,  # learning rate for critic
    'tau': 0.005,  # target network soft update parameter (parameters = tau*parameters + (1-tau)*new_parameters)
    'entropy_factor': 0.5,  # entropy factor
    'entropy_factor_final': 0.5,
    'num_episodes': 7000,  # set min 70 for tests as some parts of code starts after ~40 episodes

    'sigma_init': 2.0,
    'sigma_final': 2.0
}

feature_parameters = {
    'pretrain': True,  # pretrain the model
    'num_episodes_pretrain': 3000,  # set min 70 for tests as some parts of code starts after ~40 episodes
    'maxsize_ReplayMemory': 100000,
    'action_smoothing': True,

    'action_history_size': 3,  # number of actions to remember for the action history

    'select_action_filter': True,  # filter actions to be directed towards target # TODO: last test
    'select_action_filter_after_episode': 70,  # start filtering after this episode

    'sort_obstacles': True,  # sort obstacles by distance to target

    'apply_environment_seed': True,  # apply seed to environment to have comparable results
    'seed_init_value': 3407,

    'plot_durations': True,  # plot durations of episodes
    'plot_sigma': True,  # plot sigma of actor
    'scheduler_period': 2
}

reward_parameters = {

    ### DENSE REWARDS ###  # TODO: Keep it False we dont want Dense rewards
    'obstacle_avoidance_dense': False,
    'obstacle_distance_weight': -0.01,
    'target_seeking_dense': False,
    'target_distance_weight': 0.01,


    'total_step_limit': 1000,
    'step_limit_reached_penalty': -0.01,

    'time': True,  # if true, use time penalty
    'time_penalty': -0.01,  # 0.01 == penalty of -1 for "100" action steps

    ### SUPER SPARSE REWARDS ###
    'target_value': 1,
    'collision_value': -5,

    ### SUB-SPARSE REWARDS ###
    'collision_prediction': True,
    'collision_prediction_penalty': -0.03,  # TODO add wall as collision as well

    'predictive_obstacle_avoidance': True,
    'obstacle_proximity_penalty': -0.01,

    'checkpoints': True,  # if true, use checkpoints rewards
    'checkpoint_distance_proportion': 0.1,  # distance proportion to environment size in 1 dimension
    'checkpoint_number': 5,  # make sure checkpoint_distance_proportion * "checkpoint_number" <= 1
    'checkpoint_value': 0.1,  # make sure checkpoint_value * checkpoint_number < 1



    # Rewards below depend on action history
    'history': True,  # if true, use history
    'history_size': 15,  # >= hyper_parameters['action_history_size'] above
    # size of history to check for waiting and consistency
    # can be chosen bigger to have longer checks for below features

    'movement_tolerance': 0.1,  # position change threshold to define that an element is not moving

    'waiting': True,  # if true, use waiting rewards
    'waiting_value': 0.01,  # make sure waiting_value < 1
    'waiting_step_number_to_check': 5,  # number of steps to check for waiting (in history)
    # make sure waiting_step_number_to_check < history_size
    'max_waiting_steps': 10,  # make sure < history_size, punishment for waiting too long
    'waiting_penalty': -0.01,  # TODO: dont use negative rewards for auxiliary rewards
    # threshold


    'consistency': True,  # if true, use consistency rewards
    'consistency_step_number_to_check': 3,  # number of steps to check for consistency (in history)
    # make sure consistency_step_number_to_check < history_size
    'consistency_value': 0.01,  # make sure consistency_value * consistency_step_number < 1
    # threshold

    # fast moving etc. for sub actions for sparse rewards
}

