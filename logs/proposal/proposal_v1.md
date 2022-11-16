[Overleaf Link](https://www.overleaf.com/read/cjfmqxvrrcqg)

# Trajectory Planning with Moving Obstacles

## Team 9

1. Objective

Motion planning plays an important role in autonomous systems. For instance, a car in autonomous driving mode requires trajectory planning in free space and drones need to avoid collision with obstacles. With the increasing capability in autonomous systems, motion planning in dynamic environments attracts more attention, especially for autonomous vehicles and drones for avoiding collision with moving obstacles. Now, there are lots of algorithms for static environments. For example, the Dijkstra algorithm deals well with grid based maps. However, it still lacks deep research in motion planning in dynamic environments which require changing optimal trajectories to keep up with the changes in the environment. Reinforcement learning (RL) is vital and suitable for dealing with dynamic environments. An agent based on RL can take a different required action after obtaining environmental information in each step. Thus, this project uses RL methods to solve motion planning problems with moving obstacles.

2. Related Work

There has been some research for application of deep learning on motion planning problems.  [MPNet](http://arxiv.org/abs/1907.06013) provides a general neural network structure for motion planning. However, it can only deal with static environments. \
In a normal RL algorithm, it is hard to design a reward function, which should guide agents to learn from the trials and errors and balance reward and punishment. [Scheduled Auxiliary Control](https://arxiv.org/pdf/1802.10567.pdf) (SAC-X) focuses on problems with sparse reward signals and using the key idea of active (learned) scheduling and execution of auxiliary policies for new complex tasks. This research can help us to deal with difficulty in reward function in a complex dynamic environment. \
[Meta-Reinforcement Learning](https://arxiv.org/abs/1903.08254) (Meta-RL) is a method which can transfer experience from old tasks to new tasks. For dynamic environments, an agent cannot learn all situations. Thus,  it is important for an agent to generalize its knowledge. Kate et al. combines meta-RL with probabilistic distribution and apply it successfully for robot grasp problems.

3. Technical Outline

We will start by assuming that the agent has full knowledge of the environment, including position and velocity of the obstacles. But we will not consider dynamic constraints on agents as we will focus more on the planning instead of the representation. \
First, we establish a simple 2D motion planning problem in a dynamic environment with moving obstacles and modifying it.  \
Second, we will apply the RL algorithms from the related work in PyTorch, with a continuous action space on this problem and compare these algorithms performances in our dynamic environment. We will start with the SAC (Soft Actor Critic) algorithm as it is a standard baseline for robots with continuous action spaces and it is sample efficient. Afterwards we will improve on the performance with Meta-RL and/or the SAC-X variant and test their results with the SAC baseline. \
Lastly, we will improve on the best algorithm according to our results by considering some uncertainty of the moving obstacles, where we will add some noise on position and velocity of obstacles and optionally planning for a partially observed dynamic environment. 

## Questions
1. We would like to ask if you have any feedback for our project proposal? We also would like to ask some more specific questions
2. We are not sure of the feasibility of using a continuous action space where the actions are the direction and the velocity of our robot and wanted your take on it? For example, would a continuous action space cause lack of computation power problems in sampling? 
3. As we both are more interested in autonomous driving we want to focus on a 2D environment and make our algorithm better and consider environmental factors. Would this be enough improvement for a strong proposal as stated in the project proposal guidelines as we are giving up the scaling to a  3D environment?
4. We were not exactly sure how we would be able to compare the result of the algorithms we use as their performance results are given in a static environment. Even if we compare them in our dynamic environment it would mean we have no benchmark for the performance of our motion planning RL Algorithm so that we can know if we improved. Can you maybe provide us with a paper that we can use as a benchmark or expand the idea of comparing the return/reward-iteration/timestep information from the related work? 
    1. I would recommend to use SAC as a baseline for your problem. This is the kind of standard for robotics because it deals with continuous action spaces and it is sample efficient. 
    2. Try to solve the navigation problem first with SAC and later implement on top of SAC the Meta-RL 
    3. and/or the SAC-X variant and test their results with the SAC baseline.
    4. This is already a lot of work. Check out the open ai[ https://spinningup.openai.com/en/latest/](https://spinningup.openai.com/en/latest/)
    5. as well as[ https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/) . Here you will find some implementations of SAC which you can use as your starting point for implementing the Meta-RL and/or SAC-X algorithm. In case you use open-source code, please make sure that you cite it properly and mark your own contribution in your implementation visible. 
    6. You wrote: "Now, there are lots of algorithms for static environments. However, it still lacks deep research in motion planning in dynamic environments which require changing optimal trajectories to keep up with the changes in the environment. " -> Be a bit more specific which algorithms you mean that can handle static environments, e.g. Dijkstra etc.