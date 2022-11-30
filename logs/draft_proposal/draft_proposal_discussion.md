# Possible Projects 

## 2 Solving Complex Sparse Reinforcement Learning Tasks
* When defining Reinforcement Learning Tasks for robots, it is often desirable to stick to sparse rewards in order to avoid reward shaping. Not only does is it ease the setup since a task can be solely defined by its final desired outcome, but the optimal policy can be found ”outside of the box” without a human prior given through the reward function. Unfortunately, in big state spaces random exploration is not able to find these sparse success signals. Therefore, Riedmiller et al. [17] introduce SAC-X.
    * Implement the algorithm 1
    * Investigate how the presented method can be used for finer and more dexterous object manipulation e.g. with a hand.
    * Another option is, to apply the method to an path planning agent which often has to recover from dead-end-situations in static environments
    * [[1802.10567] Learning by Playing - Solving Sparse Reward Tasks from Scratch](http://arxiv.org/abs/1802.10567) 


## 3 Non-sequential Reinforcement Learning for Hard Exploration Problems
* Another way of dealing with sparse reward signals is to utilize “expert demonstrations” in promising regions of the state space. In the absence of experts ,Blau et al. [1] propose to first generate successful trajectories by RRT-based planning algorithms and then initialize a policy using that data. Based on this idea, you could experiment with:
* Implementing different heuristics for choosing states to explore from (see e.g. Ecoffet et al. [4]) or implementing an entirely different planning algorithm.
* Coupling the policy learning with the RRT-based data collection e.g. by sampling starting states according to RRT and executing actions according  to the current policy
* Modify the code and try it for different robots and environments.
* Is this expert knowledge necessary or can this also be achieved with a well designed curriculum?
* Look at modern approaches to represent the environment in which the robot moves (ie. Basis Points Set Prokudin et al. [14]; PointNet for Motion Planning Strudel et al. [21])
* [[2106.09203] Learning from Demonstration without Demonstrations](https://arxiv.org/abs/2106.09203) 
* [[1901.10995] Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995) 
* [[1908.09186] Efficient Learning on Point Clouds with Basis Point Sets](http://arxiv.org/abs/1908.09186) 
* [[2008.11174] Learning Obstacle Representations for Neural Motion Planning](http://arxiv.org/abs/2008.11174) 


##  4 Trajectory Planning with Moving Obstacles
* Drones not only have to plan flight paths through static environments, but also avoid collisions with dynamic objects. To learn such trajectories, a suitable encoding of the changing environment is crucial. Start with the Basis Points Set Prokudin et al. [14] and extend it to dynamic environments. Use this representation for neural motion planning Qureshi et al. [15].
* Come up with a state representation for dynamic environments.
* Set up a simple 2D (and later 3D) environment in which an agent can navigate through moving obstacles.
* Use RL to plan optimal trajectories in this environment.
* Optional: Extend the method to work with uncertainties in the motion prediction of the collision objects
* [[1907.06013] Motion Planning Networks: Bridging the Gap Between Learning-based and Classical Motion Planners](http://arxiv.org/abs/1907.06013) 
    * [https://sites.google.com/view/mpnet/home](https://sites.google.com/view/mpnet/home) 
* [[1908.09186] Efficient Learning on Point Clouds with Basis Point Sets](http://arxiv.org/abs/1908.09186) 


# 22-10-22 10:30 Call Notes
* The above two papers are missing RL
    * MPNet uses a Conv AE
    * [Ahmed Qureshi's talk on Motion Planning Networks](https://www.youtube.com/watch?v=x5hf-gjdQaA)
    * [Harnessing Reinforcement Learning for Neural Motion Planning](https://arxiv.org/pdf/1906.00214.pdf) 
         * DDPG: deep deterministic policy gradient
    * ... be faster than traditional dynamic programming and this is really nice paper with some proofs you should check out and then there is another paper from the same group which incorporates reinforcement learning to plan in narrow passage spaces using under the umbrella of neural motion planning so ...
    * 26:44
    * ... we just test if it is it was able to find it or not and in some cases where it fails is i think especially the narrow passage cases where people have extended it by incorporating reinforcement learning only for those like subspaces where they explore a little bit and find you the neural motion planning policy.
* How to estimate velocity in latent space? 
    * Get the position first and use kalman filter to obtain velocity
    * Use end to end NN to embedding position and velocity in latent space
* What can we do now:
    * Assuming that we have known the velocity and position of obstacles
    * Constructing a simple 2D simulation environment
* Alternative Project:
    * 3 Non-sequential Reinforcement Learning for Hard Exploration Problems
    * 2 Solving Complex Sparse Reinforcement Learning Tasks \

* Next Meeting 14:00 Sunday 2022-10-23 
## TODO
   * Reread the RL in TiAI 
   * Read slides for RL in ADL4R: prioritise Deepmind lecture
   * If time search for papers 
   * 3 Ideas for Project4
   * 1 short idea each for project 2,3


# 22-10-22 14:00 Call Notes
* DeepMind lecture
* s8
* s53
* s57
* s80


## 4th Topic Steps
* Come up with a state representation for dynamic environments.
    * Ego drone / collision objects trajectories 
        * Position
            * Point cloud for position
        * Velocity
            * Point cloud difference with time increments	
    * Representation types: Pointcloud, Voxels, BPS 
    * States will be given to the encoder network at each time step
* Set up a simple 2D (and later 3D) environment in which an agent can navigate through moving obstacles.
    * Num param according to 
        * 2D: 6 dim each agent 
            * Position: x, y, theta
            * Velocity: v_x ,v_y, w(angular)
        * 3D: 12 dim each agent 
            * Position: x, y, z, yaw, pitch, roll
            * Velocity: v_x ,v_y, v_z, w_x, w_y, w_z
* Use RL to plan optimal trajectories in this environment.
    * Learning Method:
        * MPNet
    * RL Method
        * Off-policy actor critic policy based?
    * Train all of them
    * Test it
    * Compare
* Optional: Extend the method to work with uncertainties in the motion prediction of the collision objects
    * Noise of collision objects states
        * Gaußian noise representation for the noise to predict
    * Influence of unseen obstacle
        * People behind a truck, for example \

* Simple Test Environment:
    * 1 square robot, 1 dynamic obstacle , no rotation

## TODO
1. Find methods with code like MPNet
2. State representation for dynamic environments 
    1. Check any other popular ones in literature (3 above)
        1. Pointcloud, voxel, latentspace
3. Learn possible RL methods
    2. From lecture 
    3. From papers with code
4. Check the deadline for team registration email
    4. 27.07. 24:00


# 26-10-22 Wed 19:00-20:30 Call Notes
* Papers with Code
    * With Tensorflow
        * [Reinforcement Learning through Asynchronous Advantage Actor-Critic on a GPU | Papers With Code](https://paperswithcode.com/paper/reinforcement-learning-through-asynchronous#code)
        * [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning | Papers With Code](https://paperswithcode.com/paper/motion-planning-among-dynamic-decision-making#code)
            * Mostly related to our project
            * Code are basically on tf
            * It directly uses obstacles’ position and velocity. We should use the original point cloud to train NN. 
        * [Deep Reinforcement learning for real autonomous mobile robot navigation in indoor environments | Papers With Code](https://paperswithcode.com/paper/deep-reinforcement-learning-for-real)


## TODO
1. Clear up RL method for Topic#4
    1. [https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html) 
2. 1 short idea each for project 2,3 -> skim the relevant papers
3. Write draft for topics #2,#3
    3. Doesn’t have to be long only mention them so that in the case of 
4. Delete point cloud from proposal ()
5. Send Email


# 21-10-22 Mon 10.00
## DISCUSSION
1. Isaac, Gym, Gazebo(ROS)
2. Robot arm movement with 3DoF as an easy start
3. Box colliders
4. Have something running for MidTerm presentation
5. Prewritten - PPO algo (Optinal if time) -> Our Future Work PROPOSAL 
6. RL FOCUSED:
    1. Simple idea 
    2. Moving circles with collision radius
    3. Random movement path
    4. Most important part is to write the Env by yourself
        1. For a faster env use a prewritten one!
    5. Start with static case
    6. THINKING: even simple representation have problems, when scaled
    7. Valid Value Function
    8. Optimise RL Algo
7. REPRESENTATION FOCUS:
    1. BPS for dynamic would be interesting nobody done it before
    2. Representation PointNet, Hierarchical Pointcloud
        1. PointNet can be easy used for dynamic env as well
    3. Model a LIDAR sensor
    4. 2D scenario SAC Algorithm and stick to it
    5. Local of full global view?
    6. Write the papers in the related work PROPOSAL
9. PITFALLS:!!


## WE CHOSE RL FOCUSED AUTONOMOUS CAR SCENARIO:
1. Framework
    1. Isaac, Gym, Gazebo(ROS)
2. We segmented and got the traj of objects
    1. Box colliders/Circle collider
    2. Fixed obejcts
    3. Moving objects
3. Meta-RL(Hierarchical RL) to train subtasks for dynamic env
    1. Different scenarios in 
    2. PEARL
    3. 6AC-X 
4. Think about subtasks in the scenario and check if improvement to regular SAC (i have data this is the goal this is the optimal way in static env)
5. We would need a long memory for traj’s
6. [[1903.08254] Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/abs/1903.08254)
    1. For each task that we find use a latent variable Z to sample on these
    2. We train on these
    3. Then in a more complex environment
        1. We train the policy again 
        2. And we infer the latent variable that conditions the policy
        3. This way we dont get a network to choose a subtask but we try to learn the subtasks first 
        4. Hindsight replay buffers
            1. When we learn the off policy RL Algo
            2. I didn’t actually achieve the goal but this might have been
            3. Sparse reward could be a problem in SAC!!!!! So it would be better with a continuous reward like distance to goal
    4. Compare SAC and SAC-X
    5. OpenAI Gym Framework is popular, but we are aloowed to write our own env
        1. Gym allows pluging in other algos easier	 \

    6. Steps
        1. Implement 2d env
            1. Take circles, take collison radius
            2. Have a continous collision detection check
                1. With my current velocity project our position in next steps no prediction
            3. No rectangles needs too many collision check
        2. Implement sac	
            1. Get over pitfulls
        3. Implement Sac + Meta Learning
        4. Compare both
        5. If we use phyton the simulation could be kind of slow
            1. Use Torchscript (good for GPU with On-policy PPO[Felix doesnt have so much experience]) to be able to parallelize it
            2. PPO
                1. Start with exploring
                2. But quickly starts exploiting one specific policy 
                3. But the algo is too straight forward
                4. It was really hard for a previous group to find a working reward function with PPO in a Project
                    1. Trying different algos for dyna env and compared them SAC was best
        6. If we stay on the CPU it can pay of as using it is cheap in comparison to GPU machines 
            1. Benchmark the simulation, how many update steps per iteration  
            2. Numpy, numba (can be compile down to C), jax, julia(for really fast simulations) 
        7. Another group worked on the Architecture so Felix can help there as well


## ORGA
1. Regular weekly meeting
2. Find a good slot propose several
3. Piazza for chatting


## PROPOSAL
1. Circle bot turtle bot in static and dynamic environment
2. 2-3 papers are enough 
3. Communication during week
4. Not the last day

## TODO
1. Read [Meta-RL](https://arxiv.org/abs/1903.08254) first, if time [Learning by Playing](https://arxiv.org/pdf/1802.10567.pdf)
2. Look at the frameworks to choose one
    1. Isaac, Gym, Gazebo(ROS)


## 02-11-22 Wed 14:00 Call Notes

## Decision
* Gym Environment
* No 3D

# Questions
1. We want to only do 2D as 3D is just scaling the problem, and we want to focus on the algorithm more instead of the simulation or visualisation
2. We can easily use this and make the environment dynamic and more complex:
    1. [https://www.gymlibrary.dev/environments/box2d/car_racing/](https://www.gymlibrary.dev/environments/box2d/car_racing/)
3. Algorithms to use from papers:
    1. SAC
    2. SAC-X [Learning by Playing](https://arxiv.org/pdf/1802.10567.pdf)
    3. [Meta-RL](https://arxiv.org/abs/1903.08254) 
4. Train and compare them on our dynamic environment(improvement) 
5. Benchmark
    1. Comparison in benchmark tasks such as in Meta-RL paper?
    2. Or use the Meta-RL Algo/SAC-X in our environment to compare it to our algorithm
    3. Comparison to a classical algorithms for dynamic environments paper?
    4. Is the comparison always done by reward/return plots over timstepse/iteration, if so is it important that we use the same environment for both compared algorithms
6.  (? here is finite action space or continuous action space (how to deal with continuous action space))	
7. Comparison


## TODO
1. Finish proposal
2. Ask questions on piazza