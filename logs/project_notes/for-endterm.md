### Last Points

#### ContSAC
- test older features from midterm -> sparse_rewards:
  - action smoothing
    - worked well 
  - time penalty
    - worked well
  - checkpoint rewards
    - was inconclusive
  - pretrain method
    - worked well
  - obstacle sorting
    - worked well 
  - env gen random seed
    - performed bad as if we pretrain in env#1 and normal train in env#1 it overfits and cant generalize
    - bad sample balance -> fixed by MO's weighting
    - [X] should be performing correctly with the "small fixes to seeding" commit

- new features
  - weighting the samples 
    - 2nd importance after pretraining  
  - continious -> see title
  - dynamic env.

- stupid agent, why? -> TODO VOLKAN
  - investigate other sac reward values
    - GYM env is too simple
  - [ ] try +50,-10
  - ask felix for advice
  - [X] check the paper for hparams and reward values
  
  - [ ] check the mean episode reward
  - start start with 0 and go towards the value of target reached 


#### Tests
- [X] regular sac w/o pretrain STATIC -> TODO MO
  - once with general purpose N2 + high vCPU 32
  - once with compute optimised C2 + 60 vCPU
- [ ] regular sac-x w/o pretrain STATIC 

- [X] regular sac w/o pretrain DYNAMIC -> TODO MO
- [ ] regular sac-x w/o pretrain DYNAMIC

- [ ] then sac with pretrain DYNAMIC
- [ ] then sac-x with pretrain DYNAMIC
  - OPTION 1
  - [ ] simulate that we used scheduler
  - OPTION 2
  - [ ] do pretrain on static initial 





### TEST FOR PRESENTATION
- [ ] start without pretraining to test the papers method
- [ ] train with scheduler
- [ ] run the robot on main task only which is the super sparse reward
  - [ ] possible problem: main task can get stuck
  - [ ] activate the needed subtask by check if the agent is not moving double the waiting  
- [ ] maybe an LSTM instead of MLP and leave MetaRL

- [ ] regular sac w/o pretrain STATIC
- [ ] regular sac-x w/o pretrain STATIC

- [ ] regular sac w/o pretrain DYNAMIC
- [ ] regular sac-x w/o pretrain DYNAMIC


- [ ] then sac with pretrain DYNAMIC
- [ ] then sac-x with pretrain DYNAMIC
  - OPTION 1
  - [ ] simulate that we used scheduler
  - OPTION 2
  - [ ] do pretrain on static initial 
    - then let it figure out

  - increase the number of steps for action smoothing
- if Sac-x comparison doesnt work
  - we can still compare static/dynamic and with or without pretrining



### ENDTERM PRESENTATION
- [ ] add dates to slides and supervisor name and page numbers
- [ ] museum is a good example
- [ ] we focus on the strategy not the representation
- [ ] we assume we have obstacle id 
- [ ] write down all assumptions
  - [ ] assume we have obstacle ids	
- [ ] dont explain the model only the training,
- [ ] only show the model features/parameters
- [ ] add dates to slides and supervisor name and page numbers
- [ ] use additional prepared slides for questinos on model 
- [ ] feedback to do in final presentation after midterm presentation:
- start directly with the objective
- we want to solve planning, but with exploration then we want subtask strategy
- very short explain the speciality of SAC-X / MetaRL
  - core idea, why it helps to nagivation problem (1 slide)
- put slides in the appendix 

- all the setup/model details go in to the appendix
- but shortly tell what all subtasks are in a list of sub sparse rewards
- maybe explain one special sub task
- then skip to results
- and explain everything even if they fail
- if Sac-x comparison doesnt work
  - we can still compare static/dynamic and with or without pretrining
- static vs dynamic : sac vs sac-u
- pretraining
  - bypass learning skills so that agent can deide the skills in a different way, should resemble to pretrained version
- SKILL DEFINITIONS from 23-01-30-week
- Test results and interpretation from plots from xcel

### ENDTERM REPORT
- [ ] only action smoothing behaves strange probably because pre-training teaches to go the goal but normal training with 10target, -50collision makes it stop around the target a while
- [ ] DIfferences from original paper: and mention why it work for their example, and why not in our example
- [ ] SAC-X, sac u makes no sense
- [ ] change scheduling depending on the distances to object and target 
- [ ] make us of appendix
- [ ] Performance is not that important but why it doesnt perform is important
- [ ] add what failed
- [ ] never descripbe the history, 
- [ ] write it as you already had the final idea right from the start
- [ ] so that the paper is not a huge version
- [ ] cite all the algorithm names
- [ ] put numbers on the visual to indacte sorting
- [ ] checkpoint rewards: use more visuals 
  - put and arrow and +1 reward etc. 
  - 2nd arrow inside the area, no reward
- [ ] put standard deviation of the result table
- [ ] send a draft first, before the final version 
- [ ] ADD IT in to the report that we differ from SAC-X Scheduler!!

#### as Appexdix
- [ ] model specifications
- [ ] maybe if we generate data we can talk about the algo
- [ ] pytorch, wandb, gcloud, own implementation of environment from open ai gym template

















### SOME IDEAS FROM PREVIOUS WEEKS

#### Inputs
- [ ] give subtasks with labels into the state
  - if done with latent variable (would be MetaRL) 
- [ ] give in possibly the acceleration in the state as well if there is time to test

### Obstacles
  - [ ] moveing obst move and stop and start again, traj generation put it in the ENDTERMPPT
  - [ ] Possible improvement: Eg. if 1 step is 10 pixel movement only occupy pixel 6-10 for all agents as “prediction”
  - [ ] sort the obstacles according the distance and give weights appropriately
    - [ ] take the time to collisions in to account calc depending on our speed  
  - [ ] or take the closest obstacle only
  - [ ] hindsight replay buffer -> works well for super sparse reward (1 target, -1 crash) (after sorting obstacle and smooting)
    - it relabels trajectories in replay buffer that could be the correct trajectory for an another environment!
    - the agent can act "idle" basically avoid obstacle, and not reach to goal as  
  - [ ] do a short google search for research based version for jump start idea -> paper from FELIX  about it
    - [ ] THIS ONE? https://deepai.org/publication/pretraining-in-deep-reinforcement-learning-a-survey  
  - [ ] only give the closest 2 obstacles during dynamic
    - like a field of view
    - explanation would be in an AD scenario this is the case, closest matter more
  - add curricilum basically incrementally adding more adn more obstacle
    - moveing obst move and stop and start again, generation
    - pretrain on the initial state!! resembles more to the real scenario

### Jump Start
- rrt* for constinious global instead of AStar

#### SAC-X
- Scheduler:
  - [ ] last 3 subtasks and nearest obstacle + target state OVERDUE
  - worth trying, better than uniform scheduing anyway

### Performance
- [X] only pretrain and see the result as well (MO) 
  - [ ] use in endterm report
- [ ] 5 variations for learning rate and batch size and check performance 
- [ ] OR use random search, but takes lots of time 
- [ ] use a curriculum
  - to make the start easier we can have the goal closer and push it away over time
    - e.g. increase the radius of initial goal distance every 10000 steps
  - switch to harder env, like 5-10-15 obstacles etc.

### Google Cloud / Wandb
- [ ] cache replay buffer 
  - maybe store it in a numpy file
- [ ] setup gcloud auto email for crashes, if that doesn't work use wandb emailing

### Plotting
- 500 Hz env update freq delta_t 1/500s, but controller has lower update freq for smoother behavior (meaning for examle that we run the training every second time the environment)
control max steps by time not nubmer of steps
- increase the number of steps for action smoothing

### Collab
- [ ] possible to see other teams PPTs?
  - if i remember correctly they were gonna be shared right?