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

  -  CONSISTENCY collect history of trajectories
    - take std
    - [ ] punish high std/variance
    - cus entropy is too high
      - because SAC pushes this behavior

- [ ]increase the number of steps for action smoothing

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
    - didnt do this as time reasons
  - OPTION 2
  - [X] do pretrain on static initial 
    - then let it figure out

  - increase the number of steps for action smoothing
- if Sac-x comparison doesnt work
  - we can still compare static/dynamic and with or without pretrining



### ENDTERM PRESENTATION
SKIP A LOT AS THERE IS NO TIME
THE QUESTION IS WHATS TO ANSWER OF THE RESEARCH QUESTION
  - show the method
    - Pretraining
  - NOT THE DETAILS, but a picture NN ARCHI
  - FEATURES
    - only overview of how it works 
    - but not the results
    - only show the finalization

  - SAC to SAC-X difference (SCHEDULER)
    - n subtasks
    - subtask overview

  RESEARCH Q ANSWER
  - [ ] regular sac w/o pretrain STATIC
  - [ ] regular sac-x w/o pretrain STATIC

  - [ ] regular sac w/o pretrain DYNAMIC
  - [ ] regular sac-x w/o pretrain DYNAMIC

  - [ ] then sac with pretrain DYNAMIC
  - [ ] then sac-x with pretrain DYNAMIC

- put the SAC-X test scheude in Appendix


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
  - [ ] https://deepai.org/publication/pretraining-in-deep-reinforcement-learning-a-survey  
    - this does it with NNs we do it with algorithmic planner
  - we reconstruct all action the agent could have taken if it went along with the traj
- SKILL DEFINITIONS from 23-01-30-week
- Test results and interpretation from plots from xcel
- [ ] only pretrain and see the result as well 
  - [ ] use in endterm report
- 500 Hz env update freq delta_t 1/500s, but controller has lower update freq for smoother behavior (meaning for examle that we run the training every second time the environment)
control max steps by time not nubmer of steps
  - WE HAVE NO INERTIA SO ALWAYS JITTERY ANWAY

#### Backup Slide
- hindsight replay buffer vs pretrain reasoning (SAC-X is a like a hindsight replay buffer)
  - SAC-XwHRB -VS- pretrain: we do with a star in static initialization, SAC-XwHRB labels the traj if it was a reward in normal traning
  - SAC-XwHRB label a non-crashing traj also as a possible traj to a target that might have been there
  - we filter out all non-succeeding ones
  - DOUBLE CHECK BEFORE WRITING IN

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
- [ ] 5 variations for learning rate and batch size and check performance 
  - or random search
- delete the subtask from being able to get chosen if its turned of for endterm report and do the tests again



#### as Appexdix
- [ ] model specifications
- [ ] maybe if we generate data we can talk about the algo
- [ ] pytorch, wandb, gcloud, own implementation of environment from open ai gym template

















### SOME IDEAS FROM PREVIOUS WEEKS
[X]: means not crucial

#### Inputs
- [X] give subtasks with labels into the state
  - if done with latent variable (would be MetaRL) 
- [X] give in possibly the acceleration in the state as well if there is time to test
  - NOT NEEDED

### Obstacles
  - [X] moveing obst move and stop and start again, traj generation put it in the ENDTERMPPT
  - [X] Possible improvement: Eg. if 1 step is 10 pixel movement only occupy pixel 6-10 for all agents as “prediction”
  - [X] give weights appropriately
    - [X] take the time to collisions in to account calc depending on our speed  
    - NETWORK LEARN
  - [X] hindsight replay buffer -> works well for super sparse reward (1 target, -1 crash) (after sorting obstacle and smooting)
    - it relabels trajectories in replay buffer that could be the correct trajectory for an another environment!
    - the agent can act "idle" basically avoid obstacle, and not reach to goal as  
    - EASY HACK FOR SAC FOR REACHING GOALS
    - we already have this partially with weighting succeeding trajectories
  - [X] only give the closest 1-2 obstacles during dynamic
    - like a field of view
    - explanation would be in an AD scenario this is the case, closest matter more
    - NOT GOOD OR GLOBAL PLANNING
  - add curricilum basically incrementally adding more and more obstacle
    - moving obst move and stop and start again as an additional, generation
    - pretrain on the initial state!! resembles more to the real scenario
    - start with a closer target and make the target further 
      - to make the start easier we can have the goal closer and push it away over time
    - e.g. increase the radius of initial goal distance every 10000 steps
    - switch to harder env, like 5-10-15 obstacles etc.
    - POSSIBLE FUTURE IMPROVEMENT BUT WE DO IT WITH PRETRAIN

### Jump Start
- rrt* for constinious global instead of AStar
  - no need, compute time 

#### SAC-X
- Scheduler:
  - last 3 subtasks and nearest obstacle + target state
  - worth trying, better than uniform scheduing anyway
  - DONE

### Performance

### Google Cloud / Wandb
- [ ] cache replay buffer / scheduler
  - maybe store it in a numpy file
- [ ] setup gcloud auto email for crashes, if that doesn't work use wandb emailing
- WAS NO URGENCY

### Plotting


### Collab
- [ ] possible to see other teams PPTs?
  - if i remember correctly they were gonna be shared right?