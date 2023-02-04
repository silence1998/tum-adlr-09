
## Organizational

## Environment

### Continous
### Obstacles

### Collision Check


### Reward Func

#### Sub sparse Rewards
  
## Algorithm
#### SAC-X


#### SAC


##### JUMP START IDEA

### Policy
### Networks
#### Input

#### Outputs

## Optimizer 
### Loss
## Performance 

### Curves 


### Google Cloud 


## MILE STONES



### MIDTERM PRESENTATION

### MIDTERM REPORT


### ENDTERM PRESENTATION

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
  - try +50,-10
  - ask felix for advice
  - [ ] check the paper for 
  


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

#### as Appexdix

### ENDTERM REPORT
only action smoothing behaves strange probably because pre-training teaches to go the goal but normal training with 10target, -50collision makes it stop around the target a while
DIfferences from original paper: and mention why it work for their example, and why not in our example
SAC-X, sac u makes no sense
change scheduling depending on the distances to object and target 

Performance is not that important but why it doesnt perform is important

### EXAM


## COLLAB TEAM 7 ?

