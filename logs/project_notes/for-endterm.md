
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
  - env gen random seed (VOLKAN)
    - performed bad as if we pretrain in env#1 and normal train in env#1 it overfits and cant generalize
    - bad sample balance 

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

#### ContSAC-X

- dont use sac-u

MAIN TASK:
- super-sparse rewards (-50, +10)
- Time penalty

- [ ] 6 Skills: (Sparsa Rewards:SR)

  - Skill 1: seeking obstacle (to follow a moving obstacle)
    - dense rewards distance to obstacle
    - stay between 2-5 * radius of the obstacle

  - Skill 2: avoiding obstacle 
    - dense rewards distance to obstacle for dynamic obstacles or for all if the latter proves to be hard
    - sparse rewards checkpoint to obstacle for static -> avoidance SR TODO VOLKAN
  
  - Skill 3: seek target 
    - sparse rewards checkpoint to target
    - NOT distance to target -> SAC-X performs badly
  
  - NOPE Skill 4: flee from target (if we have to find a new way)
    - dense rewards distance to target
    - we didnt test this first as our agent is not capable enough yet -> TODO VOLKAN implementation (2nd Prio)
 
  - Skill 5: consistency 
    - consistency SR -> TODO VOLKAN implementation
 
  - Skill 6: waiting 
    - waiting SR -> TODO VOLKAN implementation

- obstacle: can be dense if its hard to implement 
  - seeking distance threshold: smaller aggressive threshold
  - avoiding distance threshold: bigger conservative threshold
- target 
  - seeking (SR CHECKPOINTS)
  - avoiding

#### Tests
- [ ] regular sac w/o pretrain STATIC -> TODO MO
- [ ] regular sac-x w/o pretrain STATIC 

- [ ] regular sac w/o pretrain DYNAMIC -> TODO MO
- [ ] regular sac-x w/o pretrain DYNAMIC

- [ ] then sac with pretrain DYNAMIC
- [ ] then sac-x with pretrain DYNAMIC
  - OPTION 1
  - [ ] simulate that we used scheduler
  - OPTION 2
  - [ ] do pretrain on static initial 

#### as Appexdix

### ENDTERM REPORT


### EXAM


## COLLAB TEAM 7 ?

