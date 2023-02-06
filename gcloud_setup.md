# Gcloud


## Setting up a new instance
- create a VM on gCLoud from templates to have spot and ubuntu already chosen
- only have to change CPU/GPU and region to west

- get the gCloud VM instance command from and ssh in to the VM
```
gcloud compute ssh --zone "europe-west1-b" "instance-highcpu-1"  --project "tum-adlr-09"
gcloud compute ssh --zone "europe-west1-b" "instance-2" --project "tum-adlr-09"
```

- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
```
ssh-keygen -t ed25519 -C "XXX@gmail.com"
```

- copy public key to your git-hub account to use ssh for git clone
```
ls .ssh/
nano .ssh/id_ed25519.pub
```

- clone your project
```
git clone git@github.com:silence1998/tum-adlr-09.git
```

- setup the venv following this guide
- https://cloud.google.com/python/docs/setup#linux
- to activate the env: (from tum-adlr-09)
```
source env/bin/activate 
```

- run the following commands to install the required packages for the project
- we used conda on local so the generated requirements.txt is only usable with conda
- but gCloud doesn't support conda so we have to install the packages manually

```
pip install numpy
pip install matplotlib
pip install torch
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install gym
pip install wandb
pip install pygame
pip install google-cloud-storage

```


- exit ssh via Ctrl+D or
```
exit
```

## Using the instance

### SSH & Login
```
gcloud compute ssh --zone "europe-west1-b" "instance-n2-8vcpu-32gb-mem" --project "tum-adlr-09"
gcloud auth login 
```

### Activate the environment created in the setup
```
cd tum-adlr-09
source env/bin/activate  
```

### tmux for running trainings in the background in the server
```
python3 SAC-X/training.py
```

- tmux useful shortcuts
  - https://tmuxcheatsheet.com/

### to copy the files in the project bucket from current dir
```
gsutil cp file gs://tum-adlr-09/
```
```
gsutil cp -r model_pretrain gs://tum-adlr-09/
```

### to rename files in the project bucket 
```
gsutil mv gs://my_bucket/olddir gs://my_bucket/newdir
```

### to copy files in the current dir from the project bucket
```
gsutil -m cp -r "gs://tum-adlr-09/tmp" .
```
```
gsutil -m cp -r "gs://tum-adlr-09/model_pretrain" .
```



Script wise for tests

Step 1: Select instance
```
gcloud compute ssh --zone "europe-west1-b" "instance-n2-8vcpu-32gb-mem-9" --project "tum-adlr-09"
```
Step 2: get ssh to cloone project
```
ssh-keygen -t ed25519 -C "XXX@gmail.com"
ls .ssh/
nano .ssh/id_ed25519.pub
```
Step 3: give gcloud access
```
gcloud auth login
```
Step 4: download and checkout everything needed
```
sudo apt install git-all
git clone git@github.com:silence1998/tum-adlr-09.git
cd tum-adlr-09/
git checkout continous
sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
python3 -m venv env
source env/bin/activate 
pip install numpy
pip install matplotlib
pip install torch
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install gym
pip install wandb
pip install pygame
pip install google-cloud-storage
```
Step 5: run training with the needed parameters
```
cd SAC-X
nano parameters.py

python3 training.py
```
Step 6: upload the resulting model to gcloud storage bucket
```
gcloud auth login

gsutil cp -r model_pretrain gs://tum-adlr-09/model_pretrain/
gsutil mv gs://tum-adlr-09/model_pretrain/model_pretrain/ gs://tum-adlr-09/model_pretrain/Sa-t<new_name>
```
Step 7: download to your pc to run metrics and visuals
```
gsutil -m cp -r "gs://tum-adlr-09/model_pretrain/Sa-t<new_name>" .
```
