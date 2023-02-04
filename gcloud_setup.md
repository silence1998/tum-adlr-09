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
- https://cloud.google.com/python/doccd s/setup#linux
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
gcloud compute ssh --zone "europe-west1-b" "instance-highcpu"  --project "tum-adlr-09"
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

### to copy files from the project bucket to current dir
```
gsutil -m cp -r "gs://tum-adlr-09/tmp" .
```
```
gsutil -m cp -r "gs://tum-adlr-09/model_pretrain" .
```
