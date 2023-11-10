#!/bin/bash -i

# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x ./Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh -b
source /root/anaconda3/bin/activate 

# Install mamba
conda create --name initialization python=3.8 -y
conda activate initialization
conda install -c conda-forge mamba -y

# Create l2r environment using mamba
mamba create --name l2r python=3.8 -y
mamba init

source ~/.bashrc