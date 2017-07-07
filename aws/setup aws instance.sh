# make downloads directory
mkdir downloads
cd downloads
wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash Anaconda3-4.4.0-Linux-x86_64.sh
source ~/.bashrc

conda create -n tensorflow-env python=3.5 anaconda
source activate tensorflow-env

# install CUDA
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb

sudo apt-get update
sudo apt-get install python3-pip
pip3 install -U tensorflow
pip install -U keras