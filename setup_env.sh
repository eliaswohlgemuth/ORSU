#!/bin/bash

# Exit script if any command fails
set -e

# Create a new conda environment with Python 3.10
conda create -n orsu python=3.10 -y

# Activate the new conda environment
conda activate orsu

# Start dependencies by setting up Exllamav2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_exllamav2.txt
pip install exllamav2

# Install PyTorch with CUDA 11.8 support and other project dependencies
conda install pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# setup GLIP
cd GLIP
python setup.py clean --all build develop --user
# downgrade some packages for GLIP to work
pip install "pydantic<2.0"
pip install numpy==1.23.5