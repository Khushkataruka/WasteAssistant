#!/bin/bash

#SBATCH --job-name=kh-gpu

#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=test_gpu_error_%j.log
#SBATCH --output=test_gpu_job_output_%j.log
##SBATCH --gres=shard:10

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2  
module load cuda-12.8
# create a fresh virual envi
#python -m venv ~/envs/PM25
#source ~/envs/PM25/bin/activate
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#upgrade pip setuptools wheel
#pip install --upgrade pip setuptoolswheel
#pip uninstall torch
#pip install bert-score
#pip uninstall -y backports
#pip install cupy
#pip install pandas
#pip install matplotlib==3.8.0
#pip install seaborn==0.12.2
#pip install scikit-learn==1.6.1
#pip install scipy==1.11.4
#pip install openpyxl
#pip install numpy==1.26.4
export TF_ENABLE_ONEDNN_OPTS=0
#python PM2.5_SIR_CHECK_FINAL.py 
#python test.py
python create_data.py
#python PM_MM_DELHI_FINAL.py
