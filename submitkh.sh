#!/bin/bash

#SBATCH --job-name=pkc-gpu

#SBATCH --partition=gpu
#SBATCH --nodelist=node2
#SBATCH --error=test_gpu_error_%j.log
#SBATCH --output=test_gpu_job_output_%j.log
###SBATCH --gres=gpu:H100:1
#SBATCH --gres=shard:30

cd $SLURM_SUBMIT_DIR

module load anaconda3-2024.2  
module load cuda-12.8
# create a fresh virual envi
python -m venv ~/envs/PM25
de1i
source ~/envs/PM25/bin/activate
nvidia-smi
#pip install --upgrade pip
#pip install geopy
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#upgrade pip setuptools wheel
#pip install --upgrade pip setuptoolswheel
#pip uninstall torch
#pip uninstall -y backports
#pip install cupy
#pip install pandas
pip install metpy
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__)").html
pip install bresenham
pip install arrow
pip install torchinfo
pip install thop
pip install -U kaleido
pip install plotly
pip install pywt
pip install PyWavelets
#pip install matplotlib==3.8.0
#pip install seaborn==0.12.2
#pip install scikit-learn==1.6.1
#pip install scipy==1.11.4
#pip install openpyxl
#pip install numpy==1.26.4
export TF_ENABLE_ONEDNN_OPTS=0
#python PM2.5_SIR_CHECK_FINAL.py 
#python PM_25_GNN.py
#python SEGMENT_MODEL_CITY.py
#python CITY_MODEL_FINAL.py
#python PM_MM_DELHI_FI
#python CLUSTER_CODE_MA.py
python CLUSTER_CODE_INDIAN.py
#python SEGMENT_MODEL_N.py
#python CLUSTER_MODEL_LT.py
#python GNN.py
#python SEGMENT_CODE.py
