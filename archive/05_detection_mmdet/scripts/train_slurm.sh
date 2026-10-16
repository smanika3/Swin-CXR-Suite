#!/bin/bash
#SBATCH --job-name=TBX_Swin_MMD
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=logs/mmdet_train_%j.out
#SBATCH --error=logs/mmdet_train_%j.err

echo "=== Starting TBX11K Swin-B MMDetection Training ==="
date

# Load cluster modules
module purge
module load gcc-11.2.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

# Activate Conda/Mamba environment
source activate swin_mmdet_env

mkdir -p logs

# Run training via OpenMMLab MIM
mim train mmdet configs/tbx11k_swin_frcnn.py --work-dir ./work_dirs/tbx_swin_exp1

echo "=== Training Finished ==="
date
