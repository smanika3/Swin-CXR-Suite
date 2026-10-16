#!/bin/bash
#SBATCH --job-name=Swin_FRCNN_TBX
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "=== Starting Swin Faster R-CNN Training on TBX11K ==="
date

module purge
module load gcc-11.2.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

source activate swin_pytorch_env
mkdir -p logs checkpoints

python -u train.py \
    --data-dir /scratch/$USER/tbx11k-unzip \
    --checkpoint-dir ./checkpoints \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.005 \
    --num-workers 8 \
    --resume

echo "=== Training Finished ==="
date
