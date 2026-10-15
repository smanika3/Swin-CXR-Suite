#!/bin/bash
#SBATCH --job-name=SwinUnet_CheXmask
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "=== Starting Swin-Unet Segmentation Training on CheXmask ==="
date

module purge
module load gcc-11.2.0-gcc-11.2.0
module load cuda-12.6.1-gcc-12.1.0
module load mamba/latest

source activate swin_pytorch_env
mkdir -p logs checkpoints

python -u train.py \
    --img-dir /scratch/$USER/chexpert/full_uncompressed/train \
    --mask-dir /scratch/$USER/chexmask-masks/train \
    --train-list /scratch/$USER/chexmask-masks/train_files.txt \
    --val-list /scratch/$USER/chexmask-masks/valid_files.txt \
    --checkpoint-dir ./checkpoints \
    --epochs 40 \
    --batch-size 32 \
    --lr 1e-4 \
    --resume

echo "=== Training Finished ==="
date
