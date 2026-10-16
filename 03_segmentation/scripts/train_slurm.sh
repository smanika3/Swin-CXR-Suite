#!/bin/bash
# SLURM job for ASU Sol (module names, partition and QOS are Sol-specific; adapt for other clusters).
# Submit from 03_segmentation/ after creating the log directory:
#   mkdir -p logs && sbatch scripts/train_slurm.sh
# Set DATA to the folder that holds chexpert/ and chexmask-masks/.
#SBATCH --job-name=SwinUnet_CheXmask
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "=== Starting Swin-Unet Segmentation Training on CheXmask ==="
date

cd "$SLURM_SUBMIT_DIR"

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

source activate swin_pytorch_env
mkdir -p logs checkpoints

DATA=${DATA:-/scratch/$USER}

python -u train.py \
    --img-dir $DATA/chexpert/train \
    --mask-dir $DATA/chexmask-masks/train \
    --train-list $DATA/chexmask-masks/train_split.txt \
    --val-img-dir $DATA/chexpert/valid \
    --val-mask-dir $DATA/chexmask-masks/valid \
    --val-list $DATA/chexmask-masks/valid_files.txt \
    --checkpoint-dir ./checkpoints \
    --epochs 11 \
    --batch-size 96 \
    --num-workers 8 \
    --resume

echo "=== Training Finished ==="
date
