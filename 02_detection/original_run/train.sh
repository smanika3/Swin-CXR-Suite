#!/bin/bash
#SBATCH --job-name=Swin-TBX-Train
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/smanika3/swin-localization-logs/logs/swin_train_%j.out
#SBATCH --error=/scratch/smanika3/swin-localization-logs/logs/swin_train_%j.err
#SBATCH --export=NONE

echo "=== Starting Swin Transformer GPU Training Job (Job ID: ${SLURM_JOB_ID}) ==="
date

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0
source activate /home/smanika3/.conda/envs/swin_pytorch_env
unset PYTHONPATH
export LD_PRELOAD=""

python3 -u "$SLURM_SUBMIT_DIR/train.py"

echo "=== GPU Training Job Finished (Job ID: ${SLURM_JOB_ID}) ==="
date
