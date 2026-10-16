#!/bin/bash
#SBATCH --job-name=Swin-Multi-Test
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/smanika3/feedback/v1/logs/test/swin_test_%j.out
#SBATCH --error=/scratch/smanika3/feedback/v1/logs/test/swin_test_%j.err

echo "=== Starting Swin Transformer GPU Testing Job ==="
date

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0

ENV_PYTHON="$HOME/.conda/envs/swin_pytorch_env/bin/python3"
"$ENV_PYTHON" -u "$SLURM_SUBMIT_DIR/multi_test.py"

echo "=== GPU Testing Job Finished ==="
date
