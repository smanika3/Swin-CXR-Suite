#!/bin/bash
#SBATCH --job-name=Swin-Chex-Train-Student
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --array=1-6                  # 6 jobs to cover 60 epochs with 10 epochs each
#SBATCH --output=/scratch/smanika3/feedback/v2-self/logs/swin_train_student_%A_%a.out
#SBATCH --error=/scratch/smanika3/feedback/v2-self/logs/swin_train_student_%A_%a.err
#SBATCH --export=NONE

echo "=== Starting Swin Transformer GPU Training Job (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0
source activate /home/smanika3/.conda/envs/swin_pytorch_env
unset PYTHONPATH
export LD_PRELOAD=""

python3 -u "$SLURM_SUBMIT_DIR/student_train.py"

echo "=== GPU Training Job Finished (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date
