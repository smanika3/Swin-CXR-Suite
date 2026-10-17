#!/bin/bash
#SBATCH --job-name=Swin-Chex-Predict-Teacher-Multi
#SBATCH --partition=gpu
#SBATCH --qos=public
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=28
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/smanika3/feedback/v2-self/logs/swin_predict_teacher_multi_%j.out
#SBATCH --error=/scratch/smanika3/feedback/v2-self/logs/swin_predict_teacher_multi_%j.err
#SBATCH --export=NONE

echo "=== Starting Swin Transformer GPU Training Job (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date

module purge
module load mamba/latest
module load cuda-12.6.1-gcc-12.1.0
source activate /home/smanika3/.conda/envs/swin_pytorch_env
unset PYTHONPATH
export LD_PRELOAD=""

python3 -u "$SLURM_SUBMIT_DIR/teacher_predict_multi.py"

echo "=== GPU Training Job Finished (Array Task ${SLURM_ARRAY_TASK_ID}) ==="
date
