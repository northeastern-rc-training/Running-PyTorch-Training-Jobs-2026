#!/bin/bash
#SBATCH --job-name=pytorch-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=%x-%j.out

DATA_DIR=/scratch/$USER/data

module load miniconda3/25.9.1
source activate pytorch_env

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

python train.py --data-dir "$DATA_DIR" --epochs 10 "$@"
