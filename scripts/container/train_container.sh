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

# shared pytorch image
CONTAINER=/shared/container_repository/explorer/pytorch/pytorch_training.sif

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# --nv : make the GPU visible to the container
# -B   : bind /scratch directory
apptainer exec --nv -B /scratch "$CONTAINER" \
    python train.py --data-dir "$DATA_DIR" --epochs 10 "$@"
