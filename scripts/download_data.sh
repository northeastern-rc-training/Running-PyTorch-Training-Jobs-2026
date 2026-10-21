#!/bin/bash
#SBATCH --job-name=download-cifar10
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

DATA_DIR=/scratch/$USER/data

module load miniconda3/25.9.1
source activate pytorch_env
python download_data.py --data-dir "$DATA_DIR"
