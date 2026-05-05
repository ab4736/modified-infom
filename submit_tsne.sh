#!/bin/bash
#SBATCH --job-name=tsne_viz
#SBATCH --output=exp/tsne_%j.out
#SBATCH --error=exp/tsne_%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate /scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity
cd /scratch/gpfs/HASSON/ab4736/modified-infom

export XLA_PYTHON_CLIENT_PREALLOCATE=false
/scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity/bin/python tsne_viz.py
