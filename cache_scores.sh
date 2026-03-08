#!/bin/bash
#SBATCH --job-name=cache_combined_scores
#SBATCH --time=0-18:00 # D-HH:MM
#SBATCH --account=def-gigor
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100_3g.20gb:1
#SBATCH --output=slurm-cache-%j.out

# Load modules
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load opencv/4.11.0

# Run inference and cache scores
python cache_combined_scores.py
