#!/bin/bash
#SBATCH --job-name=visualize_embeddings
#SBATCH --account=def-gigor
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --gres=gpu:1

# Load modules
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load opencv/4.11.0

# Activate virtual environment (adjust path as needed)
# source ~/env/bin/activate

# Print GPU info
nvidia-smi

# Go to classifier directory
cd /home/wuroderi/projects/def-gigor/wuroderi/classifier

# Run training
python -u visualize_embeddings.py

