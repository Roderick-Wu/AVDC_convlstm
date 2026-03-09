#!/bin/bash
#SBATCH --job-name=train_combined_classifier
#SBATCH --account=def-gigor
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=350G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

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
echo "Starting training..."
python -u train.py

echo "Training completed!"
