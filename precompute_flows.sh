#!/bin/bash
#SBATCH --job-name=precompute_flows
#SBATCH --account=def-gigor
#SBATCH --time=15:00:00  # 15 hours for single task (~10.4 hours estimated)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-precompute-%x-%j.out

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

# Run flow pre-computation
echo "Starting flow map pre-computation..."
echo "Task: $1"

# Pass task name as argument to Python script
if [ -z "$1" ]; then
    echo "ERROR: No task specified!"
    echo "Usage: sbatch precompute_flows.sh <task-name>"
    exit 1
fi

python precompute_flows.py "$1"

echo "Flow pre-computation completed!"
