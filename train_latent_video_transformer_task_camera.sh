#!/bin/bash
#SBATCH --job-name=lvit_task_cam
#SBATCH --account=def-gigor
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/wuroderi/scratch/slurm_logs/lvit_task_cam/%x_%j.out

module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load opencv/4.11.0

cd /home/wuroderi/projects/def-gigor/wuroderi/classifier

if [[ -z "${TASK_NAME:-}" || -z "${CAMERA_NAME:-}" ]]; then
  echo "TASK_NAME and CAMERA_NAME must be set by the submission wrapper."
  exit 1
fi

echo "Starting task/camera transformer job: task=${TASK_NAME}, camera=${CAMERA_NAME}"

mkdir -p /home/wuroderi/scratch/slurm_logs/lvit_task_cam
mkdir -p /home/wuroderi/scratch/checkpoints_latent_video_transformer_task_camera

python -u train_latent_video_transformer_task_camera.py \
  --task "${TASK_NAME}" \
  --camera "${CAMERA_NAME}" \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-5 \
  --val-split 0.1 \
  --video-size 96 \
  --num-load-workers 32 \
  --num-workers 0 \
  --use-contrastive-loss \
  --contrastive-weight 0.03 \
  --contrastive-temperature 0.1 \
  --contrastive-warmup-epochs 5 \
  --model-save-root /home/wuroderi/scratch/checkpoints_latent_video_transformer_task_camera

echo "Training completed for task=${TASK_NAME}, camera=${CAMERA_NAME}!"
