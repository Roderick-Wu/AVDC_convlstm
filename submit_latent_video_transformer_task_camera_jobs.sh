#!/bin/bash
set -euo pipefail

TASKS=(
  assembly-v2-goal-observable
  basketball-v2-goal-observable
  button-press-topdown-v2-goal-observable
  button-press-v2-goal-observable
  door-close-v2-goal-observable
  door-open-v2-goal-observable
  faucet-close-v2-goal-observable
  faucet-open-v2-goal-observable
  hammer-v2-goal-observable
  handle-press-v2-goal-observable
  shelf-place-v2-goal-observable
)

CAMERAS=(corner corner2 corner3)

LOG_ROOT=/home/wuroderi/scratch/slurm_logs/lvit_task_cam
CKPT_ROOT=/home/wuroderi/scratch/checkpoints_latent_video_transformer_task_camera

mkdir -p "$LOG_ROOT" "$CKPT_ROOT"

echo "Submitting ${#TASKS[@]} x ${#CAMERAS[@]} = $(( ${#TASKS[@]} * ${#CAMERAS[@]} )) jobs..."

for task in "${TASKS[@]}"; do
  for camera in "${CAMERAS[@]}"; do
    echo "Submitting task=${task}, camera=${camera}"
    sbatch \
      --export=ALL,TASK_NAME="${task}",CAMERA_NAME="${camera}" \
      train_latent_video_transformer_task_camera.sh
  done
done

echo "All jobs submitted. Logs will be in ${LOG_ROOT}."
echo "Checkpoints will be written under ${CKPT_ROOT}."
