#!/bin/bash
# Submit precompute_flows.sh for all tasks in parallel
# This speeds up the process from ~115 hours sequential to ~11 hours parallel (with enough GPUs)

echo "=========================================="
echo "Submitting parallel flow pre-computation jobs"
echo "=========================================="

TASKS=(
    "assembly-v2-goal-observable"
    "basketball-v2-goal-observable"
    "button-press-topdown-v2-goal-observable"
    "button-press-v2-goal-observable"
    "door-close-v2-goal-observable"
    "door-open-v2-goal-observable"
    "faucet-close-v2-goal-observable"
    "faucet-open-v2-goal-observable"
    "hammer-v2-goal-observable"
    "handle-press-v2-goal-observable"
    "shelf-place-v2-goal-observable"
)

# Submit a job for each task
for task in "${TASKS[@]}"
do
    echo "Submitting job for: $task"
    sbatch --job-name="precompute_${task%%\-*}" precompute_flows.sh "$task"
done

echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check progress: find /home/wuroderi/scratch/activations_metaworld/avdc_policy_raw/flow_maps -name '*.npy' | wc -l"
echo "Expected total: 165,000 files"
echo ""
echo "Each task processes 15,000 samples (~10.4 hours)"
echo "All tasks will complete in ~11 hours if running in parallel"
