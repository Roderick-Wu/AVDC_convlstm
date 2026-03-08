"""
Pre-compute and cache all optical flow maps to disk.
This dramatically speeds up training by avoiding re-computing flows every epoch.

Run this ONCE before training to generate all flow maps.
"""
import torch
import numpy as np
import os
from tqdm import tqdm
import json
import sys
import data


def precompute_flow_maps(directory, task, cameras, episodes, timesteps, results_json, 
                        flow_model, device, use_gmflow=True, output_dir=None):
    """
    Pre-compute all optical flow maps and save to disk.
    
    Args:
        output_dir: Directory to save flow maps (default: directory/flow_maps/)
    """
    if output_dir is None:
        output_dir = os.path.join(directory, 'flow_maps')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("PRE-COMPUTING OPTICAL FLOW MAPS")
    print("="*80)
    print(f"Task: {task}")
    print(f"Episodes: {len(episodes)}")
    print(f"Cameras: {cameras}")
    print(f"Timesteps: {timesteps}")
    print(f"Output directory: {output_dir}")
    print(f"Flow model: {'GMFlow' if use_gmflow else 'RAFT'}")
    print("="*80)
    
    # Create episode info list
    episode_info_list = data.create_episode_info_list(
        directory=directory,
        tasks=[task],
        cameras=cameras,
        episodes=episodes,
        diffusion_time_steps=timesteps,
        results_json=results_json
    )
    
    total_samples = len(episode_info_list)
    print(f"\nTotal samples to process: {total_samples}")
    print("This will take some time (~2-3 seconds per sample)...")
    print(f"Estimated time: {total_samples * 2.5 / 3600:.1f} hours\n")
    
    # Process each sample
    processed = 0
    skipped = 0
    errors = 0
    
    with tqdm(total=total_samples, desc="Computing flows") as pbar:
        for info in episode_info_list:
            # Create output path
            # Structure: flow_maps/{task}/{camera}/episode_{episode}/flow_t{timestep}.npy
            flow_dir = os.path.join(output_dir, info['task'], info['camera'], f"episode_{info['episode']:03d}")
            os.makedirs(flow_dir, exist_ok=True)
            flow_path = os.path.join(flow_dir, f"flow_t{info['timestep']:03d}.npy")
            
            # Skip if already computed
            if os.path.exists(flow_path):
                skipped += 1
                pbar.update(1)
                pbar.set_postfix({'processed': processed, 'skipped': skipped, 'errors': errors})
                continue
            
            try:
                # Load x0 prediction video (7 frames)
                x0_frames = data.load_video_frames(info['x0_path'])
                
                # Load condition image from final video
                condition_frame = data.load_condition_frame(info['condition_path'])
                
                # Prepend condition to x0 frames to get 8 frames total
                video_frames = [condition_frame] + x0_frames  # 8 frames
                
                # Compute flow maps between consecutive frames (8 frames → 7 flows)
                flow_maps = data.compute_flow_sequence(
                    video_frames, 
                    flow_model, 
                    device, 
                    use_gmflow
                )  # (2, 7, H, W)
                
                # Save to disk
                np.save(flow_path, flow_maps.cpu().numpy())
                processed += 1
                
            except Exception as e:
                print(f"\nError processing {info['camera']}/episode_{info['episode']}/t{info['timestep']}: {e}")
                errors += 1
            
            pbar.update(1)
            pbar.set_postfix({'processed': processed, 'skipped': skipped, 'errors': errors})
    
    print("\n" + "="*80)
    print("FLOW MAP PRE-COMPUTATION COMPLETE")
    print("="*80)
    print(f"Processed: {processed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nFlow maps saved to: {output_dir}")
    print("="*80)
    
    # Save metadata
    metadata = {
        'task': task,
        'cameras': cameras,
        'episodes': episodes,
        'timesteps': timesteps,
        'total_samples': total_samples,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'use_gmflow': use_gmflow
    }
    
    # Save metadata per-task so parallel jobs don't clobber each other
    task_dir = os.path.join(output_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    metadata_path = os.path.join(task_dir, 'flow_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    directory = "/home/wuroderi/scratch/activations_metaworld/avdc_policy_raw"
    results_json = "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_complete.json"
    
    all_tasks = [
        "assembly-v2-goal-observable",
        "basketball-v2-goal-observable",
        "button-press-topdown-v2-goal-observable",
        "button-press-v2-goal-observable",
        "door-close-v2-goal-observable",
        "door-open-v2-goal-observable",
        "faucet-close-v2-goal-observable",
        "faucet-open-v2-goal-observable",
        "hammer-v2-goal-observable",
        "handle-press-v2-goal-observable",
        "shelf-place-v2-goal-observable"
    ]
    cameras = ['corner', 'corner2', 'corner3']
    episodes = list(range(0, 500))  # 500 episodes per task
    timesteps = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    
    # Check if specific task provided via command-line argument
    if len(sys.argv) > 1:
        # Single task mode (for parallel processing)
        task = sys.argv[1]
        if task not in all_tasks:
            print(f"\nERROR: Unknown task '{task}'")
            print(f"Available tasks: {', '.join(all_tasks)}")
            sys.exit(1)
        
        task_list = [task]
        print("\n" + "="*80)
        print("SINGLE-TASK FLOW PRE-COMPUTATION (PARALLEL MODE)")
        print("="*80)
        print(f"Task: {task}")
        print(f"Episodes: {len(episodes)}")
        print(f"Cameras: {len(cameras)}")
        print(f"Timesteps: {len(timesteps)}")
        print(f"Total samples: {len(episodes) * len(cameras) * len(timesteps):,}")
        print(f"Estimated time: ~{len(episodes) * len(cameras) * len(timesteps) * 2.5 / 3600:.1f} hours")
        print("="*80)
    else:
        # All tasks mode (sequential)
        task_list = all_tasks
        print("\n" + "="*80)
        print("MULTI-TASK FLOW PRE-COMPUTATION (SEQUENTIAL MODE)")
        print("="*80)
        print(f"Tasks: {len(task_list)}")
        print(f"Episodes per task: {len(episodes)}")
        print(f"Cameras: {len(cameras)}")
        print(f"Timesteps: {len(timesteps)}")
        print(f"Total samples: {len(task_list) * len(episodes) * len(cameras) * len(timesteps):,}")
        print(f"Estimated time: ~{len(task_list) * len(episodes) * len(cameras) * len(timesteps) * 2.5 / 3600:.1f} hours")
        print("="*80)
    
    # Load GMFlow model once
    print("\nLoading GMFlow model...")
    flow_model = data.load_gmflow_model(device)
    print("GMFlow model loaded!\n")
    
    # Process tasks
    for task_idx, task in enumerate(task_list, 1):
        if len(task_list) > 1:
            print("\n" + "="*80)
            print(f"TASK {task_idx}/{len(task_list)}: {task}")
            print("="*80)
        
        # Pre-compute flow maps for this task
        precompute_flow_maps(
            directory=directory,
            task=task,
            cameras=cameras,
            episodes=episodes,
            timesteps=timesteps,
            results_json=results_json,
            flow_model=flow_model,
            device=device,
            use_gmflow=True,
            output_dir=None  # Will use directory/flow_maps/
        )
    
    print("\n" + "="*80)
    if len(task_list) > 1:
        print("ALL TASKS COMPLETED!")
    else:
        print(f"TASK COMPLETED: {task_list[0]}")
    print("="*80)
    print("DONE! You can now train much faster by loading pre-computed flows.")
    print("Set use_cached_flows=True in train.py to use these pre-computed flows.")
    print("="*80)
