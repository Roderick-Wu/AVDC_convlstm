"""
Run inference with combined classifier and cache all scores to JSON for fast experimentation.
This allows quick testing of different thresholds without re-running inference.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import sys
import os

# Import from local modules
import model
from model import CombinedClassifier, CombinedConvLSTM
import data


class CombinedInferenceCache:
    """Run inference with combined classifier and cache all scores."""
    
    def __init__(self, model_path, model_type='convlstm', device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'simple' or 'convlstm'
            device: torch device
        """
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        
        # Models
        self.classifier = None
        self.flow_model = None
        
    def load_models(self):
        """Load combined classifier and flow computation model."""
        print("\n" + "="*80)
        print("Loading Models")
        print("="*80)
        
        # Load combined classifier
        print(f"\n1. Loading Combined Classifier ({self.model_type})...")
        if self.model_type == 'simple':
            self.classifier = CombinedClassifier(
                latent_channels=640,
                flow_channels=2,
                hidden_dim=128,
                fusion_dim=256,
                dropout=0.3
            )
        elif self.model_type == 'convlstm':
            self.classifier = CombinedConvLSTM(
                latent_channels=640,
                flow_channels=2,
                hidden_dim=128,
                lstm_hidden=256,
                fusion_dim=256,
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.classifier.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        print(f"   ✓ Combined classifier loaded from {self.model_path}")
        
        # Load GMFlow for optical flow computation
        print("\n2. Loading GMFlow (UniMatch) for flow computation...")
        self.flow_model = data.load_gmflow_model(self.device)
        print("   ✓ GMFlow loaded")
        
        print("\n" + "="*80)
        print("All models loaded!")
        print("="*80)
    
    def predict(self, episode_info_list):
        """
        Run inference on a list of episodes.
        
        Args:
            episode_info_list: List of dicts with episode information
        
        Returns:
            Array of probabilities (after sigmoid)
        """
        # Create dataset
        dataset = data.CombinedLazyDataset(
            episode_info_list=episode_info_list,
            flow_model=self.flow_model,
            device=self.device,
            use_gmflow=True,
            normalize_latent=False  # Must match training
        )
        
        all_probs = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1}/{len(dataset)} samples...")
                
                latent, flow, timestep, label = dataset[i]
                
                # Add batch dimension
                latent = latent.unsqueeze(0).to(self.device)
                flow = flow.unsqueeze(0).to(self.device)
                timestep = timestep.unsqueeze(0).to(self.device)
                
                # Get prediction
                logits = self.classifier(latent, flow, timestep)
                prob = torch.sigmoid(logits).squeeze().cpu().item()
                
                all_probs.append(prob)
        
        return np.array(all_probs)
    
    def cache_all_scores(self, directory, task, cameras, episodes, timesteps, results_json):
        """
        Run inference across all timesteps and cache scores.
        """
        print("\n" + "="*80)
        print(f"Caching Combined Classifier Inference Scores")
        print(f"Task: {task}")
        print(f"Episodes: {episodes[0]}-{episodes[-1]} ({len(episodes)} episodes)")
        print(f"Cameras: {cameras}")
        print(f"Timesteps: {timesteps}")
        print("="*80)
        
        with open(results_json, 'r') as f:
            results_dict = json.load(f)
        
        # Build episode mapping (camera, episode, label)
        print("\nBuilding episode mapping...")
        episode_mapping = []
        for camera in cameras:
            successes = results_dict[task]['success'][camera]
            for episode in episodes:
                episode_mapping.append({
                    'camera': camera,
                    'episode': int(episode),
                    'label': successes[episode]
                })
        
        print(f"  Total samples per timestep: {len(episode_mapping)}")
        
        # Count distribution
        num_failures = sum(1 for ep in episode_mapping if ep['label'] == 0)
        num_successes = sum(1 for ep in episode_mapping if ep['label'] == 1)
        print(f"  Failures: {num_failures} ({100*num_failures/len(episode_mapping):.1f}%)")
        print(f"  Successes: {num_successes} ({100*num_successes/len(episode_mapping):.1f}%)")
        
        # Cache structure
        cache_data = {
            'metadata': {
                'task': task,
                'cameras': cameras,
                'episodes': episodes,
                'timesteps': timesteps,
                'total_samples': len(episode_mapping),
                'results_json': results_json,
                'directory': directory,
                'model_type': self.model_type,
                'model_path': self.model_path
            },
            'episode_mapping': episode_mapping,
            'timestep_scores': {}
        }
        
        # Run inference for each timestep
        for timestep in timesteps:
            print(f"\n{'='*80}")
            print(f"Processing Diffusion Timestep: {timestep}")
            print(f"{'='*80}")
            
            # Create episode info list for this timestep
            episode_info_list = data.create_episode_info_list(
                directory=directory,
                tasks=[task],
                cameras=cameras,
                episodes=episodes,
                diffusion_time_steps=[timestep],
                results_json=results_json
            )
            
            print(f"  Running inference on {len(episode_info_list)} samples...")
            
            # Get predictions
            probs = self.predict(episode_info_list)
            
            # Store scores for this timestep
            cache_data['timestep_scores'][str(timestep)] = {
                'probs': probs.tolist(),
                'labels': [info['label'] for info in episode_info_list]
            }
            
            # Compute accuracy at default threshold (0.5)
            predictions = (probs >= 0.5).astype(int)
            labels = np.array([info['label'] for info in episode_info_list])
            accuracy = (predictions == labels).mean()
            
            print(f"  ✓ Cached {len(probs)} scores for timestep {timestep}")
            print(f"    Accuracy @ 0.5 threshold: {accuracy:.4f}")
        
        print("\n" + "="*80)
        print("Inference complete! All scores cached.")
        print("="*80)
        
        return cache_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    directory = "/home/wuroderi/scratch/single_task_basketball"
    results_json = "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_basketball.json"
    task = "basketball-v2-goal-observable"
    cameras = ['corner', 'corner2', 'corner3']
    
    # Test episodes (800-999) - matches training split (0-800 train)
    episodes = list(range(800, 1000))
    
    # Diffusion timesteps
    timesteps = [15, 25, 35, 45, 55, 65, 75, 85]
    
    # Model configuration
    model_type = 'convlstm'  # 'simple' or 'convlstm'
    model_path = f'checkpoints_combined_{model_type}/model_best.pt'
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Please train the model first or update the model_path.")
        sys.exit(1)
    
    # Create inference cache
    cache = CombinedInferenceCache(
        model_path=model_path,
        model_type=model_type,
        device=device
    )
    cache.load_models()
    
    # Run inference and cache all scores
    print("\nRunning inference (this will take some time due to flow computation)...")
    print("Note: Flow maps are computed on-the-fly for each sample.")
    cache_data = cache.cache_all_scores(
        directory=directory,
        task=task,
        cameras=cameras,
        episodes=episodes,
        timesteps=timesteps,
        results_json=results_json
    )
    
    # Save cached scores
    output_file = f'combined_scores_cache_{task.split("-")[0]}_{model_type}.json'
    print(f"\nSaving cached scores to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"CACHE SAVED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nFile: {output_file}")
    print(f"Total samples per timestep: {cache_data['metadata']['total_samples']}")
    print(f"Timesteps: {cache_data['metadata']['timesteps']}")
    print(f"\nYou can now use 'experiment_combined_cache.py' to quickly test different")
    print(f"thresholds and simulate sequential filtering without re-running inference!")
    print(f"{'='*80}")
