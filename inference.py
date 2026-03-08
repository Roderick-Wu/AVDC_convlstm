import os
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import model
from model import CombinedClassifier, CombinedConvLSTM
import data
from sklearn.metrics import confusion_matrix, classification_report


def run_inference(classifier_model,
                 device,
                 directory,
                 task_list,
                 camera_list,
                 episodes,
                 diffusion_time_steps,
                 results_json,
                 flow_model,
                 use_gmflow=True,
                 batch_size=16,
                 normalize_latent=False,
                 latent_mean=None,
                 latent_std=None,
                 threshold=0.5,
                 save_predictions=True,
                 output_file="predictions_combined.json"):
    """
    Run inference with combined classifier.
    
    Args:
        classifier_model: Trained CombinedClassifier or CombinedConvLSTM
        device: torch device
        directory: Base directory containing episode data
        task_list: List of tasks
        camera_list: List of cameras
        episodes: List of episode numbers
        diffusion_time_steps: List of diffusion timesteps
        results_json: Path to results JSON file
        flow_model: GMFlow or RAFT model
        use_gmflow: Whether to use GMFlow (vs RAFT)
        batch_size: Batch size for inference
        normalize_latent: Whether to normalize latent embeddings
        latent_mean: Mean for latent normalization
        latent_std: Std for latent normalization
        threshold: Classification threshold (default 0.5)
        save_predictions: Whether to save predictions to file
        output_file: Filename for saving predictions
    
    Returns:
        results: Dict with predictions, probabilities, labels, and metrics
    """
    
    classifier_model = classifier_model.to(device)
    classifier_model.eval()
    
    # Create episode info list
    print("Creating episode info list...")
    episode_info_list = data.create_episode_info_list(
        directory=directory,
        tasks=task_list,
        cameras=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=results_json
    )
    
    # Create dataset
    inference_dataset = data.CombinedLazyDataset(
        episode_info_list=episode_info_list,
        flow_model=flow_model,
        device=device,
        use_gmflow=use_gmflow,
        normalize_latent=normalize_latent,
        latent_mean=latent_mean,
        latent_std=latent_std
    )
    
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Running inference on {len(inference_dataset)} samples...")
    
    # Run inference
    all_logits = []
    all_probs = []
    all_labels = []
    all_timesteps = []
    
    with torch.no_grad():
        for batch_idx, (batch_latent, batch_flow, batch_t, batch_y) in enumerate(inference_loader):
            batch_latent = batch_latent.to(device)
            batch_flow = batch_flow.to(device)
            batch_t = batch_t.to(device)
            
            outputs = classifier_model(batch_latent, batch_flow, batch_t)  # Logits
            probs = torch.sigmoid(outputs)  # Probabilities
            
            all_logits.append(outputs.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(batch_y)
            all_timesteps.append(batch_t.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(inference_dataset)} samples")
    
    # Concatenate results
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_timesteps = torch.cat(all_timesteps, dim=0).numpy()
    
    # Make predictions
    predictions = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    print("\n=== Evaluation Results ===")
    print(f"Threshold: {threshold}")
    print(f"Total samples: {len(predictions)}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn}, FP: {fp}")
    print(f"  FN: {fn}, TP: {tp}")
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, predictions, target_names=['Failure', 'Success']))
    
    # Per-timestep analysis
    print("\n=== Per-Timestep Analysis ===")
    unique_timesteps = sorted(np.unique(all_timesteps))
    for t in unique_timesteps:
        mask = all_timesteps.flatten() == t
        t_preds = predictions[mask]
        t_labels = all_labels[mask]
        t_acc = (t_preds.flatten() == t_labels.flatten()).mean()
        print(f"  Timestep {int(t):3d}: Accuracy = {t_acc:.4f} ({mask.sum()} samples)")
    
    # Prepare results dictionary
    results = {
        'predictions': predictions.flatten().tolist(),
        'probabilities': all_probs.flatten().tolist(),
        'labels': all_labels.flatten().tolist(),
        'timesteps': all_timesteps.flatten().tolist(),
        'episode_info': episode_info_list,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        },
        'threshold': threshold
    }
    
    # Save predictions
    if save_predictions:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nPredictions saved to {output_file}")
    
    return results


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data parameters
    directory = "/home/wuroderi/scratch/single_task_basketball"
    task_list = ["basketball-v2-goal-observable"]
    camera_list = ["corner", "corner2", "corner3"]
    episodes = list(range(0, 500))  # Episodes 0-499
    diffusion_time_steps = [15, 25, 35, 45, 55, 65, 75, 85]
    results_json = "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_basketball.json"
    
    # Model selection (must match training)
    model_type = "convlstm"  # Options: "simple" or "convlstm"
    
    # Model checkpoint
    if model_type == "simple":
        checkpoint_path = "checkpoints_combined_simple/model_best.pt"
        classifier = model.CombinedClassifier(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            fusion_dim=256,
            dropout=0.3
        )
    elif model_type == "convlstm":
        checkpoint_path = "checkpoints_combined_convlstm/model_best.pt"
        classifier = model.CombinedConvLSTM(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            lstm_hidden=256,
            fusion_dim=256,
            dropout=0.3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    print(f"Loading model from {checkpoint_path}")
    classifier.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Model loaded successfully")
    
    # Load flow model
    use_gmflow = True
    if use_gmflow:
        flow_model = data.load_gmflow_model(device)
        print("Using GMFlow for optical flow computation")
    else:
        flow_model = data.load_raft_model(device)
        print("Using RAFT for optical flow computation")
    
    # Run inference
    results = run_inference(
        classifier_model=classifier,
        device=device,
        directory=directory,
        task_list=task_list,
        camera_list=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=results_json,
        flow_model=flow_model,
        use_gmflow=use_gmflow,
        batch_size=16,
        normalize_latent=False,  # Must match training
        latent_mean=None,
        latent_std=None,
        threshold=0.5,
        save_predictions=True,
        output_file=f"predictions_{model_type}.json"
    )
    
    print("\nInference completed!")
