import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model
from model import CombinedClassifier, CombinedConvLSTM, random_time_masking
import data
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


def train_combined_model(classifier_model, 
          epochs, 
          batch_size, 
          learning_rate, 
          device, 
          val_split, 
          directory, 
          task_list, 
          camera_list, 
          episodes, 
          diffusion_time_steps, 
          results_json,
          flow_model=None,
          use_gmflow=True,
          use_cached_flows=False,
          flow_cache_dir=None,
          skip_file_check=False,
          load_in_memory=True,
          time_mask_prob=0.0,
          normalize_latent=False,
          plot_file="loss_curve_combined.png", 
          model_save_path="checkpoints_combined"
    ):
    """
    Train combined classifier that uses both latent embeddings and flow maps.
    
    Args:
        classifier_model: CombinedClassifier or CombinedConvLSTM instance
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        device: torch device
        val_split: Validation split ratio
        directory: Base directory containing episode data
        task_list: List of tasks to train on
        camera_list: List of camera views
        episodes: List of episode numbers
        diffusion_time_steps: List of diffusion timesteps
        results_json: Path to results JSON file
        flow_model: GMFlow or RAFT model (only needed if use_cached_flows=False)
        use_gmflow: If True, uses GMFlow; if False, uses RAFT (ignored if using cached flows)
        use_cached_flows: If True, loads pre-computed flows from disk (MUCH FASTER!)
        flow_cache_dir: Directory containing cached flows (default: directory/flow_maps/)
        time_mask_prob: Probability of masking timesteps during training
        normalize_latent: Whether to normalize latent embeddings
        plot_file: Filename for saving loss curves
        model_save_path: Directory to save checkpoints
    """
    
    classifier_model = classifier_model.to(device)
    
    # Create episode info list for lazy loading
    print("Creating episode info list...")
    episode_info_list = data.create_episode_info_list(
        directory=directory,
        tasks=task_list,
        cameras=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=results_json,
        skip_file_check=skip_file_check
    )
    
    # Split by episodes to prevent data leakage
    train_info_list, val_info_list = data.episode_aware_split(
        episode_info_list,
        val_split=val_split,
        num_cameras=len(camera_list),
        num_diffusion_steps=len(diffusion_time_steps)
    )
    
    # Compute latent statistics for normalization if needed
    latent_mean, latent_std = None, None
    if normalize_latent:
        latent_mean, latent_std = data.compute_latent_statistics(train_info_list, sample_size=100)
    
    # Create datasets: use cached flows if available (MUCH faster!)
    if use_cached_flows:
        print("\n✓ Using PRE-COMPUTED flow maps (fast training!)")
        if flow_cache_dir is None:
            flow_cache_dir = os.path.join(directory, 'flow_maps')
        
        if not os.path.exists(flow_cache_dir):
            raise FileNotFoundError(
                f"Flow cache directory not found: {flow_cache_dir}\n"
                f"Run precompute_flows.py first to generate cached flows!"
            )
        
        print(f"  Flow cache directory: {flow_cache_dir}")
        
        if load_in_memory:
            print("\n📦 Loading ENTIRE dataset into RAM (fast epochs, ~15 min startup)...")
            print("  Training set:")
            train_dataset = data.InMemoryDataset(
                episode_info_list=train_info_list,
                flow_cache_dir=flow_cache_dir,
                normalize_latent=normalize_latent,
                latent_mean=latent_mean,
                latent_std=latent_std,
                num_load_workers=64
            )
            print("  Validation set:")
            val_dataset = data.InMemoryDataset(
                episode_info_list=val_info_list,
                flow_cache_dir=flow_cache_dir,
                normalize_latent=normalize_latent,
                latent_mean=latent_mean,
                latent_std=latent_std,
                num_load_workers=64
            )
        else:
            train_dataset = data.CachedFlowDataset(
                episode_info_list=train_info_list,
                flow_cache_dir=flow_cache_dir,
                normalize_latent=normalize_latent,
                latent_mean=latent_mean,
                latent_std=latent_std
            )
            val_dataset = data.CachedFlowDataset(
                episode_info_list=val_info_list,
                flow_cache_dir=flow_cache_dir,
                normalize_latent=normalize_latent,
                latent_mean=latent_mean,
                latent_std=latent_std
            )
    else:
        print("\n⚠ Computing flows ON-THE-FLY (very slow!)")
        print("  Consider running precompute_flows.py first for much faster training.")
        
        if flow_model is None:
            raise ValueError("flow_model must be provided when use_cached_flows=False")
        
        train_dataset = data.CombinedLazyDataset(
            episode_info_list=train_info_list,
            flow_model=flow_model,
            device=device,
            use_gmflow=use_gmflow,
            normalize_latent=normalize_latent,
            latent_mean=latent_mean,
            latent_std=latent_std
        )
        
        val_dataset = data.CombinedLazyDataset(
            episode_info_list=val_info_list,
            flow_model=flow_model,
            device=device,
            use_gmflow=use_gmflow,
            normalize_latent=normalize_latent,
            latent_mean=latent_mean,
            latent_std=latent_std
        )
    
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
    
    # Create checkpoint directory immediately so we can verify training started
    os.makedirs(model_save_path, exist_ok=True)
    print(f"Checkpoint directory: {model_save_path}")
    
    # Compute class weights
    train_labels = np.array([info['label'] for info in train_info_list])
    num_negative = np.sum(train_labels == 0)
    num_positive = np.sum(train_labels == 1)
    pos_weight = torch.tensor([num_negative / num_positive], device=device)
    
    print(f"\nClass distribution:")
    print(f"  Negative (failure): {num_negative} ({100*num_negative/len(train_labels):.2f}%)")
    print(f"  Positive (success): {num_positive} ({100*num_positive/len(train_labels):.2f}%)")
    print(f"  Pos weight: {pos_weight.item():.4f}")
    
    # Create data loaders
    # num_workers=0: data is already in RAM, workers add overhead not benefit
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=False)
    
    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    lowest_val_loss, best_epoch = float('inf'), 0
    
    for epoch in range(epochs):
        # Training
        classifier_model.train()
        predictions, labels, total_loss = [], [], 0
        
        for batch_idx, (batch_latent, batch_flow, batch_t, batch_y) in enumerate(train_loader):
            batch_latent = batch_latent.to(device)
            batch_flow = batch_flow.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)
            
            # Optional: apply time masking for regularization
            if time_mask_prob > 0:
                batch_latent = random_time_masking(batch_latent, mask_prob=time_mask_prob)
                batch_flow = random_time_masking(batch_flow, mask_prob=time_mask_prob)
            
            optimizer.zero_grad()
            outputs = classifier_model(batch_latent, batch_flow, batch_t)
            predictions.append(outputs.detach().cpu())
            labels.append(batch_y.detach().cpu())
            
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_losses.append(total_loss / len(train_loader))
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        train_accuracy = ((torch.sigmoid(predictions) > 0.5) == (labels > 0.5)).float().mean()
        train_accuracies.append(train_accuracy.item())
        
        # Validation
        classifier_model.eval()
        predictions, labels, val_loss = [], [], 0
        with torch.no_grad():
            for batch_latent, batch_flow, batch_t, batch_y in val_loader:
                batch_latent = batch_latent.to(device)
                batch_flow = batch_flow.to(device)
                batch_t = batch_t.to(device)
                batch_y = batch_y.to(device)
                
                outputs = classifier_model(batch_latent, batch_flow, batch_t)
                predictions.append(outputs.detach().cpu())
                labels.append(batch_y.detach().cpu())
                val_loss += loss_fn(outputs, batch_y).item()
        
        val_losses.append(val_loss / len(val_loader))
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        val_accuracy = ((torch.sigmoid(predictions) > 0.5) == (labels > 0.5)).float().mean()
        val_accuracies.append(val_accuracy.item())
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")
        
        # Save best model
        if val_losses[-1] < lowest_val_loss:
            lowest_val_loss = val_losses[-1]
            best_epoch = epoch + 1
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(classifier_model.state_dict(), os.path.join(model_save_path, "model_best.pt"))
            print(f"  ✓ Best model saved (val loss: {lowest_val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_save_path, f"model_epoch_{epoch+1}.pt")
            torch.save(classifier_model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, plot_file))
    print(f"Loss curve saved to {os.path.join(model_save_path, plot_file)}")
    
    print(f"\nTraining completed!")
    print(f"Best model at epoch {best_epoch} with val loss: {lowest_val_loss:.4f}")


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data parameters - ALL TASKS
    directory = "/home/wuroderi/scratch/activations_metaworld/avdc_policy_raw"
    task_list = [
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
    camera_list = ["corner", "corner2", "corner3"]
    episodes = list(range(0, 500))  # 500 episodes per task
    diffusion_time_steps = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    results_json = "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_complete.json"
    
    # Dataset size: 11 tasks × 500 episodes × 3 cameras × 10 timesteps = 165,000 samples
    # With val_split=0.1: ~148,500 train, ~16,500 val
    
    # Training parameters
    epochs = 50
    batch_size = 16  # Adjust based on GPU memory
    learning_rate = 1e-4
    val_split = 0.1  # 10% validation split for multi-task training
    
    # IMPORTANT: Use pre-computed flows for fast training!
    # Set to True to use cached flows (MUCH faster - recommended!)
    # Set to False to compute flows on-the-fly (very slow - only for testing)
    # NOTE: For all-tasks training, you need to precompute flows for ALL tasks first!
    #       Update precompute_flows.py with the same task_list and run it before training.
    use_cached_flows = True
    
    # Skip os.path.exists() checks during indexing - avoids ~495k stat calls on scratch
    # filesystem which would take hours before training even starts.
    skip_file_check = True
    
    # Flow model (only needed if use_cached_flows=False)
    flow_model = None
    if not use_cached_flows:
        print("⚠ WARNING: Computing flows on-the-fly is VERY slow!")
        print("  Run precompute_flows.py first, then set use_cached_flows=True")
        use_gmflow = True
        if use_gmflow:
            flow_model = data.load_gmflow_model(device)
            print("Using GMFlow for optical flow computation")
        else:
            flow_model = data.load_raft_model(device)
            print("Using RAFT for optical flow computation")
    else:
        print("✓ Using pre-computed flow maps for fast training")
        flow_cache_dir = os.path.join(directory, 'flow_maps')
        if not os.path.exists(flow_cache_dir):
            print(f"\n❌ ERROR: Flow cache not found at {flow_cache_dir}")
            print("Run precompute_flows.py first to generate cached flows!")
            print("Example:")
            print("  python precompute_flows.py")
            exit(1)
    
    # Model selection: choose one
    model_type = "convlstm"  # Options: "simple" or "convlstm"
    
    # Create model
    if model_type == "simple":
        classifier = model.CombinedClassifier(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            fusion_dim=256,
            dropout=0.3
        )
        model_name = "combined_simple"
    elif model_type == "convlstm":
        classifier = model.CombinedConvLSTM(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            lstm_hidden=256,
            fusion_dim=256,
            dropout=0.3
        )
        model_name = "combined_convlstm"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Created model: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Train - ALL TASKS MODEL
    model_save_path = f"checkpoints_{model_name}_alltasks"
    train_combined_model(
        classifier_model=classifier,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        val_split=val_split,
        directory=directory,
        task_list=task_list,
        camera_list=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=results_json,
        flow_model=flow_model,
        use_gmflow=True,  # Only used if use_cached_flows=False
        use_cached_flows=use_cached_flows,
        flow_cache_dir=os.path.join(directory, 'flow_maps') if use_cached_flows else None,
        load_in_memory=True,    # Load all data into RAM — eliminates disk I/O during training
        skip_file_check=skip_file_check,
        time_mask_prob=0.0,  # Set to 0.15 to enable time masking
        normalize_latent=False,  # Set to True to normalize latent embeddings
        plot_file="loss_curve.png",
        model_save_path=model_save_path
    )
