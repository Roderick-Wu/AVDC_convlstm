import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import model
from model import CombinedClassifier, CombinedConvLSTM, random_time_masking
import data
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


def supervised_contrastive_loss(embeddings, labels, temperature=0.1, eps=1e-8):
    """
    Supervised contrastive loss over a batch.

    Pulls embeddings with the same label together and pushes different labels apart.
    Returns zero when no valid positive pairs exist in the batch.
    """
    if embeddings.size(0) < 2:
        return embeddings.new_tensor(0.0)

    labels = labels.view(-1).long()
    z = F.normalize(embeddings, dim=1)

    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = z.size(0)
    self_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

    if pos_mask.sum() == 0:
        return embeddings.new_tensor(0.0)

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / pos_counts.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


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
          use_contrastive_loss=False,
          contrastive_weight=0.1,
          contrastive_temperature=0.1,
          contrastive_warmup_epochs=5,
          plot_file="loss_curve_combined.png", 
          model_save_path="checkpoints_combined",
          grad_accumulation_steps=1
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
        use_contrastive_loss: If True, add supervised contrastive loss on final embedding
        contrastive_weight: Weight for contrastive term in total loss
        contrastive_temperature: Temperature for supervised contrastive loss
        contrastive_warmup_epochs: Number of initial epochs to linearly ramp contrastive
            weight from 0 to contrastive_weight. Set 0 to disable warmup.
        plot_file: Filename for saving loss curves
        model_save_path: Directory to save checkpoints
        grad_accumulation_steps: Accumulate gradients over N batches before stepping.
            Effective batch size = batch_size × grad_accumulation_steps.
            Larger effective batches reduce gradient noise from class imbalance.
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
    optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Halve LR if val loss doesn't improve for 3 epochs; helps escape the epoch-1 plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    effective_batch = batch_size * grad_accumulation_steps
    print(f"Effective batch size: {effective_batch} "
          f"(batch_size={batch_size} × accumulation_steps={grad_accumulation_steps})", flush=True)
    print(f"Contrastive loss: {'ON' if use_contrastive_loss else 'OFF'}"
            f" (weight={contrastive_weight}, temperature={contrastive_temperature}, "
            f"warmup_epochs={contrastive_warmup_epochs})", flush=True)

    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    lowest_val_loss, best_epoch = float('inf'), 0
    
    for epoch in range(epochs):
        if use_contrastive_loss:
            if contrastive_warmup_epochs > 0:
                contrastive_scale = min(1.0, (epoch + 1) / contrastive_warmup_epochs)
            else:
                contrastive_scale = 1.0
            current_contrastive_weight = contrastive_weight * contrastive_scale
        else:
            current_contrastive_weight = 0.0

        # Training
        classifier_model.train()
        predictions, labels, total_loss = [], [], 0
        total_cls_loss, total_ctr_loss = 0, 0
        optimizer.zero_grad()

        for batch_idx, (batch_latent, batch_flow, batch_t, batch_y) in enumerate(train_loader):
            batch_latent = batch_latent.to(device)
            batch_flow = batch_flow.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)

            # Optional: apply time masking for regularization
            if time_mask_prob > 0:
                batch_latent = random_time_masking(batch_latent, mask_prob=time_mask_prob)
                batch_flow = random_time_masking(batch_flow, mask_prob=time_mask_prob)

            if use_contrastive_loss:
                outputs, embeddings = classifier_model(
                    batch_latent, batch_flow, batch_t, return_embedding=True
                )
            else:
                outputs = classifier_model(batch_latent, batch_flow, batch_t)
                embeddings = None
            predictions.append(outputs.detach().cpu())
            labels.append(batch_y.detach().cpu())

            # Scale loss so accumulated gradients match a single large-batch gradient
            cls_loss = loss_fn(outputs, batch_y)
            ctr_loss = (
                supervised_contrastive_loss(
                    embeddings, batch_y, temperature=contrastive_temperature
                ) if use_contrastive_loss else outputs.new_tensor(0.0)
            )

            combined_loss = cls_loss + current_contrastive_weight * ctr_loss
            loss = combined_loss / grad_accumulation_steps
            loss.backward()
            total_loss += combined_loss.item()
            total_cls_loss += cls_loss.item()
            total_ctr_loss += ctr_loss.item()

            is_last_batch = (batch_idx + 1 == len(train_loader))
            if (batch_idx + 1) % grad_accumulation_steps == 0 or is_last_batch:
                # Clip gradients — essential for LSTM stability
                torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                    f"Loss: {combined_loss.item():.4f} "
                    f"(cls: {cls_loss.item():.4f}, ctr: {ctr_loss.item():.4f}, "
                    f"ctr_w: {current_contrastive_weight:.4f})"
                )

        train_losses.append(total_loss / len(train_loader))
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        train_accuracy = ((torch.sigmoid(predictions) > 0.5) == (labels > 0.5)).float().mean()
        train_accuracies.append(train_accuracy.item())
        
        # Validation
        classifier_model.eval()
        predictions, labels, val_loss = [], [], 0
        val_cls_loss, val_ctr_loss = 0, 0
        with torch.no_grad():
            for batch_latent, batch_flow, batch_t, batch_y in val_loader:
                batch_latent = batch_latent.to(device)
                batch_flow = batch_flow.to(device)
                batch_t = batch_t.to(device)
                batch_y = batch_y.to(device)

                if use_contrastive_loss:
                    outputs, embeddings = classifier_model(
                        batch_latent, batch_flow, batch_t, return_embedding=True
                    )
                else:
                    outputs = classifier_model(batch_latent, batch_flow, batch_t)
                    embeddings = None

                predictions.append(outputs.detach().cpu())
                labels.append(batch_y.detach().cpu())

                v_cls = loss_fn(outputs, batch_y)
                v_ctr = (
                    supervised_contrastive_loss(
                        embeddings, batch_y, temperature=contrastive_temperature
                    ) if use_contrastive_loss else outputs.new_tensor(0.0)
                )
                v_total = v_cls + current_contrastive_weight * v_ctr

                val_loss += v_total.item()
                val_cls_loss += v_cls.item()
                val_ctr_loss += v_ctr.item()
        
        val_losses.append(val_loss / len(val_loader))
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        val_accuracy = ((torch.sigmoid(predictions) > 0.5) == (labels > 0.5)).float().mean()
        val_accuracies.append(val_accuracy.item())
        
        # Step scheduler based on val loss
        scheduler.step(val_losses[-1])
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}")
        print(f"    Train cls: {total_cls_loss / len(train_loader):.4f}, Train ctr: {total_ctr_loss / len(train_loader):.4f}")
        print(f"  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")
        print(f"    Val cls: {val_cls_loss / len(val_loader):.4f}, Val ctr: {val_ctr_loss / len(val_loader):.4f}")
        print(f"  Contrastive weight (current): {current_contrastive_weight:.4f}")
        print(f"  LR: {current_lr:.2e}", flush=True)

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
    epochs = 100
    batch_size = 64  # Adjust based on GPU memory
    learning_rate = 4e-6
    val_split = 0.1  # 10% validation split for multi-task training
    grad_accumulation_steps = 1

    # Optional embedding contrastive regularization
    use_contrastive_loss = True
    contrastive_weight = 0.03
    contrastive_temperature = 0.1
    contrastive_warmup_epochs = 5
    
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
        use_contrastive_loss=use_contrastive_loss,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=contrastive_temperature,
        contrastive_warmup_epochs=contrastive_warmup_epochs,
        plot_file="loss_curve.png",
        model_save_path=model_save_path,
        grad_accumulation_steps=grad_accumulation_steps
    )
