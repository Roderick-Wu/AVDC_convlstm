import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import threading


class CombinedDataset(Dataset):
    """Dataset that returns both latent embeddings and flow maps."""
    def __init__(self, latent_X, flow_X, t, y):
        """
        Args:
            latent_X: List of latent embedding tensors (640, 7, 8, 8)
            flow_X: List of flow tensors (2, 7, H, W)
            t: List of diffusion timesteps
            y: List of labels
        """
        self.latent_X = latent_X
        self.flow_X = flow_X
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.latent_X[idx], self.flow_X[idx], self.t[idx], self.y[idx]


class CombinedLazyDataset(Dataset):
    """
    Lazy-loading dataset that loads and computes data on-the-fly.
    Loads latent embeddings from disk and computes flow maps from x0 predictions.
    """
    def __init__(self, episode_info_list, flow_model, device, use_gmflow=True, 
                 normalize_latent=False, latent_mean=None, latent_std=None):
        """
        Args:
            episode_info_list: List of dicts with keys: 'latent_path', 'x0_path', 
                              'condition_path', 'timestep', 'label'
            flow_model: GMFlow or RAFT model for computing optical flow
            device: torch device
            use_gmflow: If True, uses GMFlow; if False, uses RAFT
            normalize_latent: Whether to normalize latent embeddings
            latent_mean: Mean for latent normalization
            latent_std: Std for latent normalization
        """
        self.episode_info_list = episode_info_list
        self.flow_model = flow_model
        self.device = device
        self.use_gmflow = use_gmflow
        self.normalize_latent = normalize_latent
        self.latent_mean = latent_mean
        self.latent_std = latent_std

    def __len__(self):
        return len(self.episode_info_list)

    def __getitem__(self, idx):
        info = self.episode_info_list[idx]
        
        # 1. Load latent embeddings
        latent = np.load(info['latent_path'])  # (1, 640, 7, 8, 8)
        latent = torch.from_numpy(latent).float()
        if latent.ndim == 5:
            latent = latent[0]  # (640, 7, 8, 8)
        
        # Apply normalization if requested
        if self.normalize_latent and self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean) / (self.latent_std + 1e-8)
        
        # 2. Load x0 prediction video (7 frames)
        x0_frames = load_video_frames(info['x0_path'])  # List of (C, H, W) tensors
        
        # 3. Load condition image from final video
        condition_frame = load_condition_frame(info['condition_path'])  # (C, H, W)
        
        # 4. Prepend condition to x0 frames to get 8 frames total
        video_frames = [condition_frame] + x0_frames  # 8 frames
        assert len(video_frames) == 8, f"Expected 8 frames, got {len(video_frames)}"
        
        # 5. Compute flow maps between consecutive frames (8 frames → 7 flows)
        flow_maps = compute_flow_sequence(
            video_frames, 
            self.flow_model, 
            self.device, 
            self.use_gmflow
        )  # (2, 7, H, W)
        
        # 6. Prepare timestep and label
        t = torch.tensor([info['timestep']], dtype=torch.float32)
        y = torch.tensor([info['label']], dtype=torch.float32)
        
        return latent, flow_maps, t, y


class CachedFlowDataset(Dataset):
    """
    Fast dataset that loads pre-computed flow maps from disk.
    Use this after running precompute_flows.py to avoid recomputing flows every epoch.
    """
    def __init__(self, episode_info_list, flow_cache_dir, normalize_latent=False, latent_mean=None, latent_std=None):
        """
        Args:
            episode_info_list: List of dicts with episode information
            flow_cache_dir: Directory containing pre-computed flow maps
            normalize_latent: Whether to normalize latent embeddings
            latent_mean: Mean for latent normalization
            latent_std: Std for latent normalization
        """
        self.episode_info_list = episode_info_list
        self.flow_cache_dir = flow_cache_dir
        self.normalize_latent = normalize_latent
        self.latent_mean = latent_mean
        self.latent_std = latent_std

    def __len__(self):
        return len(self.episode_info_list)

    def __getitem__(self, idx):
        info = self.episode_info_list[idx]
        
        # 1. Load latent embeddings (fast)
        latent = np.load(info['latent_path'])  # (1, 640, 7, 8, 8)
        latent = torch.from_numpy(latent).float()
        if latent.ndim == 5:
            latent = latent[0]  # (640, 7, 8, 8)
        
        # Apply normalization if requested
        if self.normalize_latent and self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean) / (self.latent_std + 1e-8)
        
        # 2. Load pre-computed flow maps (fast!)
        # Path structure: flow_cache_dir/{task}/{camera}/episode_{episode}/flow_t{timestep}.npy
        flow_path = os.path.join(
            self.flow_cache_dir,
            info['task'],
            info['camera'],
            f"episode_{info['episode']:03d}",
            f"flow_t{info['timestep']:03d}.npy"
        )
        
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Pre-computed flow not found: {flow_path}\n"
                                   f"Run precompute_flows.py first!")
        
        flow_maps = np.load(flow_path)  # (2, 7, H, W)
        flow_maps = torch.from_numpy(flow_maps).float()
        
        # 3. Prepare timestep and label
        t = torch.tensor([info['timestep']], dtype=torch.float32)
        y = torch.tensor([info['label']], dtype=torch.float32)
        
        return latent, flow_maps, t, y


class InMemoryDataset(Dataset):
    """
    Loads entire dataset into RAM once at startup to eliminate all per-batch disk I/O.
    Requires large RAM (request 350GB+ on SLURM) but makes epochs extremely fast.
    
    Flow maps (~141 GB) + latents (~88 GB) = ~229 GB total.
    Pre-allocated as contiguous float32 numpy arrays for maximum throughput.
    """

    def __init__(self, episode_info_list, flow_cache_dir, normalize_latent=False,
                 latent_mean=None, latent_std=None, num_load_workers=16):
        """
        Args:
            episode_info_list: List of dicts with episode information
            flow_cache_dir: Directory containing pre-computed flow maps
            num_load_workers: Parallel threads for loading (default 64 recommended for Lustre)
        """
        n = len(episode_info_list)
        print(f"  Loading {n:,} samples into RAM with {num_load_workers} threads...", flush=True)

        # ---- probe shapes from first sample ----
        info0 = episode_info_list[0]
        lat0 = np.load(info0['latent_path'])
        if lat0.ndim == 5:
            lat0 = lat0[0]
        flow_path0 = os.path.join(
            flow_cache_dir, info0['task'], info0['camera'],
            f"episode_{info0['episode']:03d}",
            f"flow_t{info0['timestep']:03d}.npy"
        )
        flow0 = np.load(flow_path0)

        latent_shape = lat0.shape   # (640, 7, 8, 8)
        flow_shape   = flow0.shape  # (2, 7, H, W)

        latent_mem_gb = n * lat0.astype(np.float32).nbytes / 1024**3
        flow_mem_gb   = n * flow0.astype(np.float32).nbytes / 1024**3
        print(f"  Latent array:  {latent_mem_gb:.1f} GB  {latent_shape}")
        print(f"  Flow array:    {flow_mem_gb:.1f} GB  {flow_shape}")
        print(f"  Total:         {latent_mem_gb + flow_mem_gb:.1f} GB")
        print(f"  Estimated load time: ~{(latent_mem_gb + flow_mem_gb) / 2:.0f}-{(latent_mem_gb + flow_mem_gb):.0f} min", flush=True)

        # ---- pre-allocate contiguous float32 buffers ----
        self.latents   = np.empty((n, *latent_shape), dtype=np.float32)
        self.flows     = np.empty((n, *flow_shape),   dtype=np.float32)
        self.timesteps = np.empty(n, dtype=np.float32)
        self.labels    = np.empty(n, dtype=np.float32)

        # ---- parallel load via threads ----
        # np.load releases the GIL during file I/O, so threads provide true
        # parallelism here and hide per-file open latency on Lustre scratch.
        lock = threading.Lock()
        progress = [0]

        def load_one(args):
            idx, info = args
            lat = np.load(info['latent_path'])
            if lat.ndim == 5:
                lat = lat[0]
            self.latents[idx] = lat.astype(np.float32)

            fp = os.path.join(
                flow_cache_dir, info['task'], info['camera'],
                f"episode_{info['episode']:03d}",
                f"flow_t{info['timestep']:03d}.npy"
            )
            self.flows[idx] = np.load(fp).astype(np.float32)

            self.timesteps[idx] = float(info['timestep'])
            self.labels[idx]    = float(info['label'])

            with lock:
                progress[0] += 1
                if progress[0] % 5000 == 0:
                    print(f"    {progress[0]:,}/{n:,} loaded...", flush=True)

        with ThreadPoolExecutor(max_workers=num_load_workers) as ex:
            list(ex.map(load_one, enumerate(episode_info_list)))

        print(f"  All {n:,} samples loaded into RAM.", flush=True)

        # ---- optional latent normalisation ----
        if normalize_latent and latent_mean is not None and latent_std is not None:
            self.latents = (self.latents - latent_mean) / (latent_std + 1e-8)

        # convert to torch tensors (zero-copy view of the numpy buffers)
        self.latents   = torch.from_numpy(self.latents)
        self.flows     = torch.from_numpy(self.flows)
        self.timesteps = torch.from_numpy(self.timesteps)
        self.labels    = torch.from_numpy(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.latents[idx],
            self.flows[idx],
            self.timesteps[idx].unsqueeze(0),
            self.labels[idx].unsqueeze(0),
        )


class LatentRawVideoDataset(Dataset):
    """
    Dataset for latent embeddings + raw x0 videos.

    Returns:
        latent: (640, 7, 8, 8)
        video: (3, 8, H, W) where frame 0 is condition image
        t: (1,)
        y: (1,)
    """

    def __init__(self, episode_info_list, normalize_latent=False, latent_mean=None,
                 latent_std=None, video_size=128):
        self.episode_info_list = episode_info_list
        self.normalize_latent = normalize_latent
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.video_size = video_size

    def __len__(self):
        return len(self.episode_info_list)

    def __getitem__(self, idx):
        info = self.episode_info_list[idx]

        latent = np.load(info['latent_path'])
        latent = torch.from_numpy(latent).float()
        if latent.ndim == 5:
            latent = latent[0]

        if self.normalize_latent and self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean) / (self.latent_std + 1e-8)

        x0_frames = load_video_frames(info['x0_path'])
        if len(x0_frames) == 8:
            frames = x0_frames
        elif len(x0_frames) == 7:
            condition = load_condition_frame(info['condition_path'])
            frames = [condition] + x0_frames
        else:
            raise ValueError(
                f"Expected x0 gif with 7 or 8 frames, got {len(x0_frames)}: {info['x0_path']}"
            )

        video_tchw = torch.stack(frames, dim=0)  # (T, C, H, W)
        if self.video_size is not None and (
            video_tchw.shape[-2] != self.video_size or video_tchw.shape[-1] != self.video_size
        ):
            video_tchw = F.interpolate(
                video_tchw,
                size=(self.video_size, self.video_size),
                mode='bilinear',
                align_corners=False,
            )
        video = video_tchw.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

        t = torch.tensor([info['timestep']], dtype=torch.float32)
        y = torch.tensor([info['label']], dtype=torch.float32)

        return latent, video, t, y


def load_video_frames(video_path):
    """
    Load all frames from a GIF video.
    
    Args:
        video_path: Path to .gif file
    
    Returns:
        List of tensors, each (C, H, W)
    """
    frames = []
    with Image.open(video_path) as video:
        for frame_idx in range(video.n_frames):
            video.seek(frame_idx)
            frame = video.convert("RGB")
            frame = np.array(frame)  # (H, W, C)
            frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
            frames.append(torch.from_numpy(frame).float())
    return frames


def load_condition_frame(video_path):
    """
    Load the first frame (condition image) from the final output video.
    
    Args:
        video_path: Path to metaworld_sample.gif
    
    Returns:
        Tensor (C, H, W)
    """
    with Image.open(video_path) as video:
        video.seek(0)  # First frame
        frame = video.convert("RGB")
        frame = np.array(frame)  # (H, W, C)
        frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
    return torch.from_numpy(frame).float()


def compute_optical_flow_gmflow(frame1, frame2, gmflow_model, device):
    """
    Compute optical flow between two frames using GMFlow (UniMatch).
    
    Args:
        frame1: Tensor (C, H, W) - first frame
        frame2: Tensor (C, H, W) - second frame
        gmflow_model: GMFlow/UniMatch model
        device: torch device
    
    Returns:
        flow: Tensor (2, H, W) - optical flow from frame1 to frame2
    """
    # Add batch dimension, move to device, and ensure float32
    frame1 = frame1.unsqueeze(0).to(device).float()  # (1, C, H, W)
    frame2 = frame2.unsqueeze(0).to(device).float()  # (1, C, H, W)
    
    with torch.no_grad():
        # GMFlow expects images in range [0, 1]
        if frame1.max() > 1.0:
            frame1 = frame1 / 255.0
            frame2 = frame2 / 255.0
        
        # Store original size
        ori_size = frame1.shape[-2:]
        
        # Resize to inference size (480, 640) as done in the paper
        inference_size = [480, 640]
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            frame1 = torch.nn.functional.interpolate(
                frame1, size=inference_size, mode='bilinear', align_corners=True
            )
            frame2 = torch.nn.functional.interpolate(
                frame2, size=inference_size, mode='bilinear', align_corners=True
            )
        
        # Get flow prediction with GMFlow/UniMatch
        results_dict = gmflow_model(
            frame1, frame2,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,  # Only need forward flow
        )
        
        flow = results_dict['flow_preds'][-1]  # Take final prediction (B, 2, H, W)
        
        # Resize flow back to original size if needed
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow = torch.nn.functional.interpolate(
                flow, size=ori_size, mode='bilinear', align_corners=True
            )
            # Scale flow values according to the resize ratio
            flow[:, 0, :, :] *= ori_size[1] / inference_size[1]  # Scale u
            flow[:, 1, :, :] *= ori_size[0] / inference_size[0]  # Scale v
    
    return flow.squeeze(0).cpu()  # (2, H, W), move to CPU to save GPU memory


def compute_optical_flow_raft(frame1, frame2, raft_model, device):
    """
    Compute optical flow between two frames using RAFT.
    
    Args:
        frame1: Tensor (C, H, W) - first frame
        frame2: Tensor (C, H, W) - second frame
        raft_model: RAFT model
        device: torch device
    
    Returns:
        flow: Tensor (2, H, W) - optical flow from frame1 to frame2
    """
    # Add batch dimension and move to device
    frame1 = frame1.unsqueeze(0).to(device).float()  # (1, C, H, W)
    frame2 = frame2.unsqueeze(0).to(device).float()  # (1, C, H, W)
    
    with torch.no_grad():
        # RAFT expects images in range [0, 255]
        if frame1.max() <= 1.0:
            frame1 = frame1 * 255.0
            frame2 = frame2 * 255.0
        
        # Get flow prediction
        flow_predictions = raft_model(frame1, frame2)
        flow = flow_predictions[-1]  # Take final prediction (B, 2, H, W)
    
    return flow.squeeze(0).cpu()  # (2, H, W)


def compute_flow_sequence(video_frames, flow_model, device, use_gmflow=True):
    """
    Compute optical flow between consecutive frames in a video.
    
    Args:
        video_frames: List of tensors, each (C, H, W), length T
        flow_model: GMFlow or RAFT model
        device: torch device
        use_gmflow: If True, uses GMFlow; if False, uses RAFT
    
    Returns:
        flow_sequence: Tensor (2, T-1, H, W) - optical flow for T-1 transitions
    """
    compute_flow_fn = compute_optical_flow_gmflow if use_gmflow else compute_optical_flow_raft
    
    flows = []
    for i in range(len(video_frames) - 1):
        flow = compute_flow_fn(video_frames[i], video_frames[i+1], flow_model, device)
        # flow shape: (2, H, W)
        flows.append(flow)
    
    # Stack flows: (T-1, 2, H, W) -> (2, T-1, H, W)
    flow_sequence = torch.stack(flows, dim=0).permute(1, 0, 2, 3)
    
    return flow_sequence


def load_gmflow_model(device):
    """Load pretrained GMFlow (UniMatch) model."""
    try:
        # Add parent directory to path to import from unimatch folder
        import sys
        sys.path.insert(0, '/home/wuroderi/projects/def-gigor/wuroderi/classifier_flow')
        from unimatch.unimatch import UniMatch
        
        print("Loading GMFlow (UniMatch) model...")
        
        # Model configuration (from paper)
        model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task='flow'
        ).to(device)
        
        # Load pretrained weights
        checkpoint_path = '/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/experiment/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print("GMFlow model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading GMFlow: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_raft_model(device):
    """Load pretrained RAFT model from torchvision."""
    try:
        from torchvision.models.optical_flow import raft_large
        print("Loading RAFT model...")
        model = raft_large(pretrained=True).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("RAFT model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading RAFT: {e}")
        print("Make sure you have torchvision >= 0.13.0")
        raise


def create_episode_info_list(directory, tasks, cameras, episodes, diffusion_time_steps, results_json,
                             skip_file_check=False):
    """
    Create a list of episode information dictionaries for lazy loading.
    
    Args:
        directory: Base directory containing task data
        tasks: List of task names
        cameras: List of camera views
        episodes: List of episode numbers
        diffusion_time_steps: List of diffusion timesteps
        results_json: Path to results JSON file
        skip_file_check: If True, skip os.path.exists() checks (much faster on
                         network filesystems - 495k stat calls would take hours).
                         Missing files will raise errors in the DataLoader instead.
    
    Returns:
        episode_info_list: List of dicts with paths and labels
    """
    episode_info_list = []
    
    with open(results_json, "r") as f:
        results_dict = json.load(f)
    
    for task in tasks:
        print(f"Indexing data for task: {task}")
        for camera in cameras:
            successes = results_dict[task]['success'][camera]
            for episode in episodes:
                # Path to condition image (same for all timesteps in an episode)
                condition_path = f"{directory}/{task}/{camera}/episode_{episode:03d}/video_1/metaworld_sample.gif"
                
                for t in diffusion_time_steps:
                    latent_path = f"{directory}/{task}/{camera}/episode_{episode:03d}/video_1/activations_{t:03d}.npy"
                    x0_path = f"{directory}/{task}/{camera}/episode_{episode:03d}/video_1/sample_t{t:03d}.gif"
                    
                    if skip_file_check:
                        # Skip filesystem checks - trust the data is there.
                        # Missing files will error in the DataLoader, not silently dropped.
                        episode_info_list.append({
                            'latent_path': latent_path,
                            'x0_path': x0_path,
                            'condition_path': condition_path,
                            'timestep': t,
                            'label': successes[episode],
                            'task': task,
                            'camera': camera,
                            'episode': episode
                        })
                    else:
                        # Check if files exist
                        if os.path.exists(latent_path) and os.path.exists(x0_path) and os.path.exists(condition_path):
                            episode_info_list.append({
                                'latent_path': latent_path,
                                'x0_path': x0_path,
                                'condition_path': condition_path,
                                'timestep': t,
                                'label': successes[episode],
                                'task': task,
                                'camera': camera,
                                'episode': episode
                            })
                        else:
                            if not os.path.exists(latent_path):
                                print(f"Warning: Latent file not found: {latent_path}")
                            if not os.path.exists(x0_path):
                                print(f"Warning: X0 file not found: {x0_path}")
                            if not os.path.exists(condition_path):
                                print(f"Warning: Condition file not found: {condition_path}")
        
        print(f"Indexed task {task}")
    
    print(f"Total samples indexed: {len(episode_info_list)}")
    return episode_info_list


def load_combined_data_lazy(directory, tasks, cameras, episodes, diffusion_time_steps, 
                            results_json, flow_model, device, use_gmflow=True):
    """
    Create episode info list for lazy loading of combined latent + flow data.
    
    Returns:
        episode_info_list: List of dicts for lazy dataset construction
    """
    return create_episode_info_list(
        directory, tasks, cameras, episodes, diffusion_time_steps, results_json
    )


def episode_aware_split(episode_info_list, val_split, num_cameras, num_diffusion_steps):
    """
    Split data by episode to prevent data leakage.
    Each episode generates (num_cameras * num_diffusion_steps) samples.
    This ensures all samples from an episode stay together in train or val.
    
    Args:
        episode_info_list: List of episode info dicts
        val_split: Fraction of episodes for validation
        num_cameras: Number of camera views per episode
        num_diffusion_steps: Number of diffusion timesteps per episode
    
    Returns:
        train_info_list, val_info_list
    """
    samples_per_episode = num_cameras * num_diffusion_steps
    total_samples = len(episode_info_list)
    num_episodes = total_samples // samples_per_episode
    
    if total_samples % samples_per_episode != 0:
        print(f"Warning: Sample count {total_samples} not evenly divisible by samples_per_episode {samples_per_episode}")
        print(f"This may indicate inconsistent data. Proceeding with {num_episodes} complete episodes.")
    
    # Create episode indices and shuffle
    episode_indices = np.arange(num_episodes)
    np.random.shuffle(episode_indices)
    
    # Split episodes
    num_val_episodes = int(num_episodes * val_split)
    num_train_episodes = num_episodes - num_val_episodes
    
    train_episode_indices = set(episode_indices[:num_train_episodes])
    val_episode_indices = set(episode_indices[num_train_episodes:])
    
    # Split samples by episode
    train_info_list = []
    val_info_list = []
    
    for i in range(num_episodes):
        start_idx = i * samples_per_episode
        end_idx = start_idx + samples_per_episode
        episode_samples = episode_info_list[start_idx:end_idx]
        
        if i in train_episode_indices:
            train_info_list.extend(episode_samples)
        else:
            val_info_list.extend(episode_samples)
    
    print(f"Episode-aware split: {num_train_episodes} train episodes ({len(train_info_list)} samples), "
          f"{num_val_episodes} val episodes ({len(val_info_list)} samples)")
    
    return train_info_list, val_info_list


def compute_latent_statistics(episode_info_list, sample_size=100):
    """
    Compute mean and std of latent embeddings for normalization.
    
    Args:
        episode_info_list: List of episode info dicts
        sample_size: Number of samples to use for statistics (use subset to save time)
    
    Returns:
        mean, std: Tensors for normalization
    """
    print(f"Computing latent statistics from {sample_size} samples...")
    
    # Sample random subset
    indices = np.random.choice(len(episode_info_list), 
                               size=min(sample_size, len(episode_info_list)), 
                               replace=False)
    
    latents = []
    for idx in indices:
        latent_path = episode_info_list[idx]['latent_path']
        latent = np.load(latent_path)  # (1, 640, 7, 8, 8)
        if latent.ndim == 5:
            latent = latent[0]  # (640, 7, 8, 8)
        latents.append(torch.from_numpy(latent).float())
    
    # Stack and compute statistics
    latents = torch.stack(latents)  # (N, 640, 7, 8, 8)
    mean = latents.mean()
    std = latents.std()
    
    print(f"Latent statistics: mean={mean:.4f}, std={std:.4f}")
    
    return mean, std
