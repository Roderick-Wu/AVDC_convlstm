import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import data
import model


# =========================
# Global Configuration
# =========================

DIRECTORY = "/home/wuroderi/scratch/activations_metaworld/avdc_policy_raw"
RESULTS_JSON = (
    "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/"
    "results_AVDC_mw/result_dict_complete.json"
)
CHECKPOINT = "checkpoints_combined_convlstm_alltasks/model_best.pt"
MODEL_TYPE = "convlstm"  # "simple" or "convlstm"

TASKS = [
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
    "shelf-place-v2-goal-observable",
]
CAMERAS = ["corner", "corner2", "corner3"]
DIFFUSION_STEPS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

# Test split requested: episodes 500-999 inclusive
EPISODE_START = 500
EPISODE_END = 999

USE_CACHED_FLOWS = True
FLOW_CACHE_DIR = None  # Defaults to DIRECTORY/flow_maps when None
USE_GMFLOW = True  # Only used when USE_CACHED_FLOWS=False

BATCH_SIZE = 64
NUM_WORKERS = 4
NORMALIZE_LATENT = False
SKIP_FILE_CHECK = True

OUTPUT_DIR = "embedding_plots"
SEED = 42

# UMAP controls
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1

# t-SNE controls
TSNE_MAX_POINTS = 30000
TSNE_PERPLEXITY = 30.0
TSNE_ITERATIONS = 1000


def make_model(model_type: str) -> torch.nn.Module:
    if model_type == "simple":
        return model.CombinedClassifier(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            fusion_dim=256,
            dropout=0.3,
        )
    if model_type == "convlstm":
        return model.CombinedConvLSTM(
            latent_channels=640,
            flow_channels=2,
            hidden_dim=128,
            lstm_hidden=256,
            fusion_dim=256,
            dropout=0.3,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def safe_name(task: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", task)


def stratified_sample_indices(tasks: np.ndarray, labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or len(tasks) <= max_points:
        return np.arange(len(tasks))

    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, int], np.ndarray] = {}
    unique_tasks = np.unique(tasks)
    unique_labels = np.unique(labels)

    for task in unique_tasks:
        for label in unique_labels:
            idx = np.where((tasks == task) & (labels == label))[0]
            if len(idx) > 0:
                groups[(task, int(label))] = idx

    total = len(tasks)
    sampled_parts: List[np.ndarray] = []

    for _, idx in groups.items():
        k = max(1, int(round(max_points * (len(idx) / total))))
        k = min(k, len(idx))
        sampled_parts.append(rng.choice(idx, size=k, replace=False))

    sampled = np.unique(np.concatenate(sampled_parts))

    if len(sampled) > max_points:
        sampled = rng.choice(sampled, size=max_points, replace=False)
    elif len(sampled) < max_points:
        remaining = np.setdiff1d(np.arange(total), sampled)
        add_k = min(max_points - len(sampled), len(remaining))
        if add_k > 0:
            extra = rng.choice(remaining, size=add_k, replace=False)
            sampled = np.concatenate([sampled, extra])

    return np.sort(sampled)


def build_dataset(
    episode_info_list: List[dict],
    device: torch.device,
    use_cached_flows: bool,
    flow_cache_dir: str,
    normalize_latent: bool,
    latent_mean,
    latent_std,
    use_gmflow: bool,
):
    if use_cached_flows:
        return data.CachedFlowDataset(
            episode_info_list=episode_info_list,
            flow_cache_dir=flow_cache_dir,
            normalize_latent=normalize_latent,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )

    flow_model = data.load_gmflow_model(device) if use_gmflow else data.load_raft_model(device)
    return data.CombinedLazyDataset(
        episode_info_list=episode_info_list,
        flow_model=flow_model,
        device=device,
        use_gmflow=use_gmflow,
        normalize_latent=normalize_latent,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )


def extract_embeddings(
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (batch_latent, batch_flow, batch_t, batch_y) in enumerate(loader):
            batch_latent = batch_latent.to(device)
            batch_flow = batch_flow.to(device)
            batch_t = batch_t.to(device)

            _, embeddings = classifier(batch_latent, batch_flow, batch_t, return_embedding=True)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch_y.squeeze(1).cpu().numpy().astype(np.int32))

            if (batch_idx + 1) % 50 == 0:
                seen = (batch_idx + 1) * loader.batch_size
                print(f"  Processed {min(seen, len(loader.dataset))}/{len(loader.dataset)} samples")

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def plot_by_label(points: np.ndarray, labels: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 8))

    failure_mask = labels == 0
    success_mask = labels == 1

    plt.scatter(points[failure_mask, 0], points[failure_mask, 1], s=5, alpha=0.5, c="#d62728", label="Failure")
    plt.scatter(points[success_mask, 0], points[success_mask, 1], s=5, alpha=0.5, c="#2ca02c", label="Success")

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_per_task(points: np.ndarray, labels: np.ndarray, tasks: np.ndarray, title_prefix: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for task in np.unique(tasks):
        mask = tasks == task
        task_points = points[mask]
        task_labels = labels[mask]

        out_file = os.path.join(out_dir, f"{safe_name(task)}.png")
        plot_by_label(
            task_points,
            task_labels,
            title=f"{title_prefix}: {task}",
            out_path=out_file,
        )


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    episodes = list(range(EPISODE_START, EPISODE_END + 1))

    print("Creating test episode index...")
    episode_info_list = data.create_episode_info_list(
        directory=DIRECTORY,
        tasks=TASKS,
        cameras=CAMERAS,
        episodes=episodes,
        diffusion_time_steps=DIFFUSION_STEPS,
        results_json=RESULTS_JSON,
        skip_file_check=SKIP_FILE_CHECK,
    )

    if len(episode_info_list) == 0:
        raise RuntimeError("No test samples were indexed. Check paths and arguments.")

    task_array = np.array([item["task"] for item in episode_info_list])

    latent_mean, latent_std = None, None
    if NORMALIZE_LATENT:
        print("Computing latent normalization stats from indexed set...")
        latent_mean, latent_std = data.compute_latent_statistics(episode_info_list, sample_size=500)

    flow_cache_dir = FLOW_CACHE_DIR or os.path.join(DIRECTORY, "flow_maps")
    if USE_CACHED_FLOWS and not os.path.exists(flow_cache_dir):
        raise FileNotFoundError(
            f"Flow cache directory not found: {flow_cache_dir}. "
            "Set FLOW_CACHE_DIR or set USE_CACHED_FLOWS=False."
        )

    print("Building dataset/dataloader...")
    dataset = build_dataset(
        episode_info_list=episode_info_list,
        device=device,
        use_cached_flows=USE_CACHED_FLOWS,
        flow_cache_dir=flow_cache_dir,
        normalize_latent=NORMALIZE_LATENT,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_gmflow=USE_GMFLOW,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loading model checkpoint: {CHECKPOINT}")
    classifier = make_model(MODEL_TYPE)
    classifier.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    classifier = classifier.to(device)

    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(classifier, loader, device)

    if len(embeddings) != len(task_array):
        raise RuntimeError(
            f"Mismatch between embeddings ({len(embeddings)}) and metadata ({len(task_array)})."
        )

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "embeddings_test.npz"),
        embeddings=embeddings,
        labels=labels,
        tasks=task_array,
    )
    print(f"Saved raw embeddings to {os.path.join(OUTPUT_DIR, 'embeddings_test.npz')}")

    print("Standardizing embeddings for visualization...")
    scaled_embeddings = StandardScaler().fit_transform(embeddings)

    print("Running UMAP...")
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Install with: pip install umap-learn"
        ) from exc

    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=SEED,
    )
    umap_points = umap_reducer.fit_transform(scaled_embeddings)

    plot_by_label(
        umap_points,
        labels,
        title="UMAP: All Tasks (colored by success/failure)",
        out_path=os.path.join(OUTPUT_DIR, "umap_all_by_label.png"),
    )
    plot_per_task(
        umap_points,
        labels,
        task_array,
        title_prefix="UMAP (colored by success/failure)",
        out_dir=os.path.join(OUTPUT_DIR, "umap_by_task"),
    )

    tsne_indices = stratified_sample_indices(task_array, labels, TSNE_MAX_POINTS, SEED)
    tsne_embeddings = scaled_embeddings[tsne_indices]
    tsne_labels = labels[tsne_indices]
    tsne_tasks = task_array[tsne_indices]

    if len(tsne_indices) < len(scaled_embeddings):
        print(
            "Running t-SNE on a stratified subset "
            f"({len(tsne_indices)}/{len(scaled_embeddings)} points)."
        )
    else:
        print(f"Running t-SNE on all {len(tsne_indices)} points.")

    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, max(5.0, (len(tsne_embeddings) - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        n_iter=TSNE_ITERATIONS,
        random_state=SEED,
        verbose=1,
    )
    tsne_points = tsne.fit_transform(tsne_embeddings)

    plot_by_label(
        tsne_points,
        tsne_labels,
        title="t-SNE: Test Embeddings (colored by success/failure)",
        out_path=os.path.join(OUTPUT_DIR, "tsne_all_by_label.png"),
    )
    plot_per_task(
        tsne_points,
        tsne_labels,
        tsne_tasks,
        title_prefix="t-SNE (colored by success/failure)",
        out_dir=os.path.join(OUTPUT_DIR, "tsne_by_task"),
    )

    print("Done. Generated plots:")
    print(f"  {os.path.join(OUTPUT_DIR, 'umap_all_by_label.png')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'tsne_all_by_label.png')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'umap_by_task')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'tsne_by_task')}")


if __name__ == "__main__":
    main()
