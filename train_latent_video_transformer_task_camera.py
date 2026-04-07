import argparse
import os

import torch

import model
from train import train_latent_video_transformer_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small latent+raw-video transformer for a single task and camera."
    )
    parser.add_argument("--task", required=True, help="Metaworld task name")
    parser.add_argument("--camera", required=True, help="Camera name")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda or cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--video-size", type=int, default=96)
    parser.add_argument("--num-load-workers", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--model-save-root",
        default="/home/wuroderi/scratch/checkpoints_latent_video_transformer_task_camera",
    )
    parser.add_argument("--results-json", default="/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_complete.json")
    parser.add_argument("--directory", default="/home/wuroderi/scratch/activations_metaworld/avdc_policy_raw")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--skip-file-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip filesystem existence checks while indexing the dataset.",
    )
    parser.add_argument("--normalize-latent", action="store_true", default=False)
    parser.add_argument("--use-contrastive-loss", action="store_true", default=False)
    parser.add_argument("--contrastive-weight", type=float, default=0.03)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-warmup-epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    task_list = [args.task]
    camera_list = [args.camera]
    episodes = list(range(args.episodes))
    diffusion_time_steps = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

    classifier = model.LatentVideoTransformer(
        latent_channels=640,
        video_channels=3,
        d_model=64,
        num_heads=4,
        latent_patch_size=2,
        video_patch_size=16,
        latent_frames=7,
        video_frames=8,
        video_size=args.video_size,
        latent_encoder_layers=1,
        video_encoder_layers=1,
        fusion_encoder_layers=2,
        ff_dim=256,
        dropout=0.1,
    )

    safe_task = args.task.replace("/", "-")
    safe_camera = args.camera.replace("/", "-")
    model_save_path = os.path.join(args.model_save_root, f"{safe_task}__{safe_camera}")

    print(f"Created model for task={args.task}, camera={args.camera}")
    print(f"Total parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"Saving to: {model_save_path}")

    train_latent_video_transformer_model(
        classifier_model=classifier,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        val_split=args.val_split,
        directory=args.directory,
        task_list=task_list,
        camera_list=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=args.results_json,
        skip_file_check=args.skip_file_check,
        normalize_latent=args.normalize_latent,
        video_size=args.video_size,
        load_in_memory=True,
        num_load_workers=args.num_load_workers,
        num_workers=args.num_workers,
        use_contrastive_loss=args.use_contrastive_loss,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_warmup_epochs=args.contrastive_warmup_epochs,
        plot_file="loss_curve.png",
        model_save_path=model_save_path,
    )


if __name__ == "__main__":
    main()
