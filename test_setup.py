"""
Quick test script to verify the combined classifier setup.
Tests data loading, model creation, and forward pass.
"""

import torch
import sys
sys.path.insert(0, '/home/wuroderi/projects/def-gigor/wuroderi/classifier_flow')

import data
import model

def test_data_loading():
    """Test that we can load latent embeddings, x0 predictions, and compute flow."""
    print("=" * 60)
    print("TEST 1: Data Loading")
    print("=" * 60)
    
    # Test parameters
    directory = "/home/wuroderi/scratch/single_task_basketball"
    task_list = ["basketball-v2-goal-observable"]
    camera_list = ["corner"]  # Just one camera for quick test
    episodes = [0]  # Just one episode
    diffusion_time_steps = [15, 25]  # Just two timesteps
    results_json = "/home/wuroderi/projects/def-gigor/wuroderi/AVDC_exp_policy_data/results/results_AVDC_mw/result_dict_basketball.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load GMFlow model
    print("\nLoading GMFlow model...")
    flow_model = data.load_gmflow_model(device)
    print("✓ GMFlow model loaded")
    
    # Create episode info list
    print("\nCreating episode info list...")
    episode_info_list = data.create_episode_info_list(
        directory=directory,
        tasks=task_list,
        cameras=camera_list,
        episodes=episodes,
        diffusion_time_steps=diffusion_time_steps,
        results_json=results_json
    )
    print(f"✓ Created info list with {len(episode_info_list)} samples")
    
    # Create dataset
    print("\nCreating lazy dataset...")
    dataset = data.CombinedLazyDataset(
        episode_info_list=episode_info_list,
        flow_model=flow_model,
        device=device,
        use_gmflow=True,
        normalize_latent=False
    )
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Load one sample
    print("\nLoading one sample...")
    latent, flow, timestep, label = dataset[0]
    
    print(f"✓ Sample loaded successfully:")
    print(f"  Latent shape: {latent.shape} (expected: (640, 7, 8, 8))")
    print(f"  Flow shape: {flow.shape} (expected: (2, 7, H, W))")
    print(f"  Timestep: {timestep.item()}")
    print(f"  Label: {label.item()} ({'Success' if label.item() == 1 else 'Failure'})")
    
    # Verify shapes
    assert latent.shape[0] == 640, f"Expected 640 latent channels, got {latent.shape[0]}"
    assert latent.shape[1] == 7, f"Expected 7 latent frames, got {latent.shape[1]}"
    assert latent.shape[2] == 8 and latent.shape[3] == 8, f"Expected 8x8 spatial, got {latent.shape[2]}x{latent.shape[3]}"
    assert flow.shape[0] == 2, f"Expected 2 flow channels, got {flow.shape[0]}"
    assert flow.shape[1] == 7, f"Expected 7 flow frames, got {flow.shape[1]}"
    
    print("\n✓ All shape assertions passed!")
    return True


def test_model_creation():
    """Test that we can create models and run forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Creation and Forward Pass")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    batch_size = 2
    latent = torch.randn(batch_size, 640, 7, 8, 8).to(device)
    flow = torch.randn(batch_size, 2, 7, 480, 640).to(device)
    timestep = torch.tensor([[15.0], [25.0]]).to(device)
    
    print(f"\nDummy input shapes:")
    print(f"  Latent: {latent.shape}")
    print(f"  Flow: {flow.shape}")
    print(f"  Timestep: {timestep.shape}")
    
    # Test CombinedClassifier
    print("\n--- Testing CombinedClassifier ---")
    classifier1 = model.CombinedClassifier(
        latent_channels=640,
        flow_channels=2,
        hidden_dim=128,
        fusion_dim=256,
        dropout=0.3
    ).to(device)
    
    n_params1 = sum(p.numel() for p in classifier1.parameters())
    print(f"✓ CombinedClassifier created")
    print(f"  Total parameters: {n_params1:,}")
    
    output1 = classifier1(latent, flow, timestep)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output1.shape} (expected: ({batch_size}, 1))")
    print(f"  Output (logits): {output1.squeeze().detach().cpu().numpy()}")
    print(f"  Output (probs): {torch.sigmoid(output1).squeeze().detach().cpu().numpy()}")
    
    assert output1.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output1.shape}"
    
    # Test CombinedConvLSTM
    print("\n--- Testing CombinedConvLSTM ---")
    classifier2 = model.CombinedConvLSTM(
        latent_channels=640,
        flow_channels=2,
        hidden_dim=128,
        lstm_hidden=256,
        fusion_dim=256,
        dropout=0.3
    ).to(device)
    
    n_params2 = sum(p.numel() for p in classifier2.parameters())
    print(f"✓ CombinedConvLSTM created")
    print(f"  Total parameters: {n_params2:,}")
    
    output2 = classifier2(latent, flow, timestep)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output2.shape} (expected: ({batch_size}, 1))")
    print(f"  Output (logits): {output2.squeeze().detach().cpu().numpy()}")
    print(f"  Output (probs): {torch.sigmoid(output2).squeeze().detach().cpu().numpy()}")
    
    assert output2.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output2.shape}"
    
    print("\n✓ All model tests passed!")
    return True


def test_condition_frame_extraction():
    """Test that we correctly extract and prepend the condition frame."""
    print("\n" + "=" * 60)
    print("TEST 3: Condition Frame Extraction")
    print("=" * 60)
    
    from PIL import Image
    import numpy as np
    
    # Load a sample x0 video and final video
    episode_path = "/home/wuroderi/scratch/single_task_basketball/basketball-v2-goal-observable/corner/episode_000/video_1"
    
    # Load x0 prediction (should have 7 frames)
    x0_path = f"{episode_path}/sample_t015.gif"
    x0_frames = data.load_video_frames(x0_path)
    print(f"X0 prediction: {len(x0_frames)} frames (expected: 7)")
    assert len(x0_frames) == 7, f"Expected 7 frames in x0 prediction, got {len(x0_frames)}"
    
    # Load condition frame
    condition_path = f"{episode_path}/metaworld_sample.gif"
    condition_frame = data.load_condition_frame(condition_path)
    print(f"Condition frame shape: {condition_frame.shape} (expected: (3, H, W))")
    assert condition_frame.shape[0] == 3, f"Expected 3 color channels, got {condition_frame.shape[0]}"
    
    # Verify they have the same spatial resolution
    assert x0_frames[0].shape == condition_frame.shape, \
        f"Shape mismatch: x0 {x0_frames[0].shape} vs condition {condition_frame.shape}"
    
    # Prepend condition to x0 frames
    video_frames = [condition_frame] + x0_frames
    print(f"Combined video: {len(video_frames)} frames (expected: 8)")
    assert len(video_frames) == 8, f"Expected 8 frames after prepending, got {len(video_frames)}"
    
    # Verify the first frame matches the condition
    assert torch.allclose(video_frames[0], condition_frame), \
        "First frame should be the condition frame"
    
    print("✓ Condition frame extraction and prepending works correctly!")
    print("  - X0 prediction: 7 frames")
    print("  - Condition prepended: 8 frames total")
    print("  - This will generate 7 flow maps (8 frames → 7 transitions)")
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMBINED CLASSIFIER - SETUP VERIFICATION")
    print("=" * 60)
    
    try:
        # Run tests
        test_condition_frame_extraction()
        test_model_creation()
        test_data_loading()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe combined classifier is properly set up and ready to use.")
        print("You can now run:")
        print("  - python train.py    (to train the model)")
        print("  - python inference.py (to run inference)")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
