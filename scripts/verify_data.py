"""Quick verification that the processed dataset loads correctly."""

import torch
from src.data.dataset import CameraTrajectoryDataset, collate_fn
from torch.utils.data import DataLoader

ds = CameraTrajectoryDataset(data_root='data', split='train', num_frames=48, toric_dim=6)
print(f"Dataset size: {len(ds)}")

sample = ds[0]
print(f"Sample 0:")
print(f"  y shape: {sample['y'].shape}")
print(f"  text: {sample['text'][:80]}...")
print(f"  shot_type: {sample['shot_type']}")
print(f"  motion_type: {sample['motion_type']}")

dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch = next(iter(dl))
print(f"\nBatch:")
print(f"  y shape: {batch['y'].shape}")
print(f"  shot_types: {batch['shot_types']}")
print(f"  motion_types: {batch['motion_types']}")
print(f"  texts[0]: {batch['texts'][0][:80]}...")

# Test model forward pass
from src.models.denoiser import CameraTrajectoryDenoiser
from src.models.diffusion import GaussianDiffusion

denoiser = CameraTrajectoryDenoiser(
    toric_dim=6, num_frames=48, hidden_dim=128, num_layers=2,
    num_heads=4, text_dim=512, timestep_dim=128,
    num_shot_types=5, shot_type_dim=64,
    num_motion_types=9, motion_type_dim=64,
)
diffusion = GaussianDiffusion(denoiser, num_timesteps=100, beta_schedule='cosine')

y = batch['y']
text_embed = torch.randn(4, 512)
shot_type = batch['shot_types']
motion_type = batch['motion_types']

loss = diffusion.p_losses(y, text_embed, shot_type=shot_type, motion_type=motion_type)
print(f"\nForward pass OK! Loss = {loss.item():.4f}")
print("\nEverything is ready for training!")
