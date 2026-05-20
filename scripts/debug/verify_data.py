"""Quick verification that the processed joint dataset loads correctly."""

import torch
from src.data.dataset import JointTrajectoryDataset, collate_fn
from torch.utils.data import DataLoader

data_root = '/transfer/stc-data'

ds = JointTrajectoryDataset(data_root=data_root, split='train', num_frames=48)
print(f"Dataset size: {len(ds)}")

if len(ds) == 0:
    print("No data found. Run preprocess_et_data.py first.")
    exit(0)

sample = ds[0]
print(f"\nSample 0:")
print(f"  y shape: {sample['y'].shape} (person T*3 + camera T*6 = {48*3 + 48*6})")
print(f"  text: {sample['text'][:80]}...")
print(f"  shot_type: {sample['shot_type']}")
print(f"  motion_type: {sample['motion_type']}")

# Verify split
person_part = sample['y'][:48*3].reshape(48, 3)
camera_part = sample['y'][48*3:].reshape(48, 6)
print(f"  person traj range: [{person_part.min():.3f}, {person_part.max():.3f}]")
print(f"  camera traj range: [{camera_part.min():.3f}, {camera_part.max():.3f}]")

dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch = next(iter(dl))
print(f"\nBatch:")
print(f"  y shape: {batch['y'].shape}")
print(f"  shot_types: {batch['shot_types']}")
print(f"  motion_types: {batch['motion_types']}")

# Test model forward pass
from src.models.denoiser import JointTrajectoryDenoiser
from src.models.diffusion import GaussianDiffusion

denoiser = JointTrajectoryDenoiser(
    person_dim=5, camera_dim=6, num_frames=48,
    hidden_dim=128, num_layers=2, num_heads=4,
    text_dim=512, timestep_dim=128,
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
print("Ready for training!")
