"""Tests for TPS warping implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest


def test_tps_warp_basic():
    """Test basic TPS warp functionality."""
    from warped_diffusion.warp.tps import ThinPlateSplineWarp, create_uniform_grid
    
    tps = ThinPlateSplineWarp(grid_size=8)
    
    # Create test input
    batch_size = 2
    channels = 4
    height, width = 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    # Create identity warp (src = dst)
    src_points = create_uniform_grid(8, batch_size, x.device)
    dst_points = src_points.clone()
    
    # Apply warp
    warped = tps.warp(x, src_points, dst_points)
    
    # Should be nearly identical to input
    assert warped.shape == x.shape
    assert torch.allclose(warped, x, atol=1e-4)


def test_tps_warp_inverse():
    """Test that inverse warp approximately recovers original."""
    from warped_diffusion.warp.tps import ThinPlateSplineWarp, create_uniform_grid
    
    tps = ThinPlateSplineWarp(grid_size=8)
    
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64)
    
    # Create non-trivial warp
    src_points = create_uniform_grid(8, batch_size, x.device)
    dst_points = src_points + torch.randn_like(src_points) * 0.1
    dst_points = dst_points.clamp(-1, 1)
    
    # Forward warp
    warped = tps.warp(x, src_points, dst_points)
    
    # Inverse warp
    recovered = tps.warp_inverse(warped, src_points, dst_points)
    
    # Should approximately recover original
    assert recovered.shape == x.shape
    mse = ((recovered - x) ** 2).mean()
    assert mse < 0.1, f"Inverse warp MSE too high: {mse}"


def test_tps_warp_differentiable():
    """Test that gradients flow through TPS warp."""
    from warped_diffusion.warp.tps import ThinPlateSplineWarp, create_uniform_grid
    
    tps = ThinPlateSplineWarp(grid_size=8)
    
    batch_size = 2
    x = torch.randn(batch_size, 4, 64, 64, requires_grad=True)
    
    src_points = create_uniform_grid(8, batch_size, x.device)
    dst_points = src_points + torch.randn_like(src_points) * 0.1
    dst_points = dst_points.clamp(-1, 1)
    dst_points.requires_grad = True
    
    # Forward warp
    warped = tps.warp(x, src_points, dst_points)
    
    # Compute loss and backprop
    loss = warped.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert dst_points.grad is not None, "No gradient for control points"
    assert x.grad.shape == x.shape
    assert dst_points.grad.shape == dst_points.shape


def test_control_grid_energy_to_displacement():
    """Test energy map to displacement conversion."""
    from warped_diffusion.warp.grid import ControlGrid
    
    grid = ControlGrid(grid_size=8, max_displacement=0.3)
    
    batch_size = 2
    energy = torch.rand(batch_size, 1, 64, 64)
    
    src_points, dst_points = grid.get_control_points(energy, warp_intensity=1.0)
    
    assert src_points.shape == (batch_size, 64, 2)
    assert dst_points.shape == (batch_size, 64, 2)
    
    # Displacement should be bounded
    displacement = dst_points - src_points
    max_disp = displacement.abs().max()
    assert max_disp <= 0.3 + 1e-5, f"Displacement too large: {max_disp}"


def test_control_grid_zero_warp():
    """Test that zero warp intensity gives identity."""
    from warped_diffusion.warp.grid import ControlGrid
    
    grid = ControlGrid(grid_size=8)
    
    energy = torch.rand(2, 1, 64, 64)
    src_points, dst_points = grid.get_control_points(energy, warp_intensity=0.0)
    
    assert torch.allclose(src_points, dst_points)


def test_energy_map_computation():
    """Test energy map computation."""
    from warped_diffusion.warp.energy import EnergyMapComputer
    
    computer = EnergyMapComputer(alpha=0.5, beta=0.4, gamma=0.1)
    
    batch_size = 2
    latent = torch.randn(batch_size, 4, 64, 64)
    
    # Just with Sobel (no CLIP or attention)
    energy = computer(latent)
    
    assert energy.shape == (batch_size, 1, 64, 64)
    assert energy.min() >= 0.0
    assert energy.max() <= 1.0


def test_random_warp_generation():
    """Test random warp generation for training."""
    from warped_diffusion.warp.grid import generate_random_warp
    
    src, dst = generate_random_warp(
        batch_size=4,
        grid_size=8,
        max_displacement=0.3,
        device=torch.device('cpu'),
    )
    
    assert src.shape == (4, 64, 2)
    assert dst.shape == (4, 64, 2)
    
    # All points should be in valid range
    assert src.min() >= -1.0 and src.max() <= 1.0
    assert dst.min() >= -1.0 and dst.max() <= 1.0


def test_tv_loss():
    """Test total variation loss computation."""
    from warped_diffusion.warp.grid import ControlGrid, generate_random_warp
    
    grid = ControlGrid(grid_size=8)
    
    src, dst = generate_random_warp(
        batch_size=2,
        grid_size=8,
        max_displacement=0.3,
        device=torch.device('cpu'),
    )
    
    tv_loss = grid.compute_total_variation(src, dst)
    
    assert tv_loss.shape == ()
    assert tv_loss >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
