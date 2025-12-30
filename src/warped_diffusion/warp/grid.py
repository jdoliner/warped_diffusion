"""Control grid management for TPS warping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ControlGrid(nn.Module):
    """Manages control point grids for TPS warping.
    
    Converts energy maps to displacement fields and manages the
    source/destination control point pairs.
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        max_displacement: float = 0.3,
        epsilon_floor: float = 0.1,
        smoothing_sigma: float = 1.0,
    ):
        """Initialize control grid.
        
        Args:
            grid_size: Number of control points per dimension.
            max_displacement: Maximum displacement as fraction of grid spacing.
            epsilon_floor: Minimum energy value to prevent collapse.
            smoothing_sigma: Gaussian smoothing sigma for energy map.
        """
        super().__init__()
        self.grid_size = grid_size
        self.max_displacement = max_displacement
        self.epsilon_floor = epsilon_floor
        self.smoothing_sigma = smoothing_sigma
        
        # Create Gaussian smoothing kernel
        kernel_size = int(4 * smoothing_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.register_buffer(
            'gaussian_kernel',
            self._create_gaussian_kernel(kernel_size, smoothing_sigma)
        )
        
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel for smoothing."""
        x = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def smooth_energy_map(self, energy: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to energy map.
        
        Args:
            energy: Energy map [B, 1, H, W]
            
        Returns:
            Smoothed energy map [B, 1, H, W]
        """
        # Apply same kernel to each batch item
        kernel: torch.Tensor = self.gaussian_kernel.to(energy.device, energy.dtype)  # type: ignore
        padding = kernel.shape[-1] // 2
        
        smoothed = F.conv2d(energy, kernel, padding=padding)
        return smoothed
    
    def create_uniform_points(
        self, 
        batch_size: int, 
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create uniform grid of control points.
        
        Args:
            batch_size: Number of batches
            device: Torch device
            dtype: Data type
            
        Returns:
            Control points [B, grid_size*grid_size, 2] in [-1, 1]
        """
        y = torch.linspace(-1, 1, self.grid_size, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, self.grid_size, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    
    def energy_to_displacement(
        self,
        energy: torch.Tensor,
        warp_intensity: float = 1.0
    ) -> torch.Tensor:
        """Convert energy map to control point displacements.
        
        High energy regions attract control points (denser sampling).
        Low energy regions repel control points (sparser sampling).
        
        Args:
            energy: Energy map [B, 1, H, W] in [0, 1]
            warp_intensity: Lambda scaling factor for warp strength
            
        Returns:
            Displacement vectors [B, grid_size*grid_size, 2]
        """
        batch_size = energy.shape[0]
        device = energy.device
        dtype = energy.dtype
        
        # Apply floor to prevent collapse
        energy = energy.clamp(min=self.epsilon_floor)
        
        # Smooth the energy map
        energy_smooth = self.smooth_energy_map(energy)
        
        # Downsample to control grid resolution
        energy_grid = F.interpolate(
            energy_smooth, 
            size=(self.grid_size, self.grid_size),
            mode='bilinear',
            align_corners=True
        )  # [B, 1, grid_size, grid_size]
        
        # Compute gradient of energy (direction of increasing energy)
        # Use Sobel-like gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               device=device, dtype=dtype).reshape(1, 1, 3, 3) / 8.0
        
        grad_x = F.conv2d(energy_grid, sobel_x, padding=1)
        grad_y = F.conv2d(energy_grid, sobel_y, padding=1)
        
        # Displacement is in direction of gradient (toward high energy)
        displacement_x = grad_x.reshape(batch_size, -1)  # [B, grid_size^2]
        displacement_y = grad_y.reshape(batch_size, -1)  # [B, grid_size^2]
        
        # Stack and scale by max displacement and warp intensity
        displacement = torch.stack([displacement_x, displacement_y], dim=-1)
        
        # Normalize by max gradient magnitude to bound displacement
        max_grad = displacement.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        max_grad = max_grad.clamp(min=1e-6)
        displacement = displacement / max_grad * self.max_displacement * warp_intensity
        
        return displacement
    
    def get_control_points(
        self,
        energy: torch.Tensor,
        warp_intensity: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get source and destination control points from energy map.
        
        Args:
            energy: Energy map [B, 1, H, W] in [0, 1]
            warp_intensity: Lambda scaling factor for warp strength
            
        Returns:
            Tuple of (src_points, dst_points), each [B, N, 2]
        """
        batch_size = energy.shape[0]
        device = energy.device
        dtype = energy.dtype
        
        # Get uniform source points
        src_points = self.create_uniform_points(batch_size, device, dtype)
        
        if warp_intensity == 0.0:
            # No warping - return identical points
            return src_points, src_points.clone()
        
        # Compute displacements from energy
        displacement = self.energy_to_displacement(energy, warp_intensity)
        
        # Destination points = source + displacement
        dst_points = src_points + displacement
        
        # Clamp to valid range [-1, 1]
        dst_points = dst_points.clamp(-1.0, 1.0)
        
        return src_points, dst_points
    
    def compute_total_variation(
        self,
        src_points: torch.Tensor,
        dst_points: torch.Tensor
    ) -> torch.Tensor:
        """Compute total variation loss on displacement field.
        
        Encourages smooth warps and prevents self-intersection.
        
        Args:
            src_points: Source control points [B, N, 2]
            dst_points: Destination control points [B, N, 2]
            
        Returns:
            TV loss scalar
        """
        displacement = dst_points - src_points  # [B, N, 2]
        
        # Reshape to grid
        batch_size = displacement.shape[0]
        disp_grid = displacement.reshape(batch_size, self.grid_size, self.grid_size, 2)
        
        # Compute differences along x and y axes
        diff_x = disp_grid[:, :, 1:, :] - disp_grid[:, :, :-1, :]  # [B, H, W-1, 2]
        diff_y = disp_grid[:, 1:, :, :] - disp_grid[:, :-1, :, :]  # [B, H-1, W, 2]
        
        # L2 TV loss
        tv_x = (diff_x ** 2).sum(dim=-1).mean()
        tv_y = (diff_y ** 2).sum(dim=-1).mean()
        
        return tv_x + tv_y


def generate_random_warp(
    batch_size: int,
    grid_size: int,
    max_displacement: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random smooth warp for training augmentation.
    
    Args:
        batch_size: Number of samples
        grid_size: Control grid size
        max_displacement: Maximum displacement magnitude
        device: Torch device
        dtype: Data type
        
    Returns:
        Tuple of (src_points, dst_points)
    """
    # Create uniform source points
    y = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, grid_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    src_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    src_points = src_points.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    
    # Generate random displacements
    displacement = torch.randn(batch_size, grid_size * grid_size, 2, device=device, dtype=dtype)
    
    # Smooth the displacements by averaging with neighbors
    disp_grid = displacement.reshape(batch_size, grid_size, grid_size, 2)
    kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9.0
    
    for dim in range(2):
        disp_channel = disp_grid[..., dim].unsqueeze(1)  # [B, 1, H, W]
        disp_smooth = F.conv2d(disp_channel, kernel, padding=1)
        disp_grid[..., dim] = disp_smooth.squeeze(1)
    
    displacement = disp_grid.reshape(batch_size, -1, 2)
    
    # Normalize and scale
    max_mag = displacement.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    max_mag = max_mag.clamp(min=1e-6)
    displacement = displacement / max_mag * max_displacement
    
    dst_points = (src_points + displacement).clamp(-1.0, 1.0)
    
    return src_points, dst_points
