"""Thin Plate Spline (TPS) warping implementation.

TPS provides smooth, differentiable warping with closed-form inverse
via control point swapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ThinPlateSplineWarp(nn.Module):
    """Differentiable Thin Plate Spline warping.
    
    Given source and destination control points, computes a smooth
    warp that can be applied to any tensor using grid_sample.
    
    The inverse warp is computed by simply swapping source and destination
    control points - this is an approximation but is stable and fast.
    """
    
    def __init__(self, grid_size: int = 8, regularization: float = 0.0):
        """Initialize TPS warp.
        
        Args:
            grid_size: Size of the control point grid (grid_size x grid_size).
            regularization: Regularization term for TPS solve (prevents overfitting).
        """
        super().__init__()
        self.grid_size = grid_size
        self.regularization = regularization
        
    def _compute_tps_kernel(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Compute the TPS radial basis function kernel.
        
        K(r) = r^2 * log(r) for TPS in 2D.
        
        Args:
            src: Source points [B, N, 2]
            dst: Destination points [B, N, 2]
            
        Returns:
            Kernel matrix [B, N, N]
        """
        # Compute pairwise distances
        diff = src.unsqueeze(2) - dst.unsqueeze(1)  # [B, N, N, 2]
        dist_sq = (diff ** 2).sum(-1)  # [B, N, N]
        
        # TPS kernel: r^2 * log(r), with special handling for r=0
        # Use r^2 * log(r^2) / 2 = r^2 * log(r) for numerical stability
        dist_sq = dist_sq.clamp(min=1e-8)
        kernel = 0.5 * dist_sq * torch.log(dist_sq)
        
        return kernel
    
    def _solve_tps_coefficients(
        self, 
        src_points: torch.Tensor, 
        dst_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve for TPS transformation coefficients.
        
        The TPS transformation is: f(p) = A @ [1, p_x, p_y]^T + W @ K(p, src_points)
        
        Args:
            src_points: Source control points [B, N, 2] in [-1, 1]
            dst_points: Destination control points [B, N, 2] in [-1, 1]
            
        Returns:
            Tuple of (W, A) where:
                W: Kernel weights [B, N, 2]
                A: Affine coefficients [B, 3, 2]
        """
        batch_size, n_points, _ = src_points.shape
        device = src_points.device
        dtype = src_points.dtype
        
        # Compute kernel matrix K [B, N, N]
        K = self._compute_tps_kernel(src_points, src_points)
        
        # Add regularization to diagonal
        if self.regularization > 0:
            K = K + self.regularization * torch.eye(n_points, device=device, dtype=dtype)
        
        # Build P matrix: [1, x, y] for each point [B, N, 3]
        ones = torch.ones(batch_size, n_points, 1, device=device, dtype=dtype)
        P = torch.cat([ones, src_points], dim=-1)
        
        # Build the full system matrix L:
        # | K   P | | W |   | dst |
        # | P^T 0 | | A | = |  0  |
        zeros = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        
        # Top row: [K, P]
        top = torch.cat([K, P], dim=-1)  # [B, N, N+3]
        # Bottom row: [P^T, 0]
        bottom = torch.cat([P.transpose(-1, -2), zeros], dim=-1)  # [B, 3, N+3]
        # Full matrix
        L = torch.cat([top, bottom], dim=-2)  # [B, N+3, N+3]
        
        # Right-hand side: [dst, 0]
        zeros_rhs = torch.zeros(batch_size, 3, 2, device=device, dtype=dtype)
        rhs = torch.cat([dst_points, zeros_rhs], dim=-2)  # [B, N+3, 2]
        
        # Solve the system
        # Add small regularization to L for numerical stability
        L = L + 1e-6 * torch.eye(n_points + 3, device=device, dtype=dtype)
        solution = torch.linalg.solve(L, rhs)  # [B, N+3, 2]
        
        # Extract W and A
        W = solution[:, :n_points, :]  # [B, N, 2]
        A = solution[:, n_points:, :].transpose(-1, -2)  # [B, 2, 3]
        
        return W, A
    
    def compute_warp_grid(
        self,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
        output_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute the sampling grid for warping.
        
        Args:
            src_points: Source control points [B, N, 2] in [-1, 1]
            dst_points: Destination control points [B, N, 2] in [-1, 1]
            output_size: (H, W) of the output grid
            
        Returns:
            Sampling grid [B, H, W, 2] for use with grid_sample
        """
        batch_size = src_points.shape[0]
        H, W = output_size
        device = src_points.device
        dtype = src_points.dtype
        
        # Solve for TPS coefficients
        W, A = self._solve_tps_coefficients(src_points, dst_points)
        
        # Create a regular grid of query points
        y = torch.linspace(-1, 1, int(H), device=device, dtype=dtype)
        x = torch.linspace(-1, 1, int(W), device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        query_points = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        query_points = query_points.reshape(1, -1, 2).expand(batch_size, -1, -1)  # [B, H*W, 2]
        
        # Compute kernel between query points and source control points
        K = self._compute_tps_kernel(query_points, src_points)  # [B, H*W, N]
        
        # Apply TPS transformation: f(p) = A @ [1, p_x, p_y]^T + K @ W
        n_query = int(H) * int(W)
        ones = torch.ones(batch_size, n_query, 1, device=device, dtype=dtype)
        P = torch.cat([ones, query_points], dim=-1)  # [B, H*W, 3]
        
        # Affine part: [B, H*W, 2]
        affine = torch.bmm(P, A.transpose(-1, -2))
        
        # TPS part: [B, H*W, 2]
        tps = torch.bmm(K, W)
        
        # Combined transformation
        warped = affine + tps  # [B, H*W, 2]
        
        # Reshape to grid format
        warp_grid = warped.reshape(batch_size, int(H), int(W), 2)
        
        return warp_grid
    
    def warp(
        self,
        tensor: torch.Tensor,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'reflection'
    ) -> torch.Tensor:
        """Apply TPS warp to a tensor.
        
        Args:
            tensor: Input tensor [B, C, H, W]
            src_points: Source control points [B, N, 2] in [-1, 1]
            dst_points: Destination control points [B, N, 2] in [-1, 1]
            mode: Interpolation mode for grid_sample
            padding_mode: Padding mode for grid_sample
            
        Returns:
            Warped tensor [B, C, H, W]
        """
        _, _, H, W = tensor.shape
        
        # Compute the sampling grid
        grid = self.compute_warp_grid(src_points, dst_points, (H, W))
        
        # Apply the warp using grid_sample
        warped = F.grid_sample(
            tensor, 
            grid, 
            mode=mode, 
            padding_mode=padding_mode,
            align_corners=True
        )
        
        return warped
    
    def warp_inverse(
        self,
        tensor: torch.Tensor,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
        mode: str = 'bilinear',
        padding_mode: str = 'reflection'
    ) -> torch.Tensor:
        """Apply inverse TPS warp by swapping control points.
        
        This is an approximation of the true inverse, but is stable
        and differentiable.
        
        Args:
            tensor: Input tensor [B, C, H, W]
            src_points: Original source control points [B, N, 2]
            dst_points: Original destination control points [B, N, 2]
            mode: Interpolation mode for grid_sample
            padding_mode: Padding mode for grid_sample
            
        Returns:
            Inverse-warped tensor [B, C, H, W]
        """
        # Swap source and destination to get approximate inverse
        return self.warp(tensor, dst_points, src_points, mode, padding_mode)


def create_uniform_grid(grid_size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a uniform grid of control points.
    
    Args:
        grid_size: Number of points per dimension
        batch_size: Batch size
        device: Torch device
        
    Returns:
        Uniform control points [B, grid_size*grid_size, 2] in [-1, 1]
    """
    y = torch.linspace(-1, 1, grid_size, device=device)
    x = torch.linspace(-1, 1, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    return points.unsqueeze(0).expand(batch_size, -1, -1)
