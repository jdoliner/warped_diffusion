"""Semantic Warp Layer - the core module for warped diffusion.

Integrates TPS warping, energy map computation, and the lambda schedule.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..warp.tps import ThinPlateSplineWarp
from ..warp.grid import ControlGrid
from ..warp.energy import EnergyMapComputer, CLIPSaliencyComputer


class WarpSchedule:
    """Manages the warp intensity schedule over denoising timesteps.
    
    - t > t_structural: lambda = 1.0 (maximum warp)
    - t_refinement < t <= t_structural: lambda decays linearly
    - t <= t_refinement: lambda = 0.0 (no warp)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        t_structural: int = 700,
        t_refinement: int = 300,
    ):
        """Initialize warp schedule.
        
        Args:
            num_timesteps: Total number of denoising timesteps.
            t_structural: Timestep below which warp starts decaying.
            t_refinement: Timestep below which warp is zero.
        """
        self.num_timesteps = num_timesteps
        self.t_structural = t_structural
        self.t_refinement = t_refinement
    
    def get_lambda(self, t: int) -> float:
        """Get warp intensity for timestep t.
        
        Args:
            t: Current timestep (0 = clean, num_timesteps = pure noise)
            
        Returns:
            Warp intensity lambda in [0, 1]
        """
        if t > self.t_structural:
            return 1.0
        elif t <= self.t_refinement:
            return 0.0
        else:
            # Linear decay between t_structural and t_refinement
            progress = (t - self.t_refinement) / (self.t_structural - self.t_refinement)
            return float(progress)
    
    def __call__(self, t: int) -> float:
        return self.get_lambda(t)


class SemanticWarpLayer(nn.Module):
    """Complete semantic warping module.
    
    Performs:
    1. Energy map computation from latent + attention + CLIP
    2. Conversion of energy to control point displacements
    3. Forward TPS warp of input
    4. Inverse TPS warp of output (for noise prediction)
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        max_displacement: float = 0.3,
        epsilon_floor: float = 0.1,
        energy_alpha: float = 0.5,
        energy_beta: float = 0.4,
        energy_gamma: float = 0.1,
        t_structural: int = 700,
        t_refinement: int = 300,
        num_timesteps: int = 1000,
        latent_clip_projector: Optional[nn.Module] = None,
    ):
        """Initialize SemanticWarpLayer.
        
        Args:
            grid_size: Size of control point grid.
            max_displacement: Maximum control point displacement.
            epsilon_floor: Minimum energy value to prevent collapse.
            energy_alpha: Weight for CLIP saliency.
            energy_beta: Weight for attention maps.
            energy_gamma: Weight for Sobel edges.
            t_structural: Timestep where warp starts decaying.
            t_refinement: Timestep where warp becomes zero.
            num_timesteps: Total denoising timesteps.
            latent_clip_projector: Optional projector for CLIP saliency.
        """
        super().__init__()
        
        self.tps = ThinPlateSplineWarp(grid_size=grid_size)
        self.control_grid = ControlGrid(
            grid_size=grid_size,
            max_displacement=max_displacement,
            epsilon_floor=epsilon_floor,
        )
        self.energy_computer = EnergyMapComputer(
            alpha=energy_alpha,
            beta=energy_beta,
            gamma=energy_gamma,
        )
        self.schedule = WarpSchedule(
            num_timesteps=num_timesteps,
            t_structural=t_structural,
            t_refinement=t_refinement,
        )
        
        # Optional CLIP saliency
        if latent_clip_projector is not None:
            self.clip_saliency = CLIPSaliencyComputer(latent_clip_projector)
        else:
            self.clip_saliency = None
        
        # Cache for control points (to apply inverse warp to noise)
        self._cached_src_points: Optional[torch.Tensor] = None
        self._cached_dst_points: Optional[torch.Tensor] = None
    
    def compute_energy_map(
        self,
        latent: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        noise_pred: Optional[torch.Tensor] = None,
        alpha_cumprod_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute energy map from all available sources.
        
        Args:
            latent: Current noisy latent [B, C, H, W]
            attention_maps: Cross-attention maps [B, (heads), H, W]
            text_embedding: Text CLIP embedding for saliency [B, D]
            noise_pred: Noise prediction for z_0 estimation [B, C, H, W]
            alpha_cumprod_t: Alpha cumulative product at timestep t
            
        Returns:
            Energy map [B, 1, H, W] in [0, 1]
        """
        clip_saliency = None
        
        # Compute CLIP saliency if all requirements are met
        if (self.clip_saliency is not None and 
            text_embedding is not None and 
            noise_pred is not None and 
            alpha_cumprod_t is not None):
            
            # Estimate clean latent
            z_0_est = self.clip_saliency.compute_estimated_z0(
                latent, noise_pred, alpha_cumprod_t
            )
            clip_saliency = self.clip_saliency.compute_saliency(z_0_est, text_embedding)
        
        # Compute combined energy
        energy = self.energy_computer(
            latent=latent,
            clip_saliency=clip_saliency,
            attention_maps=attention_maps,
        )
        
        return energy
    
    def forward_warp(
        self,
        latent: torch.Tensor,
        timestep: int,
        attention_maps: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        noise_pred: Optional[torch.Tensor] = None,
        alpha_cumprod_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply forward warp to latent before U-Net.
        
        Caches control points for later inverse warp.
        
        Args:
            latent: Input latent [B, C, H, W]
            timestep: Current denoising timestep
            attention_maps: Cross-attention maps from previous step
            text_embedding: Text CLIP embedding
            noise_pred: Previous noise prediction for z_0 estimation
            alpha_cumprod_t: Alpha cumulative product at timestep
            
        Returns:
            Warped latent [B, C, H, W]
        """
        warp_intensity = self.schedule(timestep)
        
        # No warp needed
        if warp_intensity == 0.0:
            self._cached_src_points = None
            self._cached_dst_points = None
            return latent
        
        # Compute energy map
        energy = self.compute_energy_map(
            latent=latent,
            attention_maps=attention_maps,
            text_embedding=text_embedding,
            noise_pred=noise_pred,
            alpha_cumprod_t=alpha_cumprod_t,
        )
        
        # Get control points from energy
        src_points, dst_points = self.control_grid.get_control_points(
            energy, warp_intensity
        )
        
        # Cache for inverse warp
        self._cached_src_points = src_points
        self._cached_dst_points = dst_points
        
        # Apply forward warp
        warped = self.tps.warp(latent, src_points, dst_points)
        
        return warped
    
    def inverse_warp(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse warp to tensor (e.g., noise prediction).
        
        Uses cached control points from forward_warp.
        
        Args:
            tensor: Tensor to inverse-warp [B, C, H, W]
            
        Returns:
            Inverse-warped tensor [B, C, H, W]
        """
        if self._cached_src_points is None or self._cached_dst_points is None:
            # No warp was applied
            return tensor
        
        # Inverse warp by swapping control points
        inverse_warped = self.tps.warp_inverse(
            tensor, 
            self._cached_src_points, 
            self._cached_dst_points
        )
        
        return inverse_warped
    
    def get_tv_loss(self) -> torch.Tensor:
        """Get total variation loss on current displacement field.
        
        Returns:
            TV loss scalar (0 if no warp applied)
        """
        if self._cached_src_points is None or self._cached_dst_points is None:
            return torch.tensor(0.0)
        
        return self.control_grid.compute_total_variation(
            self._cached_src_points,
            self._cached_dst_points
        )
    
    def forward(
        self,
        latent: torch.Tensor,
        timestep: int,
        attention_maps: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        noise_pred: Optional[torch.Tensor] = None,
        alpha_cumprod_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - applies forward warp.
        
        Equivalent to forward_warp. Use inverse_warp separately on output.
        """
        return self.forward_warp(
            latent=latent,
            timestep=timestep,
            attention_maps=attention_maps,
            text_embedding=text_embedding,
            noise_pred=noise_pred,
            alpha_cumprod_t=alpha_cumprod_t,
        )
