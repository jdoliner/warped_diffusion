"""Energy map computation for semantic warping.

The energy map combines:
- CLIP gradient saliency (semantic importance)
- Cross-attention maps (text-image alignment)
- Sobel edge detection (structural preservation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EnergyMapComputer(nn.Module):
    """Computes the combined energy map for warping.
    
    E = Norm(alpha * CLIP_grad + beta * Attn_cross + gamma * Sobel_L2)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.4,
        gamma: float = 0.1,
        sobel_blur_sigma: float = 1.0,
        attention_temperature: float = 0.5,
    ):
        """Initialize energy map computer.
        
        Args:
            alpha: Weight for CLIP gradient saliency.
            beta: Weight for cross-attention maps.
            gamma: Weight for Sobel edge detection.
            sobel_blur_sigma: Gaussian blur sigma before Sobel.
            attention_temperature: Softmax temperature for attention sharpening.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sobel_blur_sigma = sobel_blur_sigma
        self.attention_temperature = attention_temperature
        
        # Register Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.reshape(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.reshape(1, 1, 3, 3))
        
        # Gaussian blur kernel for pre-Sobel smoothing
        kernel_size = int(4 * sobel_blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.register_buffer(
            'gaussian_kernel',
            self._create_gaussian_kernel(kernel_size, sobel_blur_sigma)
        )
    
    def _create_gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        x = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.outer(gauss)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def compute_sobel_energy(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute edge energy using Sobel filter on L2 norm.
        
        Args:
            latent: Latent tensor [B, C, H, W]
            
        Returns:
            Edge energy [B, 1, H, W] in [0, 1]
        """
        # Compute L2 norm across channels
        l2_norm = torch.norm(latent, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply Gaussian blur before Sobel
        kernel: torch.Tensor = self.gaussian_kernel.to(l2_norm.device, l2_norm.dtype)  # type: ignore
        padding = kernel.shape[-1] // 2
        l2_smooth = F.conv2d(l2_norm, kernel, padding=padding)
        
        # Apply Sobel filters
        sobel_x: torch.Tensor = self.sobel_x.to(l2_smooth.device, l2_smooth.dtype)  # type: ignore
        sobel_y: torch.Tensor = self.sobel_y.to(l2_smooth.device, l2_smooth.dtype)  # type: ignore
        
        grad_x = F.conv2d(l2_smooth, sobel_x, padding=1)
        grad_y = F.conv2d(l2_smooth, sobel_y, padding=1)
        
        # Gradient magnitude
        edge_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        edge_mag = self._normalize(edge_mag)
        
        return edge_mag
    
    def process_attention_maps(
        self, 
        attention_maps: torch.Tensor,
        target_size: tuple[int, int]
    ) -> torch.Tensor:
        """Process and normalize attention maps.
        
        Args:
            attention_maps: Attention weights [B, H, W] or [B, num_heads, H, W]
            target_size: (H, W) to resize to
            
        Returns:
            Normalized attention energy [B, 1, H, W] in [0, 1]
        """
        # Average across heads if present
        if attention_maps.dim() == 4:
            attention_maps = attention_maps.mean(dim=1)  # [B, H, W]
        
        # Add channel dimension
        attn = attention_maps.unsqueeze(1)  # [B, 1, H, W]
        
        # Resize to target
        attn = F.interpolate(
            attn, 
            size=target_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Apply temperature-scaled softmax for sharpening
        B, _, H, W = attn.shape
        attn_flat = attn.reshape(B, -1)  # [B, H*W]
        attn_sharp = F.softmax(attn_flat / self.attention_temperature, dim=-1)
        attn = attn_sharp.reshape(B, 1, H, W)
        
        # Normalize to [0, 1]
        attn = self._normalize(attn)
        
        return attn
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Min-max normalize tensor to [0, 1] per batch."""
        B = tensor.shape[0]
        flat = tensor.reshape(B, -1)
        min_val = flat.min(dim=1, keepdim=True)[0]
        max_val = flat.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        range_val = (max_val - min_val).clamp(min=1e-8)
        normalized = (flat - min_val) / range_val
        
        return normalized.reshape_as(tensor)
    
    def forward(
        self,
        latent: torch.Tensor,
        clip_saliency: Optional[torch.Tensor] = None,
        attention_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined energy map.
        
        Args:
            latent: Current latent [B, C, H, W]
            clip_saliency: CLIP gradient saliency [B, 1, H, W] (optional)
            attention_maps: Cross-attention maps [B, (heads,) H, W] (optional)
            
        Returns:
            Combined energy map [B, 1, H, W] in [0, 1]
        """
        B, C, H, W = latent.shape
        device = latent.device
        dtype = latent.dtype
        
        # Always compute Sobel energy
        sobel_energy = self.compute_sobel_energy(latent)
        energy = self.gamma * sobel_energy
        
        # Add CLIP saliency if provided
        if clip_saliency is not None:
            clip_norm = self._normalize(clip_saliency)
            energy = energy + self.alpha * clip_norm
        
        # Add attention maps if provided
        if attention_maps is not None:
            attn_energy = self.process_attention_maps(attention_maps, (H, W))
            energy = energy + self.beta * attn_energy
        
        # Final normalization
        energy = self._normalize(energy)
        
        return energy


class CLIPSaliencyComputer(nn.Module):
    """Computes CLIP gradient saliency for latent space.
    
    Uses a projection head from latent space to CLIP embedding space,
    then computes gradients of similarity w.r.t. the estimated clean latent.
    """
    
    def __init__(self, latent_clip_projector: nn.Module):
        """Initialize CLIP saliency computer.
        
        Args:
            latent_clip_projector: Module that projects latents to CLIP space.
        """
        super().__init__()
        self.projector = latent_clip_projector
    
    def compute_estimated_z0(
        self,
        z_t: torch.Tensor,
        noise_pred: torch.Tensor,
        alpha_cumprod_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute estimated clean latent from noisy latent and noise prediction.
        
        z_0 ≈ (z_t - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
        
        Args:
            z_t: Noisy latent [B, C, H, W]
            noise_pred: Predicted noise [B, C, H, W]
            alpha_cumprod_t: Cumulative alpha at timestep t [B] or scalar
            
        Returns:
            Estimated clean latent [B, C, H, W]
        """
        if alpha_cumprod_t.dim() == 0:
            alpha_cumprod_t = alpha_cumprod_t.unsqueeze(0)
        
        # Reshape for broadcasting
        alpha = alpha_cumprod_t.reshape(-1, 1, 1, 1)
        
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)
        
        z_0_est = (z_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        
        return z_0_est
    
    def compute_saliency(
        self,
        z_0_est: torch.Tensor,
        text_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute CLIP gradient saliency.
        
        Args:
            z_0_est: Estimated clean latent [B, C, H, W]
            text_embedding: Text CLIP embedding [B, D]
            
        Returns:
            Saliency map [B, 1, H, W]
        """
        # Enable gradients for saliency computation
        z_0_est = z_0_est.detach().requires_grad_(True)
        
        # Project to CLIP space
        image_embedding = self.projector(z_0_est)  # [B, D]
        
        # Normalize embeddings
        image_embedding = F.normalize(image_embedding, dim=-1)
        text_embedding = F.normalize(text_embedding, dim=-1)
        
        # Compute cosine similarity
        similarity = (image_embedding * text_embedding).sum(dim=-1)  # [B]
        
        # Compute gradients
        grad = torch.autograd.grad(
            outputs=similarity.sum(),
            inputs=z_0_est,
            create_graph=False,
            retain_graph=False
        )[0]  # [B, C, H, W]
        
        # Take L2 norm across channels for saliency
        saliency = torch.norm(grad, dim=1, keepdim=True)  # [B, 1, H, W]
        
        return saliency
