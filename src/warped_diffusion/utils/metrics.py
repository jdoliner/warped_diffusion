"""Evaluation metrics for warped diffusion."""

import torch
import torch.nn.functional as F
from typing import Optional
import lpips


class ReflectionConsistencyMetric:
    """Measures reflection/symmetry consistency in generated images.
    
    For prompts like "A man in a red hat looking into a blue mirror",
    measures the color similarity between the object and its reflection
    using segmented LPIPS.
    """
    
    def __init__(self, device: torch.device = torch.device('cuda')):
        """Initialize metric.
        
        Args:
            device: Torch device
        """
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Compute LPIPS between two images or image regions.
        
        Args:
            image1: First image [B, 3, H, W] in [-1, 1]
            image2: Second image [B, 3, H, W] in [-1, 1]
            mask: Optional binary mask [B, 1, H, W]
            
        Returns:
            LPIPS score (lower = more similar)
        """
        if mask is not None:
            # Apply mask
            image1 = image1 * mask
            image2 = image2 * mask
        
        score = self.lpips_fn(image1, image2)
        return score.mean().item()
    
    def extract_reflection_regions(
        self,
        image: torch.Tensor,
        left_ratio: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract left and right halves as proxy for object/reflection.
        
        This is a simple heuristic - for better results, use segmentation.
        
        Args:
            image: Image [B, 3, H, W]
            left_ratio: Fraction of image that's the "left" region
            
        Returns:
            Tuple of (left_region, right_region_flipped)
        """
        _, _, H, W = image.shape
        split_point = int(W * left_ratio)
        
        left = image[:, :, :, :split_point]
        right = image[:, :, :, split_point:]
        
        # Flip right horizontally for comparison
        right_flipped = torch.flip(right, dims=[-1])
        
        # Resize to match if needed
        if left.shape[-1] != right_flipped.shape[-1]:
            min_w = min(left.shape[-1], right_flipped.shape[-1])
            left = F.interpolate(left, size=(H, min_w), mode='bilinear')
            right_flipped = F.interpolate(right_flipped, size=(H, min_w), mode='bilinear')
        
        return left, right_flipped
    
    @torch.no_grad()
    def compute_reflection_consistency(self, image: torch.Tensor) -> float:
        """Compute reflection consistency score.
        
        Args:
            image: Generated image [B, 3, H, W] in [-1, 1]
            
        Returns:
            Reflection consistency score (higher = more consistent)
        """
        left, right_flipped = self.extract_reflection_regions(image)
        lpips_score = self.compute_lpips(left, right_flipped)
        
        # Convert to consistency score (1 - lpips for higher = better)
        consistency = 1.0 - min(lpips_score, 1.0)
        return consistency


class ColorConsistencyMetric:
    """Measures color consistency between specified regions."""
    
    @staticmethod
    def extract_dominant_color(
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract dominant color from image or masked region.
        
        Args:
            image: Image [B, 3, H, W]
            mask: Optional binary mask [B, 1, H, W]
            
        Returns:
            Dominant color [B, 3]
        """
        if mask is not None:
            # Masked mean
            mask = mask.expand_as(image)
            masked_sum = (image * mask).sum(dim=[-1, -2])
            mask_count = mask.sum(dim=[-1, -2]).clamp(min=1)
            color = masked_sum / mask_count
        else:
            color = image.mean(dim=[-1, -2])
        
        return color
    
    @staticmethod
    def color_similarity(color1: torch.Tensor, color2: torch.Tensor) -> float:
        """Compute cosine similarity between colors.
        
        Args:
            color1: First color [B, 3]
            color2: Second color [B, 3]
            
        Returns:
            Similarity score in [0, 1]
        """
        color1_norm = F.normalize(color1, dim=-1)
        color2_norm = F.normalize(color2, dim=-1)
        similarity = (color1_norm * color2_norm).sum(dim=-1).mean()
        return similarity.item()


def compute_warp_quality_metrics(
    original: torch.Tensor,
    warped: torch.Tensor,
    reconstructed: torch.Tensor
) -> dict[str, float]:
    """Compute quality metrics for warp/unwarp cycle.
    
    Args:
        original: Original tensor [B, C, H, W]
        warped: Warped tensor [B, C, H, W]
        reconstructed: Tensor after warp->inverse_warp [B, C, H, W]
        
    Returns:
        Dictionary of metrics
    """
    # Reconstruction MSE (should be small for invertible warp)
    reconstruction_mse = F.mse_loss(reconstructed, original).item()
    
    # Warp magnitude (how much was changed)
    warp_magnitude = F.mse_loss(warped, original).item()
    
    # PSNR
    mse = reconstruction_mse + 1e-10
    psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    
    return {
        "reconstruction_mse": reconstruction_mse,
        "warp_magnitude": warp_magnitude,
        "reconstruction_psnr": psnr,
    }
