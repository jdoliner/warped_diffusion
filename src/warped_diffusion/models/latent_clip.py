"""Latent-CLIP projection head.

Projects VAE latent space to CLIP embedding space for saliency computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCLIPProjector(nn.Module):
    """Projects VAE latents to CLIP embedding space.
    
    Architecture:
    - Spatial pooling (adaptive average pool)
    - MLP projection to CLIP dimension
    
    Trained to minimize MSE between projected latent and frozen CLIP embedding.
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        clip_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        pool_size: int = 8,
    ):
        """Initialize Latent-CLIP projector.
        
        Args:
            latent_channels: Number of VAE latent channels (4 for SD).
            latent_size: Spatial size of latent (64 for 512px SD).
            clip_dim: CLIP embedding dimension (768 for ViT-L/14).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of MLP layers.
            pool_size: Spatial size after pooling.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.clip_dim = clip_dim
        self.pool_size = pool_size
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        # Input size after pooling and flattening
        input_dim = latent_channels * pool_size * pool_size
        
        # Build MLP
        layers: list[nn.Module] = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        
        # Final projection to CLIP dim
        layers.append(nn.Linear(current_dim, clip_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent to CLIP embedding space.
        
        Args:
            latent: VAE latent [B, C, H, W]
            
        Returns:
            CLIP-like embedding [B, clip_dim]
        """
        # Pool spatially
        x = self.pool(latent)  # [B, C, pool_size, pool_size]
        
        # Flatten
        x = x.flatten(start_dim=1)  # [B, C * pool_size * pool_size]
        
        # Project through MLP
        embedding = self.mlp(x)  # [B, clip_dim]
        
        return embedding


class LatentCLIPTrainer:
    """Trainer for the Latent-CLIP projection head.
    
    Trains the projector to minimize MSE between projected latent
    and frozen CLIP image embedding.
    """
    
    def __init__(
        self,
        projector: LatentCLIPProjector,
        clip_model: nn.Module,
        vae: nn.Module,
        learning_rate: float = 1e-4,
        device: torch.device = torch.device('cuda'),
    ):
        """Initialize trainer.
        
        Args:
            projector: The LatentCLIPProjector to train.
            clip_model: Frozen CLIP model for target embeddings.
            vae: Frozen VAE for encoding images to latents.
            learning_rate: Learning rate for optimizer.
            device: Device to train on.
        """
        self.projector = projector.to(device)
        self.clip_model = clip_model.to(device)
        self.vae = vae.to(device)
        self.device = device
        
        # Freeze CLIP and VAE
        self.clip_model.eval()
        self.vae.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
    
    @torch.no_grad()
    def encode_image_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents.
        
        Args:
            images: RGB images [B, 3, H, W] in [-1, 1]
            
        Returns:
            Latents [B, 4, H//8, W//8]
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def get_clip_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embedding.
        
        Args:
            images: RGB images [B, 3, H, W] in [0, 1] or preprocessed
            
        Returns:
            CLIP embedding [B, clip_dim]
        """
        # Assume images are already preprocessed for CLIP
        embedding = self.clip_model.get_image_features(images)
        return embedding
    
    def train_step(
        self, 
        images: torch.Tensor,
        images_for_clip: torch.Tensor
    ) -> dict[str, float]:
        """Perform one training step.
        
        Args:
            images: Images for VAE encoding [B, 3, H, W] in [-1, 1]
            images_for_clip: Images preprocessed for CLIP [B, 3, 224, 224]
            
        Returns:
            Dictionary of metrics
        """
        self.projector.train()
        
        # Encode to latent
        latents = self.encode_image_to_latent(images)
        
        # Get target CLIP embedding
        clip_embedding = self.get_clip_embedding(images_for_clip)
        clip_embedding = F.normalize(clip_embedding, dim=-1)
        
        # Forward through projector
        projected = self.projector(latents)
        projected = F.normalize(projected, dim=-1)
        
        # MSE loss on normalized embeddings (equivalent to 1 - cosine similarity)
        loss = F.mse_loss(projected, clip_embedding)
        
        # Compute cosine similarity for logging
        cosine_sim = (projected * clip_embedding).sum(dim=-1).mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'cosine_similarity': cosine_sim.item()
        }
    
    def save(self, path: str):
        """Save projector weights."""
        torch.save(self.projector.state_dict(), path)
    
    def load(self, path: str):
        """Load projector weights."""
        self.projector.load_state_dict(torch.load(path, map_location=self.device))
