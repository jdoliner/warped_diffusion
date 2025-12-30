"""Latent-CLIP projection head.

Projects VAE latent space to CLIP embedding space for saliency computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCLIPProjector(nn.Module):
    """Projects VAE latents to CLIP embedding space.

    Architecture:
    - Convolutional feature extraction (preserves spatial info longer)
    - Global average pooling
    - MLP projection to CLIP dimension

    Trained to maximize cosine similarity with frozen CLIP embedding.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 64,
        clip_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        pool_size: int = 8,  # kept for backward compat, not used
    ):
        """Initialize Latent-CLIP projector.

        Args:
            latent_channels: Number of VAE latent channels (4 for SD).
            latent_size: Spatial size of latent (64 for 512px SD).
            clip_dim: CLIP embedding dimension (768 for ViT-L/14).
            hidden_dim: Hidden layer dimension.
            num_layers: Number of MLP layers.
            pool_size: Deprecated, kept for compatibility.
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.clip_dim = clip_dim

        # Convolutional feature extractor - gradually reduce spatial size
        # while increasing channels to preserve information
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32, 4 -> 64 channels
            nn.Conv2d(latent_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # 32x32 -> 16x16, 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            # 16x16 -> 8x8, 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            # 8x8 -> 4x4, 256 -> 512 channels
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
        )

        # After conv: 512 channels × 4 × 4 = 8192 features
        # Global average pool to 512
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent to CLIP embedding space.

        Args:
            latent: VAE latent [B, C, H, W]

        Returns:
            CLIP-like embedding [B, clip_dim]
        """
        # Extract features with convolutions
        x = self.conv_layers(latent)  # [B, 512, 4, 4]

        # Global average pooling
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.flatten(start_dim=1)  # [B, 512]

        # Project to CLIP space
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
        device: torch.device = torch.device("cuda"),
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
            self.projector.parameters(), lr=learning_rate, weight_decay=0.01
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

    def train_step(self, images: torch.Tensor, images_for_clip: torch.Tensor) -> dict[str, float]:
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

        return {"loss": loss.item(), "cosine_similarity": cosine_sim.item()}

    def save(self, path: str):
        """Save projector weights."""
        torch.save(self.projector.state_dict(), path)

    def load(self, path: str):
        """Load projector weights."""
        self.projector.load_state_dict(torch.load(path, map_location=self.device))
