"""Training script for Latent-CLIP projection head.

Trains a projection head to map VAE latents to CLIP embedding space.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from diffusers import AutoencoderKL
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms

from ..models.latent_clip import LatentCLIPProjector


@dataclass
class LatentCLIPConfig:
    """Configuration for Latent-CLIP training."""

    # Model
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    clip_model_name: str = "openai/clip-vit-large-patch14"

    # Projector architecture
    latent_channels: int = 4
    latent_size: int = 64
    clip_dim: int = 768
    hidden_dim: int = 1024
    num_layers: int = 3
    pool_size: int = 8

    # Training
    num_train_steps: int = 5000
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Data
    dataset_name: str = "detection-datasets/coco"
    dataset_config: Optional[str] = None  # Use default config
    image_column: str = "image"
    resolution: int = 512
    num_workers: int = 8

    # Logging
    output_dir: str = "outputs/latent_clip"
    logging_steps: int = 50
    save_steps: int = 1000

    # Hardware
    device: str = "cuda"
    seed: int = 42


class ImageDataset(torch.utils.data.Dataset):
    """Simple image dataset for Latent-CLIP training."""

    def __init__(
        self,
        dataset,
        clip_processor: CLIPProcessor,
        image_column: str = "image",
        resolution: int = 512,
    ):
        self.dataset = dataset
        self.clip_processor = clip_processor
        self.image_column = image_column

        # Transform for VAE input (512x512, normalized to [-1, 1])
        self.vae_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # VAE input
        vae_input = self.vae_transform(image)

        # CLIP input (processor handles resize, crop, normalize)
        clip_input = self.clip_processor(images=image, return_tensors="pt")
        clip_pixel_values = clip_input["pixel_values"].squeeze(0)

        return {
            "vae_input": vae_input,
            "clip_input": clip_pixel_values,
        }


class LatentCLIPTrainer:
    """Trainer for Latent-CLIP projection head."""

    def __init__(self, config: LatentCLIPConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)

        self._load_models()
        self._setup_data()
        self._setup_optimizer()
        self._setup_logging()

    def _load_models(self):
        """Load VAE and CLIP models."""
        print("Loading models...")

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="vae",
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # CLIP
        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)

        # Latent-CLIP projector
        self.projector = LatentCLIPProjector(
            latent_channels=self.config.latent_channels,
            latent_size=self.config.latent_size,
            clip_dim=self.config.clip_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            pool_size=self.config.pool_size,
        ).to(self.device)

        print("Models loaded.")

    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Loading dataset...")

        load_kwargs = {
            "path": self.config.dataset_name,
            "split": "train",
            "trust_remote_code": True,
        }
        if self.config.dataset_config is not None:
            load_kwargs["name"] = self.config.dataset_config

        dataset = load_dataset(**load_kwargs)

        self.dataset = ImageDataset(
            dataset,
            self.clip_processor,
            image_column=self.config.image_column,
            resolution=self.config.resolution,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )

        print(f"Dataset loaded: {len(self.dataset)} samples")

    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

    def _setup_logging(self):
        """Setup tensorboard logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.config.output_dir, "logs"))

    @torch.no_grad()
    def encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latents."""
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_clip_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Get CLIP image embedding."""
        embedding = self.clip_model.get_image_features(pixel_values=images)
        return embedding

    def train_step(self, batch: dict) -> dict[str, float]:
        """Perform one training step."""
        vae_input = batch["vae_input"].to(self.device)
        clip_input = batch["clip_input"].to(self.device)

        # Encode to latent
        latents = self.encode_to_latent(vae_input)

        # Get target CLIP embedding
        clip_embedding = self.get_clip_embedding(clip_input)
        clip_embedding = F.normalize(clip_embedding, dim=-1)

        # Forward through projector
        self.projector.train()
        projected = self.projector(latents)
        projected = F.normalize(projected, dim=-1)

        # Cosine similarity loss (1 - cos_sim) - works better for embeddings
        cosine_sim = (projected * clip_embedding).sum(dim=-1).mean()
        loss = 1.0 - cosine_sim

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "cosine_similarity": cosine_sim.item(),
        }

    def save_checkpoint(self, step: int):
        """Save projector weights."""
        checkpoint_path = os.path.join(self.config.output_dir, f"latent_clip_{step}.pt")
        torch.save(self.projector.state_dict(), checkpoint_path)
        print(f"Saved checkpoint at step {step}")

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.config.num_train_steps} steps...")

        data_iter = iter(self.dataloader)
        progress_bar = tqdm(range(self.config.num_train_steps), desc="Training Latent-CLIP")
        running_loss = 0.0
        running_sim = 0.0

        for step in progress_bar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            running_loss += metrics["loss"]
            running_sim += metrics["cosine_similarity"]

            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = running_loss / self.config.logging_steps
                avg_sim = running_sim / self.config.logging_steps
                running_loss = 0.0
                running_sim = 0.0

                self.writer.add_scalar("train/loss", avg_loss, step)
                self.writer.add_scalar("train/cosine_similarity", avg_sim, step)

                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "cos_sim": f"{avg_sim:.4f}",
                    }
                )

                print(f"Step {step + 1}: loss={avg_loss:.4f}, cos_sim={avg_sim:.4f}")

            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)

        # Final save
        self.save_checkpoint(self.config.num_train_steps)
        print("Training complete!")

        self.writer.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Latent-CLIP projector")
    parser.add_argument("--output_dir", type=str, default="outputs/latent_clip")
    parser.add_argument("--num_train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = LatentCLIPConfig(
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = LatentCLIPTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
