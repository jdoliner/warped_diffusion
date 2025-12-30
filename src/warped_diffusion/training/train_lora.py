"""LoRA training for warped diffusion.

Trains the U-Net with LoRA adapters to handle warped latent distributions.
Uses random smooth warps for augmentation to teach equivariance.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms

from ..warp.tps import ThinPlateSplineWarp
from ..warp.grid import ControlGrid, generate_random_warp


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    
    # Model
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    
    # Training
    num_train_steps: int = 10000
    batch_size: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Warp
    grid_size: int = 8
    max_displacement: float = 0.3
    tv_loss_weight: float = 0.01
    
    # Data
    dataset_name: str = "detection-datasets/coco"
    dataset_config: str = "2017"
    image_column: str = "image"
    caption_column: str = "sentences"
    resolution: int = 512
    num_workers: int = 8
    
    # Logging
    output_dir: str = "outputs/lora_training"
    logging_steps: int = 50
    save_steps: int = 1000
    sample_steps: int = 500
    
    # Hardware
    device: str = "cuda"
    seed: int = 42


class COCODataset(torch.utils.data.Dataset):
    """MS-COCO dataset for training."""
    
    def __init__(
        self,
        dataset,
        tokenizer: CLIPTokenizer,
        image_column: str = "image",
        caption_column: str = "sentences",
        resolution: int = 512,
    ):
        """Initialize dataset.
        
        Args:
            dataset: HuggingFace dataset
            tokenizer: CLIP tokenizer
            image_column: Column name for images
            caption_column: Column name for captions
            resolution: Target image resolution
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_column = image_column
        self.caption_column = caption_column
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
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
        
        pixel_values = self.transform(image)
        
        # Get caption - handle different formats
        caption = item[self.caption_column]
        if isinstance(caption, list):
            # COCO has multiple captions, pick first or random
            if len(caption) > 0:
                if isinstance(caption[0], dict):
                    caption = caption[0].get('raw', caption[0].get('text', ''))
                else:
                    caption = caption[0]
            else:
                caption = ""
        elif isinstance(caption, dict):
            caption = caption.get('raw', caption.get('text', ''))
        
        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


class WarpedLoRATrainer:
    """Trainer for LoRA with warp augmentation."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed
        torch.manual_seed(config.seed)
        
        # Load models
        self._load_models()
        
        # Setup warp
        self._setup_warp()
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup data
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup logging
        self._setup_logging()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None
    
    def _load_models(self):
        """Load pretrained models."""
        print("Loading pretrained models...")
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="vae",
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        # Text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="text_encoder",
        ).to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        # Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="tokenizer",
        )
        
        # U-Net (will add LoRA)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="unet",
        ).to(self.device)
        
        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name,
            subfolder="scheduler",
        )
        
        print("Models loaded.")
    
    def _setup_warp(self):
        """Setup warping components."""
        self.tps = ThinPlateSplineWarp(grid_size=self.config.grid_size)
        self.control_grid = ControlGrid(
            grid_size=self.config.grid_size,
            max_displacement=self.config.max_displacement,
        )
    
    def _setup_lora(self):
        """Setup LoRA adapters on U-Net."""
        print("Setting up LoRA...")
        
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            init_lora_weights="gaussian",
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Loading dataset...")
        
        # Load MS-COCO
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split="train",
            trust_remote_code=True,
        )
        
        # Create dataset
        self.dataset = COCODataset(
            dataset,
            self.tokenizer,
            image_column=self.config.image_column,
            caption_column=self.config.caption_column,
            resolution=self.config.resolution,
        )
        
        # Create dataloader with multiprocessing
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
        """Setup optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=self.config.num_train_steps,
        )
    
    def _setup_logging(self):
        """Setup tensorboard logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.config.output_dir, "logs"))
    
    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to latents."""
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings."""
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        return encoder_hidden_states
    
    def apply_random_warp(
        self,
        latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random smooth warp to latents.
        
        Args:
            latents: Input latents [B, C, H, W]
            
        Returns:
            Tuple of (warped_latents, src_points, dst_points)
        """
        batch_size = latents.shape[0]
        
        # Generate random warp
        src_points, dst_points = generate_random_warp(
            batch_size=batch_size,
            grid_size=self.config.grid_size,
            max_displacement=self.config.max_displacement,
            device=latents.device,
            dtype=latents.dtype,
        )
        
        # Apply warp
        warped_latents = self.tps.warp(latents, src_points, dst_points)
        
        return warped_latents, src_points, dst_points
    
    def inverse_warp(
        self,
        tensor: torch.Tensor,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
    ) -> torch.Tensor:
        """Apply inverse warp to tensor."""
        return self.tps.warp_inverse(tensor, src_points, dst_points)
    
    def compute_tv_loss(
        self,
        src_points: torch.Tensor,
        dst_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total variation loss on displacement."""
        return self.control_grid.compute_total_variation(src_points, dst_points)
    
    def train_step(
        self,
        batch: dict,
        step: int,
    ) -> dict[str, float]:
        """Perform one training step.
        
        Args:
            batch: Data batch
            step: Current training step
            
        Returns:
            Dictionary of metrics
        """
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        
        # Encode images and text
        latents = self.encode_images(pixel_values)
        encoder_hidden_states = self.encode_text(input_ids)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Apply random warp to noisy latents
        warped_latents, src_points, dst_points = self.apply_random_warp(noisy_latents)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision == "fp16"):
            # Predict noise on warped latents
            noise_pred = self.unet(
                warped_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]
            
            # Inverse warp the noise prediction
            noise_pred_unwarped = self.inverse_warp(noise_pred, src_points, dst_points)
            
            # MSE loss between original noise and unwarped prediction
            mse_loss = F.mse_loss(noise_pred_unwarped, noise)
            
            # TV regularization on displacement field
            tv_loss = self.compute_tv_loss(src_points, dst_points)
            
            # Total loss
            loss = mse_loss + self.config.tv_loss_weight * tv_loss
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
        else:
            loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
        
        return {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "tv_loss": tv_loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        self.unet.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save({
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"Checkpoint saved at step {step}")
    
    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.config.num_train_steps} steps...")
        
        self.unet.train()
        
        # Create infinite data iterator
        data_iter = iter(self.dataloader)
        
        progress_bar = tqdm(range(self.config.num_train_steps), desc="Training")
        running_loss = 0.0
        
        for step in progress_bar:
            # Get batch (with cycling)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # Train step
            metrics = self.train_step(batch, step)
            running_loss += metrics["loss"]
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = running_loss / self.config.logging_steps
                running_loss = 0.0
                
                # Tensorboard
                for key, value in metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, step)
                
                # Progress bar
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "mse": f"{metrics['mse_loss']:.4f}",
                    "tv": f"{metrics['tv_loss']:.4f}",
                    "lr": f"{metrics['lr']:.2e}",
                })
                
                # Print to stdout
                print(f"Step {step + 1}: loss={avg_loss:.4f}, mse={metrics['mse_loss']:.4f}, "
                      f"tv={metrics['tv_loss']:.4f}, lr={metrics['lr']:.2e}")
            
            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)
        
        # Final save
        self.save_checkpoint(self.config.num_train_steps)
        print("Training complete!")
        
        self.writer.close()


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LoRA for warped diffusion")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_training")
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )
    
    trainer = WarpedLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
