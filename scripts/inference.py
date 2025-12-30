#!/usr/bin/env python3
"""Inference script for warped diffusion."""

import sys
from pathlib import Path
import argparse
import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from warped_diffusion.pipeline.warped_diffusion_pipeline import WarpedDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate images with warped diffusion")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to base model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--latent_clip_path", type=str, default=None,
                        help="Path to Latent-CLIP projector weights")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--enable_warp", action="store_true", default=True,
                        help="Enable semantic warping")
    parser.add_argument("--no_warp", action="store_true", help="Disable semantic warping")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load pipeline
    print(f"Loading model from {args.model_path}...")
    
    warp_config = {
        "grid_size": 8,
        "max_displacement": 0.3,
        "epsilon_floor": 0.1,
        "energy_alpha": 0.5,
        "energy_beta": 0.4,
        "energy_gamma": 0.1,
        "t_structural": 700,
        "t_refinement": 300,
    }
    
    pipe = WarpedDiffusionPipeline.from_pretrained_with_warp(
        args.model_path,
        latent_clip_path=args.latent_clip_path,
        warp_config=warp_config,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(args.device)
    
    # Load LoRA if provided
    if args.lora_path is not None:
        print(f"Loading LoRA weights from {args.lora_path}...")
        pipe.unet.load_adapter(args.lora_path)
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Determine warp setting
    enable_warp = args.enable_warp and not args.no_warp
    
    # Generate
    print(f"Generating image with prompt: {args.prompt}")
    print(f"Warping enabled: {enable_warp}")
    
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=generator,
        enable_warp=enable_warp,
    )
    
    # Save
    image = result.images[0]
    image.save(args.output)
    print(f"Image saved to {args.output}")


if __name__ == "__main__":
    main()
