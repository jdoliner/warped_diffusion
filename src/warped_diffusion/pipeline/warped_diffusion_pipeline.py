"""Warped Diffusion Pipeline - Modified Stable Diffusion with semantic warping."""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Callable, Any
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPTextModel, CLIPTokenizer

from ..models.semantic_warp_layer import SemanticWarpLayer
from ..models.latent_clip import LatentCLIPProjector
from ..utils.attention_hooks import (
    setup_attention_extraction,
    get_attention_maps_from_processors,
    CrossAttentionProcessor,
)


class WarpedDiffusionPipeline(StableDiffusionPipeline):
    """Stable Diffusion pipeline with semantic warping.
    
    Wraps the standard SD pipeline to apply semantic warping before
    the U-Net and inverse warping after noise prediction.
    """
    
    def __init__(
        self,
        vae,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet,
        scheduler: DDPMScheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker: bool = False,
        # Warp-specific arguments
        latent_clip_projector: Optional[LatentCLIPProjector] = None,
        warp_config: Optional[dict] = None,
    ):
        """Initialize WarpedDiffusionPipeline.
        
        Args:
            vae: VAE model
            text_encoder: CLIP text encoder
            tokenizer: CLIP tokenizer
            unet: U-Net model
            scheduler: Diffusion scheduler
            safety_checker: Optional safety checker
            feature_extractor: Optional feature extractor
            requires_safety_checker: Whether to require safety checker
            latent_clip_projector: Trained projector for CLIP saliency
            warp_config: Configuration for semantic warp layer
        """
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )
        
        # Default warp config
        if warp_config is None:
            warp_config = {}
        
        # Initialize semantic warp layer
        self.warp_layer = SemanticWarpLayer(
            grid_size=warp_config.get('grid_size', 8),
            max_displacement=warp_config.get('max_displacement', 0.3),
            epsilon_floor=warp_config.get('epsilon_floor', 0.1),
            energy_alpha=warp_config.get('energy_alpha', 0.5),
            energy_beta=warp_config.get('energy_beta', 0.4),
            energy_gamma=warp_config.get('energy_gamma', 0.1),
            t_structural=warp_config.get('t_structural', 700),
            t_refinement=warp_config.get('t_refinement', 300),
            num_timesteps=warp_config.get('num_timesteps', 1000),
            latent_clip_projector=latent_clip_projector,
        )
        
        # Setup attention extraction
        self.attention_processors: dict[str, CrossAttentionProcessor] = {}
        self._setup_attention_extraction()
    
    def _setup_attention_extraction(self):
        """Setup attention extraction from U-Net."""
        self.attention_processors = setup_attention_extraction(self.unet)
    
    def _get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get aggregated attention maps from last forward pass."""
        return get_attention_maps_from_processors(
            self.attention_processors,
            target_size=(64, 64)
        )
    
    def _get_text_embedding_for_clip(
        self, 
        prompt_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Extract text embedding for CLIP saliency.
        
        Args:
            prompt_embeds: Prompt embeddings from text encoder [B, seq_len, dim]
            
        Returns:
            Pooled text embedding [B, dim]
        """
        # Use the EOS token embedding (last non-padding token)
        # For simplicity, use mean pooling
        return prompt_embeds.mean(dim=1)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        # Warp-specific
        enable_warp: bool = True,
    ):
        """Generate images with semantic warping.
        
        Most arguments are identical to StableDiffusionPipeline.
        
        Args:
            enable_warp: Whether to enable semantic warping (default True)
        """
        # Set default height/width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Check inputs and prepare
        self.check_inputs(
            prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        
        # Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]  # type: ignore
        
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Encode prompt
        prompt_embeds_out = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        # Handle different return types from encode_prompt
        if isinstance(prompt_embeds_out, tuple):
            prompt_embeds, negative_prompt_embeds = prompt_embeds_out[:2]
        else:
            prompt_embeds = prompt_embeds_out
            negative_prompt_embeds = None
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latents
        num_channels_latent = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latent,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # Prepare extra kwargs for scheduler step
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Get text embedding for CLIP saliency
        text_embedding = self._get_text_embedding_for_clip(prompt_embeds)
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            # Use only the positive prompt embedding for saliency
            text_embedding = text_embedding[batch_size * num_images_per_prompt:]
        
        # Track previous noise prediction for z_0 estimation
        prev_noise_pred: Optional[torch.Tensor] = None
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Get alpha_cumprod for this timestep
                alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
                
                # Get attention maps from previous step
                attention_maps = self._get_attention_maps() if enable_warp else None
                
                # Apply semantic warp if enabled
                if enable_warp:
                    latents_warped = self.warp_layer.forward_warp(
                        latent=latents,
                        timestep=int(t),
                        attention_maps=attention_maps,
                        text_embedding=text_embedding,
                        noise_pred=prev_noise_pred,
                        alpha_cumprod_t=alpha_cumprod_t,
                    )
                else:
                    latents_warped = latents
                
                # Expand for classifier-free guidance
                latent_model_input = torch.cat([latents_warped] * 2) if do_classifier_free_guidance else latents_warped
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Prepare prompt embeds for U-Net
                if do_classifier_free_guidance and negative_prompt_embeds is not None:
                    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
                else:
                    prompt_embeds_input = prompt_embeds
                
                # Predict noise
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_input,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                
                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Apply inverse warp to noise prediction if warping was applied
                if enable_warp:
                    noise_pred = self.warp_layer.inverse_warp(noise_pred)
                
                # Store for next iteration's z_0 estimation
                prev_noise_pred = noise_pred.detach()
                
                # Scheduler step on unwarped latent
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)  # type: ignore
        
        # Decode latents
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        
        # Post-process
        if output_type != "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    @classmethod
    def from_pretrained_with_warp(
        cls,
        pretrained_model_name_or_path: str,
        latent_clip_path: Optional[str] = None,
        warp_config: Optional[dict] = None,
        **kwargs,
    ):
        """Load pipeline from pretrained with warp configuration.
        
        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            latent_clip_path: Path to trained LatentCLIP projector weights
            warp_config: Warp layer configuration
            **kwargs: Additional arguments for from_pretrained
            
        Returns:
            WarpedDiffusionPipeline instance
        """
        # Load base pipeline components
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        # Load latent CLIP projector if provided
        latent_clip_projector = None
        if latent_clip_path is not None:
            latent_clip_projector = LatentCLIPProjector()
            latent_clip_projector.load_state_dict(
                torch.load(latent_clip_path, map_location='cpu')
            )
        
        # Create warped pipeline
        return cls(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            unet=pipe.unet,
            scheduler=pipe.scheduler,
            safety_checker=pipe.safety_checker,
            feature_extractor=pipe.feature_extractor,
            latent_clip_projector=latent_clip_projector,
            warp_config=warp_config,
        )
