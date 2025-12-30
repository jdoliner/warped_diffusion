"""Attention extraction hooks for U-Net cross-attention layers."""

import torch
import torch.nn as nn
from typing import Optional
from collections import defaultdict


class AttentionExtractor:
    """Extracts cross-attention maps from U-Net during forward pass.
    
    Registers hooks on specified attention layers to capture attention weights.
    Attention maps are detached to prevent gradient flow through the warp.
    """
    
    def __init__(
        self,
        unet: nn.Module,
        layer_names: Optional[list[str]] = None,
        target_resolutions: tuple[int, ...] = (16, 32),
    ):
        """Initialize attention extractor.
        
        Args:
            unet: The U-Net model to extract attention from.
            layer_names: Specific layer names to hook. If None, auto-detect.
            target_resolutions: Only extract from layers with these spatial sizes.
        """
        self.unet = unet
        self.target_resolutions = target_resolutions
        self.attention_maps: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        
        if layer_names is None:
            layer_names = self._find_cross_attention_layers()
        
        self.layer_names = layer_names
        self._register_hooks()
    
    def _find_cross_attention_layers(self) -> list[str]:
        """Auto-detect cross-attention layers in U-Net."""
        cross_attn_layers = []
        
        for name, module in self.unet.named_modules():
            # Look for cross-attention layers (diffusers naming convention)
            if hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                # Check if it's cross-attention (has different key/value source)
                if 'attn2' in name or 'cross' in name.lower():
                    cross_attn_layers.append(name)
        
        return cross_attn_layers
    
    def _make_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            # For cross-attention, we want the attention weights
            # These are computed as softmax(QK^T / sqrt(d))
            # We need to compute them ourselves from the module state
            
            # Get the attention scores if available
            # Note: This depends on the specific attention implementation
            if hasattr(module, 'attention_scores'):
                attn = module.attention_scores.detach()  # type: ignore
            elif hasattr(module, 'attn_weights'):
                attn = module.attn_weights.detach()  # type: ignore
            else:
                # Fallback: use output magnitude as proxy
                # Reshape output to get spatial attention-like map
                attn = output.detach()
                if attn.dim() == 3:  # [B, seq_len, dim]
                    B, seq_len, _ = attn.shape
                    h = w = int(seq_len ** 0.5)
                    if h * w == seq_len:
                        attn = attn.norm(dim=-1).reshape(B, h, w)
            
            self.attention_maps[name] = attn
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on attention layers."""
        for name, module in self.unet.named_modules():
            if name in self.layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear stored attention maps."""
        self.attention_maps = {}
    
    def get_aggregated_attention(
        self, 
        target_size: tuple[int, int] = (64, 64)
    ) -> Optional[torch.Tensor]:
        """Get aggregated attention map across all captured layers.
        
        Args:
            target_size: (H, W) to resize all maps to.
            
        Returns:
            Aggregated attention [B, 1, H, W] or None if no maps captured.
        """
        if not self.attention_maps:
            return None
        
        aggregated = None
        count = 0
        
        for name, attn in self.attention_maps.items():
            # Ensure 4D: [B, 1, H, W]
            if attn.dim() == 2:
                attn = attn.unsqueeze(0).unsqueeze(0)
            elif attn.dim() == 3:
                attn = attn.unsqueeze(1)
            
            # Resize to target
            attn_resized = torch.nn.functional.interpolate(
                attn.float(),
                size=target_size,
                mode='bilinear',
                align_corners=True
            )
            
            if aggregated is None:
                aggregated = attn_resized
            else:
                aggregated = aggregated + attn_resized
            
            count += 1
        
        if aggregated is not None and count > 0:
            aggregated = aggregated / count
        
        return aggregated
    
    def __enter__(self):
        """Context manager entry."""
        self.clear()
        return self
    
    def __exit__(self, *args):
        """Context manager exit - remove hooks."""
        self.remove_hooks()


class CrossAttentionProcessor:
    """Custom attention processor that stores attention weights.
    
    Use this as a drop-in replacement for the default attention processor
    in diffusers to capture attention weights during forward pass.
    """
    
    def __init__(self, store_attention: bool = True):
        """Initialize processor.
        
        Args:
            store_attention: Whether to store attention weights.
        """
        self.store_attention = store_attention
        self.attention_weights: Optional[torch.Tensor] = None
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Process attention and optionally store weights."""
        residual = hidden_states
        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)  # type: ignore

        input_ndim = hidden_states.ndim
        channel = height = width = 0
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)  # type: ignore

        query = attn.to_q(hidden_states)  # type: ignore

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # Store cross-attention weights (not self-attention)
        if self.store_attention and is_cross_attention:
            self.attention_weights = attention_probs.detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def setup_attention_extraction(unet: nn.Module) -> dict[str, CrossAttentionProcessor]:
    """Setup attention extraction by replacing processors in U-Net.
    
    Args:
        unet: The U-Net model.
        
    Returns:
        Dictionary mapping layer names to their processors.
    """
    processors = {}
    
    # Get all attention processors
    attn_procs = unet.attn_processors  # type: ignore
    
    # Replace cross-attention processors
    new_attn_procs: dict = {}
    for name, proc in attn_procs.items():  # type: ignore
        if 'attn2' in name:  # Cross-attention layers
            new_proc = CrossAttentionProcessor(store_attention=True)
            new_attn_procs[name] = new_proc
            processors[name] = new_proc
        else:
            new_attn_procs[name] = proc
    
    unet.set_attn_processor(new_attn_procs)  # type: ignore
    
    return processors


def get_attention_maps_from_processors(
    processors: dict[str, CrossAttentionProcessor],
    target_size: tuple[int, int] = (64, 64)
) -> Optional[torch.Tensor]:
    """Extract and aggregate attention maps from processors.
    
    Args:
        processors: Dictionary of attention processors.
        target_size: Size to resize maps to.
        
    Returns:
        Aggregated attention map [B, 1, H, W] or None.
    """
    maps = []
    
    for name, proc in processors.items():
        if proc.attention_weights is not None:
            attn = proc.attention_weights  # [B*heads, seq_len, text_len]
            
            # Average over text tokens (keep spatial)
            attn_spatial = attn.mean(dim=-1)  # [B*heads, seq_len]
            
            # Reshape to 2D spatial
            seq_len = attn_spatial.shape[-1]
            h = w = int(seq_len ** 0.5)
            if h * w == seq_len:
                attn_2d = attn_spatial.reshape(-1, 1, h, w)  # [B*heads, 1, h, w]
                
                # Resize to target
                attn_resized = torch.nn.functional.interpolate(
                    attn_2d,
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                )
                maps.append(attn_resized)
    
    if not maps:
        return None
    
    # Stack and average
    stacked = torch.cat(maps, dim=0)  # [N, 1, H, W]
    
    # Average across all maps (heads and layers)
    aggregated = stacked.mean(dim=0, keepdim=True)  # [1, 1, H, W]
    
    return aggregated
