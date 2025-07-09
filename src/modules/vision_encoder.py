import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config
from typing import Optional, Tuple
import math


class DiNOv2VisionEncoder(nn.Module):
    """
    Vision encoder using DiNOv2 ViT-Base model for Tiny-MultiModal-Larimar.
    Handles image input and provides visual features for multimodal processing.
    """

    def __init__(self,
                 model_name: str = "facebook/dinov2-base",
                 latent_size: int = 384,
                 freeze_backbone: bool = False,
                 add_projection: bool = True):
        super(DiNOv2VisionEncoder, self).__init__()

        self.latent_size = latent_size
        self.freeze_backbone = freeze_backbone

        # Load DiNOv2 model
        self.vision_model = Dinov2Model.from_pretrained(model_name)
        self.vision_config = self.vision_model.config

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # DiNOv2 outputs 768-dimensional features
        self.hidden_size = self.vision_config.hidden_size  # 768

        # Optional projection layer to match latent size
        if add_projection and self.hidden_size != latent_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, latent_size),
                nn.LayerNorm(latent_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None

        # Positional encoding for spatial features
        # 256 patches for 224x224 image with 14x14 patches
        self.max_patches = (224 // 14) ** 2
        self.spatial_pos_emb = nn.Parameter(
            torch.randn(1, self.max_patches, latent_size))

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, vision_input: torch.Tensor,
                output_hidden_states: bool = False,
                return_spatial_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through vision encoder
        Args:
            vision_input: Either:
                - [batch_size, channels, height, width] for raw pixel values
                - [batch_size, hidden_size] for pre-computed DiNOv2 embeddings
            output_hidden_states: Whether to return all hidden states
            return_spatial_features: Whether to return spatial patch features
        Returns:
            pooled_output: [batch_size, latent_size] - Global image representation
            spatial_features: [batch_size, num_patches, latent_size] - Spatial patch features (optional)
        """
        batch_size = vision_input.size(0)

        # Check if input is pre-computed embeddings or raw pixel values
        if vision_input.dim() == 2:
            # Pre-computed DiNOv2 embeddings: [batch_size, hidden_size]
            # Apply projection directly to the embeddings
            if self.projection is not None:
                pooled_output = self.projection(vision_input)
            else:
                pooled_output = vision_input
            
            # Apply layer norm
            pooled_output = self.layer_norm(pooled_output)
            
            # No spatial features for pre-computed embeddings
            spatial_features = None
            
        elif vision_input.dim() == 4:
            # Raw pixel values: [batch_size, channels, height, width]
            # Pass through DiNOv2
            outputs = self.vision_model(
                pixel_values=vision_input, output_hidden_states=output_hidden_states)

            # Get last hidden state: [batch_size, num_patches + 1, hidden_size]
            # Note: DiNOv2 doesn't have a CLS token, so we use the first token as global representation
            last_hidden_state = outputs.last_hidden_state

            # Apply projection if needed
            if self.projection is not None:
                last_hidden_state = self.projection(last_hidden_state)

            # Extract global representation (mean pooling over all patches)
            pooled_output = last_hidden_state.mean(
                dim=1)  # [batch_size, latent_size]

            # Apply layer norm
            pooled_output = self.layer_norm(pooled_output)

            if return_spatial_features:
                # Add positional encoding to spatial features
                num_patches = last_hidden_state.size(1)
                if num_patches <= self.max_patches:
                    spatial_features = last_hidden_state + \
                        self.spatial_pos_emb[:, :num_patches, :]
                else:
                    # Handle case where image has more patches than expected
                    spatial_features = last_hidden_state

                spatial_features = self.layer_norm(spatial_features)
            else:
                spatial_features = None
                
        else:
            raise ValueError(f"Invalid vision_input shape: {vision_input.shape}. Expected either [batch_size, channels, height, width] or [batch_size, hidden_size]")

        if return_spatial_features:
            return pooled_output, spatial_features
        else:
            return pooled_output

    def get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings for attention visualization
        Args:
            pixel_values: [batch_size, channels, height, width]
        Returns:
            patch_embeddings: [batch_size, num_patches, latent_size]
        """
        with torch.no_grad():
            _, spatial_features = self.forward(
                pixel_values, return_spatial_features=True)
        return spatial_features

    def freeze_encoder(self):
        """Freeze the vision encoder backbone"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.freeze_backbone = True

    def unfreeze_encoder(self):
        """Unfreeze the vision encoder backbone"""
        for param in self.vision_model.parameters():
            param.requires_grad = True
        self.freeze_backbone = False


class VisionTextProjector(nn.Module):
    """
    Projection layer to align vision and text representations
    """

    def __init__(self, vision_dim: int, text_dim: int, latent_dim: int):
        super(VisionTextProjector, self).__init__()

        self.vision_proj = nn.Linear(vision_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project vision and text features to common latent space
        Args:
            vision_features: [batch_size, vision_dim]
            text_features: [batch_size, text_dim]
        Returns:
            projected_vision: [batch_size, latent_dim]
            projected_text: [batch_size, latent_dim]
        """
        projected_vision = self.layer_norm(self.vision_proj(vision_features))
        projected_text = self.layer_norm(self.text_proj(text_features))

        return projected_vision, projected_text


class MultiModalFusion(nn.Module):
    """
    Cross-attention fusion module for vision and text features
    """

    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiModalFusion, self).__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        # Cross-attention layers
        self.vision_to_text_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.text_to_vision_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward networks
        self.vision_ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.text_ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # Layer norms
        self.vision_norm1 = nn.LayerNorm(latent_dim)
        self.vision_norm2 = nn.LayerNorm(latent_dim)
        self.text_norm1 = nn.LayerNorm(latent_dim)
        self.text_norm2 = nn.LayerNorm(latent_dim)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse vision and text features using cross-attention
        Args:
            vision_features: [batch_size, vision_seq_len, latent_dim]
            text_features: [batch_size, text_seq_len, latent_dim]
        Returns:
            fused_vision: [batch_size, vision_seq_len, latent_dim]
            fused_text: [batch_size, text_seq_len, latent_dim]
        """
        # Cross-attention: vision attends to text
        vision_attended, _ = self.vision_to_text_attn(
            query=vision_features,
            key=text_features,
            value=text_features
        )
        vision_features = self.vision_norm1(vision_features + vision_attended)

        # Cross-attention: text attends to vision
        text_attended, _ = self.text_to_vision_attn(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        text_features = self.text_norm1(text_features + text_attended)

        # Feed-forward
        vision_ff_out = self.vision_ff(vision_features)
        vision_features = self.vision_norm2(vision_features + vision_ff_out)

        text_ff_out = self.text_ff(text_features)
        text_features = self.text_norm2(text_features + text_ff_out)

        return vision_features, text_features
