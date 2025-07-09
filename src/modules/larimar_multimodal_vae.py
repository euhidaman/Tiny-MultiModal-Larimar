import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import math

from .larimar_text_encoder import LarimarTextEncoder
from .larimar_gpt2_decoder import LarimarGPT2Decoder
from .larimar_memory import TinyLarimarMemory, LarimarMemoryVAE
from .vision_encoder import DiNOv2VisionEncoder


class LarimarMultiModalVAE(nn.Module):
    """
    Multimodal VAE combining Larimar text components with DiNOv2 vision.
    Main model for Tiny-MultiModal-Larimar using authentic Larimar architecture.
    """

    def __init__(self,
                 # Text encoder settings
                 text_model_name: str = "bert-base-uncased",
                 text_latent_size: int = 384,
                 text_freeze_backbone: bool = False,

                 # Vision encoder settings
                 vision_model_name: str = "facebook/dinov2-base",
                 vision_latent_size: int = 384,
                 vision_freeze_backbone: bool = True,

                 # Decoder settings
                 decoder_model_name: str = "gpt2-medium",
                 vocab_size: int = 50257,
                 max_length: int = 512,

                 # Memory settings
                 memory_size: int = 512,
                 use_memory: bool = True,
                 memory_direct_writing: bool = True,
                 observation_noise_std: float = 0.1,

                 # Fusion settings
                 fusion_type: str = "cross_attention",  # "cross_attention", "concat", "add"
                 num_fusion_layers: int = 2,

                 # Loss weights
                 kl_weight: float = 1.0,
                 memory_weight: float = 1.0,
                 reconstruction_weight: float = 1.0):

        super(LarimarMultiModalVAE, self).__init__()

        self.text_latent_size = text_latent_size
        self.vision_latent_size = vision_latent_size
        self.use_memory = use_memory
        self.fusion_type = fusion_type

        # Loss weights
        self.kl_weight = kl_weight
        self.memory_weight = memory_weight
        self.reconstruction_weight = reconstruction_weight

        # Text encoder (Larimar-style BERT)
        self.text_encoder = LarimarTextEncoder(
            model_name=text_model_name,
            latent_size=text_latent_size,
            freeze_backbone=text_freeze_backbone
        )

        # Vision encoder (DiNOv2)
        self.vision_encoder = DiNOv2VisionEncoder(
            model_name=vision_model_name,
            latent_size=vision_latent_size,
            freeze_backbone=vision_freeze_backbone
        )

        # Multimodal fusion
        self.multimodal_latent_size = text_latent_size  # Use text latent size as base

        if fusion_type == "cross_attention":
            self.fusion = MultiModalCrossAttention(
                text_dim=text_latent_size,
                vision_dim=vision_latent_size,
                output_dim=self.multimodal_latent_size,
                num_layers=num_fusion_layers
            )
        elif fusion_type == "concat":
            self.vision_projection = nn.Linear(
                vision_latent_size, text_latent_size)
            self.fusion = nn.Sequential(
                nn.Linear(text_latent_size + text_latent_size,
                          self.multimodal_latent_size),
                nn.LayerNorm(self.multimodal_latent_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_type == "add":
            self.vision_projection = nn.Linear(
                vision_latent_size, text_latent_size)
            self.fusion = nn.Sequential(
                nn.LayerNorm(text_latent_size),
                nn.Linear(text_latent_size, self.multimodal_latent_size)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Decoder (Larimar-style GPT2)
        self.decoder = LarimarGPT2Decoder(
            model_name=decoder_model_name,
            latent_size=self.multimodal_latent_size,
            vocab_size=vocab_size,
            max_length=max_length
        )

        # Memory module (optional)
        if use_memory:
            self.memory = TinyLarimarMemory(
                code_size=self.multimodal_latent_size,
                memory_size=memory_size,
                direct_writing=memory_direct_writing,
                observation_noise_std=observation_noise_std,
                identity_init=True
            )
        else:
            self.memory = None

        # VAE latent projection
        self.latent_projection = nn.Linear(
            self.multimodal_latent_size, self.multimodal_latent_size * 2)

        print(f"Initialized LarimarMultiModalVAE:")
        print(f"  Text: {text_model_name} -> {text_latent_size}D")
        print(f"  Vision: {vision_model_name} -> {vision_latent_size}D")
        print(f"  Decoder: {decoder_model_name}")
        print(f"  Fusion: {fusion_type}, Memory: {use_memory}")
        print(f"  Multimodal latent: {self.multimodal_latent_size}D")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text input"""
        return self.text_encoder.encode_to_latent(input_ids, attention_mask)

    def encode_vision(self, vision_embedding: torch.Tensor) -> torch.Tensor:
        """Encode vision input (pre-computed DiNOv2 embeddings)"""
        return self.vision_encoder(vision_embedding)

    def fuse_modalities(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """Fuse text and vision features"""
        if self.fusion_type == "cross_attention":
            fused_features = self.fusion(text_features, vision_features)
        elif self.fusion_type == "concat":
            vision_projected = self.vision_projection(vision_features)
            concatenated = torch.cat([text_features, vision_projected], dim=-1)
            fused_features = self.fusion(concatenated)
        elif self.fusion_type == "add":
            vision_projected = self.vision_projection(vision_features)
            added = text_features + vision_projected
            fused_features = self.fusion(added)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return fused_features

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                vision_embedding: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                use_memory: Optional[bool] = None,
                return_latents: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal VAE

        Args:
            input_ids: [batch_size, seq_len] - Tokenized text
            attention_mask: [batch_size, seq_len] - Attention mask
            vision_embedding: [batch_size, vision_dim] - Pre-computed DiNOv2 features
            labels: [batch_size, seq_len] - Target tokens for reconstruction
            use_memory: Whether to use memory (overrides instance setting)
            return_latents: Whether to return latent representations

        Returns:
            Dictionary containing losses, logits, and optional latents
        """
        batch_size = input_ids.size(0)
        use_memory = use_memory if use_memory is not None else self.use_memory

        # Encode text
        text_latent_z, text_mu, text_logvar = self.text_encoder.encode_to_latent(
            input_ids, attention_mask)

        # Encode vision (if provided)
        if vision_embedding is not None:
            vision_features = self.encode_vision(vision_embedding)

            # Fuse modalities
            multimodal_features = self.fuse_modalities(
                text_mu, vision_features)
        else:
            # Text-only mode
            multimodal_features = text_mu

        # Project to VAE latent space
        latent_params = self.latent_projection(multimodal_features)
        multimodal_mu, multimodal_logvar = latent_params.chunk(2, -1)
        multimodal_z = self.reparameterize(multimodal_mu, multimodal_logvar)

        # Memory interaction
        memory_kl = torch.tensor(0.0, device=multimodal_z.device)
        if use_memory and self.memory is not None:
            # Add episode dimension for memory
            multimodal_z_mem = multimodal_z.unsqueeze(
                0)  # [1, batch_size, latent_size]

            # Write to memory and read back
            memory_state, memory_kl = self.memory.write_to_memory(
                multimodal_z_mem)
            retrieved_z, attention_weights = self.memory.read_from_memory(
                multimodal_z_mem, memory_state)

            # Use retrieved representation
            multimodal_z = retrieved_z.squeeze(0)  # Remove episode dimension

        # Decode
        decoder_output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            latent_conditioning=multimodal_z
        )

        # Compute losses
        reconstruction_loss = decoder_output.get(
            'loss', torch.tensor(0.0, device=multimodal_z.device))

        # Text KL divergence
        text_kl = -0.5 * \
            torch.sum(1 + text_logvar - text_mu.pow(2) -
                      text_logvar.exp()) / batch_size

        # Multimodal KL divergence
        multimodal_kl = -0.5 * \
            torch.sum(1 + multimodal_logvar - multimodal_mu.pow(2) -
                      multimodal_logvar.exp()) / batch_size

        total_kl = text_kl + multimodal_kl

        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                      self.kl_weight * total_kl +
                      self.memory_weight * memory_kl)

        result = {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'text_kl_loss': text_kl,
            'multimodal_kl_loss': multimodal_kl,
            'total_kl_loss': total_kl,
            'memory_kl_loss': memory_kl,
            'logits': decoder_output.get('logits'),
        }

        if return_latents:
            result.update({
                'text_latent_z': text_latent_z,
                'text_mu': text_mu,
                'text_logvar': text_logvar,
                'multimodal_z': multimodal_z,
                'multimodal_mu': multimodal_mu,
                'multimodal_logvar': multimodal_logvar,
            })

        return result

    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 vision_embedding: Optional[torch.Tensor] = None,
                 max_length: int = 50,
                 **generation_kwargs) -> torch.Tensor:
        """
        Generate text conditioned on input text and optional vision
        """
        with torch.no_grad():
            # Encode inputs
            text_latent_z, text_mu, text_logvar = self.text_encoder.encode_to_latent(
                input_ids, attention_mask)

            if vision_embedding is not None:
                vision_features = self.encode_vision(vision_embedding)
                multimodal_features = self.fuse_modalities(
                    text_mu, vision_features)
            else:
                multimodal_features = text_mu

            # Project to multimodal latent space (use mean for generation)
            latent_params = self.latent_projection(multimodal_features)
            multimodal_mu, multimodal_logvar = latent_params.chunk(2, -1)

            # Memory interaction (if enabled)
            conditioning_latent = multimodal_mu
            if self.use_memory and self.memory is not None:
                multimodal_z_mem = multimodal_mu.unsqueeze(0)
                memory_state, _ = self.memory.write_to_memory(multimodal_z_mem)
                retrieved_z, _ = self.memory.read_from_memory(
                    multimodal_z_mem, memory_state, deterministic=True)
                conditioning_latent = retrieved_z.squeeze(0)

            # Generate
            generated = self.decoder.generate(
                input_ids=input_ids,
                latent_conditioning=conditioning_latent,
                max_length=max_length,
                **generation_kwargs
            )

        return generated


class MultiModalCrossAttention(nn.Module):
    """
    Cross-attention module for fusing text and vision features
    """

    def __init__(self, text_dim: int, vision_dim: int, output_dim: int, num_layers: int = 2):
        super(MultiModalCrossAttention, self).__init__()

        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.output_dim = output_dim

        # Project to common dimension
        common_dim = min(text_dim, vision_dim)
        self.text_projection = nn.Linear(text_dim, common_dim)
        self.vision_projection = nn.Linear(vision_dim, common_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(common_dim, num_heads=8, batch_first=True)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(common_dim) for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(common_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: [batch_size, text_dim]
            vision_features: [batch_size, vision_dim]
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # Project to common dimension
        text_projected = self.text_projection(
            text_features).unsqueeze(1)  # [batch_size, 1, common_dim]
        vision_projected = self.vision_projection(
            vision_features).unsqueeze(1)  # [batch_size, 1, common_dim]

        # Cross-attention
        attended_text = text_projected
        for layer, norm in zip(self.cross_attention_layers, self.layer_norms):
            # Text attends to vision
            attn_output, _ = layer(
                attended_text, vision_projected, vision_projected)
            attended_text = norm(attended_text + attn_output)

        # Output
        fused_features = self.output_projection(
            attended_text.squeeze(1))  # [batch_size, output_dim]

        return fused_features


class LarimarMultiModalConfig:
    """Configuration class for LarimarMultiModalVAE"""

    def __init__(self,
                 # Model architecture
                 text_model_name: str = "bert-base-uncased",
                 vision_model_name: str = "facebook/dinov2-base",
                 decoder_model_name: str = "gpt2-medium",

                 # Latent dimensions
                 text_latent_size: int = 384,
                 vision_latent_size: int = 384,

                 # Memory settings
                 memory_size: int = 512,
                 use_memory: bool = True,

                 # Training settings
                 max_length: int = 512,
                 vocab_size: int = 50257,

                 # Loss weights
                 kl_weight: float = 1.0,
                 memory_weight: float = 1.0,
                 reconstruction_weight: float = 1.0):

        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        self.decoder_model_name = decoder_model_name
        self.text_latent_size = text_latent_size
        self.vision_latent_size = vision_latent_size
        self.memory_size = memory_size
        self.use_memory = use_memory
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.kl_weight = kl_weight
        self.memory_weight = memory_weight
        self.reconstruction_weight = reconstruction_weight

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
