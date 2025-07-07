import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .memory import TinyMemory
from .vision_encoder import DiNOv2VisionEncoder, MultiModalFusion
from .text_encoder import DistilBERTTextEncoder
from .decoder import DistilGPT2Decoder


class TinyMultiModalVAE(nn.Module):
    """
    Tiny Multimodal VAE with episodic memory for vision-language understanding.
    Based on Larimar architecture but smaller and multimodal.
    """

    def __init__(self,
                 # Model dimensions
                 latent_size: int = 384,
                 hidden_size: int = 768,

                 # Vision encoder
                 vision_model_name: str = "facebook/dinov2-base",
                 freeze_vision: bool = False,

                 # Text encoder
                 text_model_name: str = "distilbert-base-uncased",
                 freeze_text: bool = False,

                 # Decoder
                 decoder_model_name: str = "distilgpt2",
                 vocab_size: int = 50257,
                 max_length: int = 512,

                 # Memory
                 memory_size: int = 128,
                 use_memory: bool = True,
                 direct_writing: bool = True,
                 ordering: bool = False,

                 # Training
                 beta: float = 0.5,
                 memory_strength: float = 1.0,
                 reconstruction_strength: float = 1.0,

                 # Multimodal fusion
                 use_cross_attention: bool = True,
                 num_attention_heads: int = 8,

                 # Device
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        super(TinyMultiModalVAE, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.use_memory = use_memory
        self.beta = beta
        self.memory_strength = memory_strength
        self.reconstruction_strength = reconstruction_strength
        self.use_cross_attention = use_cross_attention
        self.device = device

        # Vision encoder
        self.vision_encoder = DiNOv2VisionEncoder(
            model_name=vision_model_name,
            latent_size=latent_size,
            freeze_backbone=freeze_vision,
            add_projection=True
        )

        # Text encoder
        self.text_encoder = DistilBERTTextEncoder(
            model_name=text_model_name,
            latent_size=latent_size,
            freeze_backbone=freeze_text,
            add_projection=True,
            max_length=max_length
        )

        # Multimodal fusion
        if use_cross_attention:
            self.multimodal_fusion = MultiModalFusion(
                latent_dim=latent_size,
                num_heads=num_attention_heads,
                dropout=0.1
            )

        # Latent space projections
        self.vision_to_latent = nn.Linear(
            latent_size, latent_size * 2)  # mean and logvar
        self.text_to_latent = nn.Linear(
            latent_size, latent_size * 2)    # mean and logvar
        self.multimodal_to_latent = nn.Linear(
            latent_size * 2, latent_size * 2)  # combined latent

        # Memory module
        if use_memory:
            self.memory = TinyMemory(
                code_size=latent_size,
                memory_size=memory_size,
                direct_writing=direct_writing,
                ordering=ordering,
                device=0 if device == "cuda" else -1
            )

        # Decoder
        self.decoder = DistilGPT2Decoder(
            model_name=decoder_model_name,
            latent_size=latent_size,
            vocab_size=vocab_size,
            max_length=max_length,
            add_latent_conditioning=True
        )

        # Prior distribution
        self.register_buffer('prior_mean', torch.zeros(latent_size))
        self.register_buffer('prior_logvar', torch.zeros(latent_size))

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_vision(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode vision input to latent space
        Args:
            pixel_values: [batch_size, channels, height, width]
        Returns:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        vision_features, _ = self.vision_encoder(pixel_values)
        mean_logvar = self.vision_to_latent(vision_features)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text input to latent space
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        text_features, _ = self.text_encoder(input_ids, attention_mask)
        mean_logvar = self.text_to_latent(text_features)
        mean, logvar = mean_logvar.chunk(2, dim=-1)
        return mean, logvar

    def encode_multimodal(self, pixel_values: torch.Tensor,
                          input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multimodal input to latent space
        Args:
            pixel_values: [batch_size, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        # Get individual modality features
        vision_features, vision_spatial = self.vision_encoder(
            pixel_values, return_spatial_features=True)
        text_features, text_sequence = self.text_encoder(
            input_ids, attention_mask, return_sequence_features=True)

        # Apply cross-attention fusion if enabled
        if self.use_cross_attention and vision_spatial is not None and text_sequence is not None:
            # Reshape for cross-attention
            vision_spatial = vision_spatial.unsqueeze(
                1) if vision_spatial.dim() == 2 else vision_spatial
            text_sequence = text_sequence if text_sequence.dim(
            ) == 3 else text_sequence.unsqueeze(1)

            # Apply fusion
            fused_vision, fused_text = self.multimodal_fusion(
                vision_spatial, text_sequence)

            # Pool fused features
            vision_features = fused_vision.mean(dim=1)
            text_features = fused_text.mean(dim=1)

        # Combine features
        combined_features = torch.cat([vision_features, text_features], dim=-1)

        # Project to latent space
        mean_logvar = self.multimodal_to_latent(combined_features)
        mean, logvar = mean_logvar.chunk(2, dim=-1)

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor,
                       num_samples: int = 1) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        Args:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
            num_samples: Number of samples
        Returns:
            z: [batch_size, num_samples, latent_size]
        """
        batch_size, latent_size = mean.size()
        std = torch.exp(0.5 * logvar)

        if num_samples == 1:
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z.unsqueeze(1)
        else:
            eps = torch.randn(batch_size, num_samples,
                              latent_size, device=mean.device)
            mean_expanded = mean.unsqueeze(1).expand(
                batch_size, num_samples, latent_size)
            std_expanded = std.unsqueeze(1).expand(
                batch_size, num_samples, latent_size)
            z = mean_expanded + eps * std_expanded
            return z

    def decode(self, z: torch.Tensor,
               input_ids: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to text
        Args:
            z: [batch_size, latent_size] or [batch_size, num_samples, latent_size]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        Returns:
            outputs: Dictionary with decoder outputs
        """
        # Handle multiple samples
        if z.dim() == 3:
            z = z.mean(dim=1)  # Average over samples

        # Decode with latent conditioning
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            latent_conditioning=z
        )

        return outputs

    def forward(self, pixel_values: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mode: str = "multimodal",
                beta: Optional[float] = None,
                num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        Args:
            pixel_values: [batch_size, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
            mode: "vision", "text", or "multimodal"
            beta: KL regularization strength
            num_samples: Number of latent samples
        Returns:
            outputs: Dictionary with loss and other outputs
        """
        batch_size = input_ids.size(
            0) if input_ids is not None else pixel_values.size(0)

        if beta is None:
            beta = self.beta

        # Encode to latent space
        if mode == "vision" and pixel_values is not None:
            mean, logvar = self.encode_vision(pixel_values)
        elif mode == "text" and input_ids is not None:
            mean, logvar = self.encode_text(input_ids, attention_mask)
        elif mode == "multimodal" and pixel_values is not None and input_ids is not None:
            mean, logvar = self.encode_multimodal(
                pixel_values, input_ids, attention_mask)
        else:
            raise ValueError(f"Invalid mode '{mode}' or missing inputs")

        # Reparameterize
        z = self.reparameterize(mean, logvar, num_samples)

        # Memory operations
        memory_loss = 0.0
        memory_state = None
        if self.use_memory:
            # Use mean of samples for memory
            z_for_memory = z.mean(dim=1) if z.dim() == 3 else z
            z_retrieved, memory_state, memory_kl = self.memory(z_for_memory)
            memory_loss = memory_kl.mean() * self.memory_strength

            # Use retrieved latent for decoding
            z_for_decode = z_retrieved
        else:
            z_for_decode = z.mean(dim=1) if z.dim() == 3 else z

        # Decode
        decoder_outputs = self.decode(
            z_for_decode, input_ids, attention_mask, labels)

        # Compute losses
        reconstruction_loss = 0.0
        if labels is not None:
            reconstruction_loss = decoder_outputs.loss * self.reconstruction_strength

        # KL divergence
        kl_loss = self.compute_kl_loss(mean, logvar)

        # Total loss
        total_loss = reconstruction_loss + beta * kl_loss + memory_loss

        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'memory_loss': memory_loss,
            'logits': decoder_outputs.logits,
            'mean': mean,
            'logvar': logvar,
            'z': z,
            'memory_state': memory_state
        }

    def compute_kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior
        Args:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        Returns:
            kl_loss: Scalar KL loss
        """
        # KL divergence with standard normal prior
        kl_loss = -0.5 * torch.sum(1 + logvar -
                                   mean.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def generate(self, pixel_values: Optional[torch.Tensor] = None,
                 input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 mode: str = "multimodal",
                 max_length: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 do_sample: bool = True,
                 num_return_sequences: int = 1) -> List[torch.Tensor]:
        """
        Generate text conditioned on vision and/or text input
        Args:
            pixel_values: [batch_size, channels, height, width]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            mode: "vision", "text", or "multimodal"
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
        Returns:
            generated_sequences: List of generated sequences
        """
        self.eval()

        with torch.no_grad():
            # Encode to latent space
            if mode == "vision" and pixel_values is not None:
                mean, logvar = self.encode_vision(pixel_values)
            elif mode == "text" and input_ids is not None:
                mean, logvar = self.encode_text(input_ids, attention_mask)
            elif mode == "multimodal" and pixel_values is not None and input_ids is not None:
                mean, logvar = self.encode_multimodal(
                    pixel_values, input_ids, attention_mask)
            else:
                raise ValueError(f"Invalid mode '{mode}' or missing inputs")

            # Sample from latent
            z = self.reparameterize(mean, logvar, num_samples=1).squeeze(1)

            # Memory operations
            if self.use_memory:
                z_retrieved, _, _ = self.memory(z)
                z_for_decode = z_retrieved
            else:
                z_for_decode = z

            # Generate
            if input_ids is None:
                # Start with BOS token
                input_ids = torch.full(
                    (z.size(0), 1), self.decoder.bos_token_id, device=z.device)

            generated_sequences = self.decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_conditioning=z_for_decode,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences
            )

        return generated_sequences

    def save_pretrained(self, save_directory: str):
        """Save model to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save state dict
        torch.save(self.state_dict(), os.path.join(
            save_directory, 'model.bin'))

        # Save config
        config = {
            'latent_size': self.latent_size,
            'hidden_size': self.hidden_size,
            'memory_size': self.memory_size,
            'use_memory': self.use_memory,
            'beta': self.beta,
            'memory_strength': self.memory_strength,
            'reconstruction_strength': self.reconstruction_strength,
            'use_cross_attention': self.use_cross_attention,
        }

        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from directory"""
        import os
        import json

        # Load config
        config_path = os.path.join(load_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update config with kwargs
        config.update(kwargs)

        # Create model
        model = cls(**config)

        # Load state dict
        state_dict_path = os.path.join(load_directory, 'model.bin')
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict)

        return model
