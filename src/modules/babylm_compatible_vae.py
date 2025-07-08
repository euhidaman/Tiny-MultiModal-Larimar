from multimodal_vae import TinyMultiModalVAE
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import sys
import os

# Add the parent directory to the path to import the original modules
sys.path.append(os.path.join(os.path.dirname(__file__)))


class BabyLMCompatibleVAE(TinyMultiModalVAE):
    """
    Modified TinyMultiModalVAE that can accept pre-computed vision embeddings
    from DiNOv2 instead of raw pixel values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add a projection layer to handle pre-computed embeddings
        # DiNOv2 base outputs 768-dim features, we need to match our latent size
        self.precomputed_vision_proj = nn.Linear(768, self.latent_size)

    def forward(self,
                pixel_values: Optional[torch.Tensor] = None,
                vision_embedding: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mode: str = "multimodal",
                beta: Optional[float] = None,
                num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Modified forward pass that can handle pre-computed vision embeddings

        Args:
            pixel_values: Raw images [batch_size, channels, height, width] (optional)
            vision_embedding: Pre-computed DiNOv2 features [batch_size, 768] (optional)
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
            mode: "vision", "text", or "multimodal"
            beta: KL regularization strength
            num_samples: Number of latent samples
        """

        # Handle vision input - either raw pixels or pre-computed embeddings
        if vision_embedding is not None:
            # Use pre-computed embeddings directly
            batch_size = vision_embedding.size(0)

            # Project to latent size and expand to sequence format
            vision_features = self.precomputed_vision_proj(
                vision_embedding)  # [batch, latent_size]
            vision_features = vision_features.unsqueeze(
                1)  # [batch, 1, latent_size]

            # Skip the vision encoder and use projected features directly
            vision_latent_mean = vision_features
            vision_latent_logvar = torch.zeros_like(vision_features)

        elif pixel_values is not None:
            # Use original vision processing pipeline
            batch_size = pixel_values.size(0)

            # Encode vision
            vision_outputs = self.vision_encoder(pixel_values)
            # [batch, seq_len, hidden]
            vision_features = vision_outputs['features']

            # Vision latent encoding
            vision_latent_mean = self.vision_mu(vision_features)
            vision_latent_logvar = self.vision_logvar(vision_features)

        else:
            vision_latent_mean = None
            vision_latent_logvar = None
            batch_size = input_ids.size(0) if input_ids is not None else 1

        # Continue with text processing (unchanged)
        if input_ids is not None:
            # Encode text
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # [batch, seq_len, hidden]
            text_features = text_outputs['features']

            # Text latent encoding
            text_latent_mean = self.text_mu(text_features)
            text_latent_logvar = self.text_logvar(text_features)
        else:
            text_latent_mean = None
            text_latent_logvar = None

        # Continue with the rest of the original forward logic
        # (latent sampling, fusion, memory, decoding, loss calculation)

        # For now, let's use a simplified version that focuses on the key components
        outputs = {}

        # Sample from latent distributions
        if vision_latent_mean is not None:
            vision_latent = self.reparameterize(
                vision_latent_mean, vision_latent_logvar)
        else:
            vision_latent = None

        if text_latent_mean is not None:
            text_latent = self.reparameterize(
                text_latent_mean, text_latent_logvar)
        else:
            text_latent = None

        # Multimodal fusion
        if mode == "multimodal" and vision_latent is not None and text_latent is not None:
            if self.use_cross_attention:
                fused_latent = self.fusion_layer(
                    vision_features=vision_latent,
                    text_features=text_latent
                )
            else:
                fused_latent = torch.cat([vision_latent, text_latent], dim=-1)
        elif mode == "vision" and vision_latent is not None:
            fused_latent = vision_latent
        elif mode == "text" and text_latent is not None:
            fused_latent = text_latent
        else:
            raise ValueError(f"Invalid mode {mode} or missing required inputs")

        # Memory interaction
        if self.use_memory:
            memory_outputs = self.memory(fused_latent)
            memory_latent = memory_outputs['output']
            memory_loss = memory_outputs.get('memory_loss', 0.0)
        else:
            memory_latent = fused_latent
            memory_loss = 0.0

        # Decode
        if labels is not None:
            decoder_outputs = self.decoder(
                latent_features=memory_latent,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            reconstruction_loss = decoder_outputs['loss']
            logits = decoder_outputs['logits']
        else:
            reconstruction_loss = 0.0
            logits = None

        # KL divergence
        kl_div = 0.0
        if vision_latent_mean is not None:
            kl_div += self.kl_divergence(vision_latent_mean,
                                         vision_latent_logvar)
        if text_latent_mean is not None:
            kl_div += self.kl_divergence(text_latent_mean, text_latent_logvar)

        # Total loss
        if beta is None:
            beta = self.beta

        total_loss = (self.reconstruction_strength * reconstruction_loss +
                      beta * kl_div +
                      self.memory_strength * memory_loss)

        outputs.update({
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_div': kl_div,
            'memory_loss': memory_loss,
            'logits': logits,
            'latent': memory_latent
        })

        return outputs
