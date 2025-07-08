import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .babylm_compatible_vae import BabyLMCompatibleVAE


class BabyLMLightningModel(L.LightningModule):
    """
    Lightning wrapper for BabyLM compatible multimodal VAE that works with
    pre-computed DiNOv2 embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Extract model config
        model_config = config.get('model', {})

        # Initialize the compatible VAE
        self.model = BabyLMCompatibleVAE(
            latent_size=model_config.get('latent_size', 512),
            hidden_size=model_config.get('hidden_size', 1024),
            memory_size=model_config.get('memory_size', 256),
            use_memory=model_config.get('use_memory', True),
            vision_model_name=model_config.get(
                'vision_model_name', 'facebook/dinov2-base'),
            text_model_name=model_config.get(
                'text_model_name', 'microsoft/deberta-v3-base'),
            decoder_model_name=model_config.get(
                'decoder_model_name', 'gpt2-medium'),
            beta=model_config.get('beta', 0.5),
            memory_strength=model_config.get('memory_strength', 1.0),
            reconstruction_strength=model_config.get(
                'reconstruction_strength', 1.0),
            use_cross_attention=model_config.get('use_cross_attention', True),
            num_attention_heads=model_config.get('num_attention_heads', 8),
            freeze_vision=model_config.get('freeze_vision', False),
            freeze_text=model_config.get('freeze_text', False),
        )

        # Training parameters
        self.learning_rate = model_config.get('learning_rate', 3e-5)
        self.weight_decay = model_config.get('weight_decay', 0.01)
        self.warmup_steps = model_config.get('warmup_steps', 2000)
        self.max_steps = model_config.get('max_steps', 15000)

        # Beta scheduling
        self.beta_schedule = model_config.get('beta_schedule', 'linear')
        self.beta_start = model_config.get('beta_start', 0.0)
        self.beta_end = model_config.get('beta_end', 0.5)

    def forward(self, batch: Dict[str, torch.Tensor], mode: str = "multimodal"):
        """Forward pass"""
        return self.model(
            vision_embedding=batch.get('vision_embedding'),
            input_ids=batch.get('input_ids'),
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels'),
            mode=mode,
            beta=self._get_current_beta()
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        # Determine mode based on available inputs
        if batch.get('vision_embedding') is not None and batch.get('input_ids') is not None:
            mode = "multimodal"
        elif batch.get('vision_embedding') is not None:
            mode = "vision"
        elif batch.get('input_ids') is not None:
            mode = "text"
        else:
            raise ValueError(
                "At least one of vision_embedding or input_ids must be provided")

        # Forward pass
        outputs = self.forward(batch, mode=mode)

        # Log metrics
        self.log('train_loss', outputs['loss'], prog_bar=True)
        self.log('train_reconstruction_loss', outputs['reconstruction_loss'])
        self.log('train_kl_div', outputs['kl_div'])
        self.log('train_memory_loss', outputs['memory_loss'])
        self.log('beta', self._get_current_beta())

        return outputs['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        # Determine mode
        if batch.get('vision_embedding') is not None and batch.get('input_ids') is not None:
            mode = "multimodal"
        elif batch.get('vision_embedding') is not None:
            mode = "vision"
        elif batch.get('input_ids') is not None:
            mode = "text"
        else:
            raise ValueError(
                "At least one of vision_embedding or input_ids must be provided")

        # Forward pass
        outputs = self.forward(batch, mode=mode)

        # Log metrics
        self.log('val_loss', outputs['loss'], prog_bar=True)
        self.log('val_reconstruction_loss', outputs['reconstruction_loss'])
        self.log('val_kl_div', outputs['kl_div'])
        self.log('val_memory_loss', outputs['memory_loss'])

        return outputs['loss']

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def _get_current_beta(self) -> float:
        """Get current beta value based on schedule"""
        if self.beta_schedule == "constant":
            return self.beta_end
        elif self.beta_schedule == "linear":
            progress = min(1.0, self.global_step / self.max_steps)
            return self.beta_start + (self.beta_end - self.beta_start) * progress
        elif self.beta_schedule == "cosine":
            progress = min(1.0, self.global_step / self.max_steps)
            return self.beta_start + (self.beta_end - self.beta_start) * (1 - np.cos(np.pi * progress)) / 2
        else:
            return self.beta_end
