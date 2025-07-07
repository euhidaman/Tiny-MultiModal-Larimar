import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from transformers import get_linear_schedule_with_warmup
from typing import Optional, Dict, Any, Tuple
import numpy as np
import wandb

from .multimodal_vae import TinyMultiModalVAE


class TinyMultiModalLitModel(L.LightningModule):
    """
    Lightning wrapper for Tiny Multimodal VAE
    """

    def __init__(self,
                 # Model parameters
                 latent_size: int = 384,
                 hidden_size: int = 768,
                 memory_size: int = 128,
                 use_memory: bool = True,

                 # Model paths
                 vision_model_name: str = "facebook/dinov2-base",
                 text_model_name: str = "distilbert-base-uncased",
                 decoder_model_name: str = "distilgpt2",

                 # Training parameters
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 10000,
                 beta: float = 0.5,
                 beta_schedule: str = "linear",  # "linear", "cosine", "constant"
                 beta_start: float = 0.0,
                 beta_end: float = 0.5,

                 # Loss weights
                 reconstruction_strength: float = 1.0,
                 memory_strength: float = 1.0,

                 # Optimizer parameters
                 optimizer: str = "adamw",
                 scheduler: str = "cosine",

                 # Freezing parameters
                 freeze_vision: bool = False,
                 freeze_text: bool = False,

                 # Multimodal parameters
                 use_cross_attention: bool = True,
                 num_attention_heads: int = 8,

                 # Generation parameters
                 max_length: int = 512,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,

                 # Other parameters
                 vocab_size: int = 50257,
                 save_samples: bool = True,
                 log_every_n_steps: int = 100):

        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Initialize model
        self.model = TinyMultiModalVAE(
            latent_size=latent_size,
            hidden_size=hidden_size,
            memory_size=memory_size,
            use_memory=use_memory,
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
            decoder_model_name=decoder_model_name,
            vocab_size=vocab_size,
            max_length=max_length,
            beta=beta,
            memory_strength=memory_strength,
            reconstruction_strength=reconstruction_strength,
            use_cross_attention=use_cross_attention,
            num_attention_heads=num_attention_heads,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text
        )

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Generation parameters
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Other parameters
        self.save_samples = save_samples
        self.log_every_n_steps = log_every_n_steps

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def get_beta(self) -> float:
        """Get current beta value based on schedule"""
        if self.beta_schedule == "constant":
            return self.hparams.beta
        elif self.beta_schedule == "linear":
            progress = min(self.global_step / self.warmup_steps, 1.0)
            return self.beta_start + (self.beta_end - self.beta_start) * progress
        elif self.beta_schedule == "cosine":
            progress = min(self.global_step / self.warmup_steps, 1.0)
            return self.beta_start + (self.beta_end - self.beta_start) * (1 - np.cos(np.pi * progress)) / 2
        else:
            return self.hparams.beta

    def forward(self, pixel_values: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                mode: str = "multimodal") -> Dict[str, torch.Tensor]:
        """Forward pass"""
        beta = self.get_beta()
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mode=mode,
            beta=beta
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        # Extract batch data
        pixel_values = batch.get('pixel_values')
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')

        # Determine mode based on available inputs
        if pixel_values is not None and input_ids is not None:
            mode = "multimodal"
        elif pixel_values is not None:
            mode = "vision"
        elif input_ids is not None:
            mode = "text"
        else:
            raise ValueError(
                "At least one of pixel_values or input_ids must be provided")

        # Forward pass
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mode=mode
        )

        # Extract losses
        total_loss = outputs['loss']
        reconstruction_loss = outputs['reconstruction_loss']
        kl_loss = outputs['kl_loss']
        memory_loss = outputs['memory_loss']

        # Log losses
        self.log('train/total_loss', total_loss,
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/reconstruction_loss',
                 reconstruction_loss, on_step=True, on_epoch=True)
        self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True)
        self.log('train/memory_loss', memory_loss, on_step=True, on_epoch=True)
        self.log('train/beta', self.get_beta(), on_step=True, on_epoch=True)

        # Log learning rate
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            self.log('train/lr', scheduler.get_last_lr()
                     [0], on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        # Extract batch data
        pixel_values = batch.get('pixel_values')
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')

        # Determine mode
        if pixel_values is not None and input_ids is not None:
            mode = "multimodal"
        elif pixel_values is not None:
            mode = "vision"
        elif input_ids is not None:
            mode = "text"
        else:
            raise ValueError(
                "At least one of pixel_values or input_ids must be provided")

        # Forward pass
        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mode=mode
        )

        # Extract losses
        total_loss = outputs['loss']
        reconstruction_loss = outputs['reconstruction_loss']
        kl_loss = outputs['kl_loss']
        memory_loss = outputs['memory_loss']

        # Log losses
        self.log('val/total_loss', total_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/reconstruction_loss', reconstruction_loss,
                 on_step=False, on_epoch=True)
        self.log('val/kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val/memory_loss', memory_loss, on_step=False, on_epoch=True)

        return total_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step"""
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        if self.save_samples and self.current_epoch % 5 == 0:
            self.generate_samples()

    def generate_samples(self, num_samples: int = 4) -> None:
        """Generate sample outputs for logging"""
        self.model.eval()

        # Create dummy inputs for generation
        device = next(self.model.parameters()).device

        # Generate from random latent
        batch_size = num_samples
        dummy_input_ids = torch.full(
            (batch_size, 1), self.model.decoder.bos_token_id, device=device)

        try:
            with torch.no_grad():
                # Generate text-only samples
                generated_text = self.model.generate(
                    input_ids=dummy_input_ids,
                    mode="text",
                    max_length=50,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                    num_return_sequences=1
                )

                # Log to wandb if available
                if self.logger is not None and hasattr(self.logger, 'experiment'):
                    samples_table = []
                    for i, sample in enumerate(generated_text):
                        # Decode tokens to text (assuming we have a tokenizer)
                        # This is a placeholder - in practice, you'd use the actual tokenizer
                        text = f"Generated sample {i}: {sample.tolist()}"
                        samples_table.append([i, text])

                    if hasattr(self.logger.experiment, 'log'):
                        self.logger.experiment.log({
                            "generated_samples": wandb.Table(
                                columns=["Sample ID", "Generated Text"],
                                data=samples_table
                            )
                        })

        except Exception as e:
            print(f"Error generating samples: {e}")

        self.model.train()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers"""
        # Prepare optimizer parameters
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        if self.hparams.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=self.learning_rate,
                              eps=1e-8,
                              betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # Initialize scheduler
        if self.hparams.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.max_steps, eta_min=0)
        elif self.hparams.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps
            )
        else:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving checkpoint"""
        # Save additional info
        checkpoint['model_config'] = {
            'latent_size': self.hparams.latent_size,
            'hidden_size': self.hparams.hidden_size,
            'memory_size': self.hparams.memory_size,
            'use_memory': self.hparams.use_memory,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading checkpoint"""
        # Load additional info if available
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print(f"Loaded model config: {model_config}")

    def freeze_vision_encoder(self):
        """Freeze vision encoder"""
        self.model.vision_encoder.freeze_encoder()

    def freeze_text_encoder(self):
        """Freeze text encoder"""
        self.model.text_encoder.freeze_encoder()

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder"""
        self.model.vision_encoder.unfreeze_encoder()

    def unfreeze_text_encoder(self):
        """Unfreeze text encoder"""
        self.model.text_encoder.unfreeze_encoder()

    def get_model_size(self) -> int:
        """Get number of parameters in the model"""
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_model_info(self):
        """Print model information"""
        total_params = self.get_model_size()
        trainable_params = self.get_trainable_params()

        print(f"Model Information:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(
            f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params / total_params:.2%}")

        # Print module sizes
        print(f"\nModule sizes:")
        for name, module in self.model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            trainable_module_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad)
            print(
                f"  {name}: {module_params:,} total, {trainable_module_params:,} trainable")
