import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, Optional
import math

from .larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
from .babylm_data import BabyLMMultiModalDataModule


class LarimarBabyLMLightningModel(pl.LightningModule):
    """
    Lightning module for training the Larimar-style Tiny-MultiModal model on BabyLM data.
    This combines the authentic Larimar architecture with DiNOv2 vision encoding.
    """

    def __init__(self,
                 # Model configuration
                 config: Optional[LarimarMultiModalConfig] = None,

                 # Training hyperparameters
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_epochs: int = 10,

                 # Loss scheduling
                 kl_warmup_steps: int = 5000,
                 memory_warmup_steps: int = 3000,

                 # Generation settings
                 max_generation_length: int = 50,

                 # Optimizer settings
                 optimizer_type: str = "adamw",  # "adamw", "adam"
                 scheduler_type: str = "linear",  # "linear", "cosine"

                 # Logging
                 log_every_n_steps: int = 100,
                 generate_every_n_steps: int = 500):

        super(LarimarBabyLMLightningModel, self).__init__()

        # Store hyperparameters
        self.save_hyperparameters(ignore=['config'])

        # Model configuration
        if config is None:
            config = LarimarMultiModalConfig()
        self.config = config

        # Initialize model
        self.model = LarimarMultiModalVAE(
            text_model_name=config.text_model_name,
            vision_model_name=config.vision_model_name,
            decoder_model_name=config.decoder_model_name,
            text_latent_size=config.text_latent_size,
            vision_latent_size=config.vision_latent_size,
            memory_size=config.memory_size,
            use_memory=config.use_memory,
            max_length=config.max_length,
            vocab_size=config.vocab_size,
            kl_weight=config.kl_weight,
            memory_weight=config.memory_weight,
            reconstruction_weight=config.reconstruction_weight
        )

        # Tokenizer for evaluation
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'

        # Add special tokens to decoder if needed
        special_tokens = {'pad_token': '<PAD>',
                          'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        if num_added_tokens > 0:
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))
            print(f"Added {num_added_tokens} special tokens to decoder")

        # Loss weights scheduling
        self.kl_warmup_steps = kl_warmup_steps
        self.memory_warmup_steps = memory_warmup_steps

        # Training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

        print(f"Initialized LarimarBabyLMLightningModel with config:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  KL warmup: {kl_warmup_steps} steps")
        print(f"  Memory warmup: {memory_warmup_steps} steps")

    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """Get loss weights based on training step (warmup scheduling)"""

        # KL annealing
        if step < self.kl_warmup_steps:
            kl_weight = step / self.kl_warmup_steps * self.config.kl_weight
        else:
            kl_weight = self.config.kl_weight

        # Memory annealing
        if step < self.memory_warmup_steps:
            memory_weight = step / self.memory_warmup_steps * self.config.memory_weight
        else:
            memory_weight = self.config.memory_weight

        return {
            'kl_weight': kl_weight,
            'memory_weight': memory_weight,
            'reconstruction_weight': self.config.reconstruction_weight
        }

    def forward(self, batch: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            vision_embedding=batch.get('vision_embedding'),
            labels=batch.get('labels'),
            **kwargs
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        step = self.global_step

        # Get current loss weights
        loss_weights = self.get_loss_weights(step)

        # Temporarily update model loss weights
        original_kl_weight = self.model.kl_weight
        original_memory_weight = self.model.memory_weight

        self.model.kl_weight = loss_weights['kl_weight']
        self.model.memory_weight = loss_weights['memory_weight']

        # Forward pass
        outputs = self.forward(batch)

        # Restore original weights
        self.model.kl_weight = original_kl_weight
        self.model.memory_weight = original_memory_weight

        # Manual loss computation with current weights
        loss = (loss_weights['reconstruction_weight'] * outputs['reconstruction_loss'] +
                loss_weights['kl_weight'] * outputs['total_kl_loss'] +
                loss_weights['memory_weight'] * outputs['memory_kl_loss'])

        # Check for NaN or inf losses
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected at step {step}")
            print(f"  Reconstruction loss: {outputs['reconstruction_loss']}")
            print(f"  Total KL loss: {outputs['total_kl_loss']}")
            print(f"  Memory KL loss: {outputs['memory_kl_loss']}")
            print(f"  Loss weights: {loss_weights}")
            
            # Check individual components for NaN
            if torch.isnan(outputs['reconstruction_loss']):
                print("  -> Reconstruction loss is NaN - likely decoder issue")
            if torch.isnan(outputs['memory_kl_loss']):
                print("  -> Memory KL loss is NaN - likely memory initialization issue")
            if torch.isnan(outputs['total_kl_loss']):
                print("  -> Total KL loss is NaN - likely encoder issue")
                
            # Use a small finite loss to prevent training crash
            loss = torch.tensor(1.0, requires_grad=True, device=self.device)
            print(f"  -> Using fallback loss: {loss}")

        # Get batch size for proper logging
        batch_size = batch['input_ids'].size(0)

        # Log metrics with batch size
        self.log('train/loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/reconstruction_loss',
                 outputs['reconstruction_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/text_kl_loss',
                 outputs['text_kl_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/multimodal_kl_loss',
                 outputs['multimodal_kl_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/memory_kl_loss',
                 outputs['memory_kl_loss'], on_step=True, on_epoch=True, batch_size=batch_size)

        # Log loss weights
        self.log('train/kl_weight', loss_weights['kl_weight'], on_step=True, batch_size=batch_size)
        self.log('train/memory_weight',
                 loss_weights['memory_weight'], on_step=True, batch_size=batch_size)

        # Store for epoch end
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'reconstruction_loss': outputs['reconstruction_loss'].detach(),
            'text_kl_loss': outputs['text_kl_loss'].detach(),
            'multimodal_kl_loss': outputs['multimodal_kl_loss'].detach(),
            'memory_kl_loss': outputs['memory_kl_loss'].detach()
        })

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step"""
        outputs = self.forward(batch)

        # Check for NaN or inf losses
        if torch.isnan(outputs['loss']) or torch.isinf(outputs['loss']):
            print(f"WARNING: NaN/Inf validation loss detected at batch {batch_idx}")
            # Return a dummy output to prevent training failure
            dummy_output = {k: torch.tensor(0.0, device=self.device) for k in outputs.keys()}
            return dummy_output

        # Get batch size for proper logging
        batch_size = batch['input_ids'].size(0)

        # Log validation metrics with batch size
        self.log('val/loss', outputs['loss'],
                 on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/reconstruction_loss',
                 outputs['reconstruction_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/text_kl_loss',
                 outputs['text_kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/multimodal_kl_loss',
                 outputs['multimodal_kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/memory_kl_loss',
                 outputs['memory_kl_loss'], on_step=False, on_epoch=True, batch_size=batch_size)

        # Store for epoch end
        self.validation_step_outputs.append({
            'loss': outputs['loss'].detach(),
            'reconstruction_loss': outputs['reconstruction_loss'].detach(),
            'text_kl_loss': outputs['text_kl_loss'].detach(),
            'multimodal_kl_loss': outputs['multimodal_kl_loss'].detach(),
            'memory_kl_loss': outputs['memory_kl_loss'].detach()
        })

        return outputs

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch"""
        if len(self.training_step_outputs) > 0:
            # Calculate epoch averages
            avg_loss = torch.stack([x['loss']
                                   for x in self.training_step_outputs]).mean()
            avg_rec_loss = torch.stack(
                [x['reconstruction_loss'] for x in self.training_step_outputs]).mean()
            avg_text_kl = torch.stack([x['text_kl_loss']
                                      for x in self.training_step_outputs]).mean()
            avg_multimodal_kl = torch.stack(
                [x['multimodal_kl_loss'] for x in self.training_step_outputs]).mean()
            avg_memory_kl = torch.stack(
                [x['memory_kl_loss'] for x in self.training_step_outputs]).mean()

            # Log epoch averages
            self.log('train/epoch_loss', avg_loss)
            self.log('train/epoch_reconstruction_loss', avg_rec_loss)
            self.log('train/epoch_text_kl_loss', avg_text_kl)
            self.log('train/epoch_multimodal_kl_loss', avg_multimodal_kl)
            self.log('train/epoch_memory_kl_loss', avg_memory_kl)

            # Clear outputs
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        if len(self.validation_step_outputs) > 0:
            # Calculate epoch averages
            avg_loss = torch.stack(
                [x['loss'] for x in self.validation_step_outputs]).mean()
            avg_rec_loss = torch.stack(
                [x['reconstruction_loss'] for x in self.validation_step_outputs]).mean()
            avg_text_kl = torch.stack([x['text_kl_loss']
                                      for x in self.validation_step_outputs]).mean()
            avg_multimodal_kl = torch.stack(
                [x['multimodal_kl_loss'] for x in self.validation_step_outputs]).mean()
            avg_memory_kl = torch.stack(
                [x['memory_kl_loss'] for x in self.validation_step_outputs]).mean()

            # Log epoch averages
            self.log('val/epoch_loss', avg_loss)
            self.log('val/epoch_reconstruction_loss', avg_rec_loss)
            self.log('val/epoch_text_kl_loss', avg_text_kl)
            self.log('val/epoch_multimodal_kl_loss', avg_multimodal_kl)
            self.log('val/epoch_memory_kl_loss', avg_memory_kl)

            # Clear outputs
            self.validation_step_outputs.clear()

            # Generate samples for evaluation
            self.generate_samples()

    def generate_samples(self, num_samples: int = 3) -> None:
        """Generate sample texts for evaluation"""
        self.model.eval()

        try:
            # Create some sample inputs
            sample_texts = [
                "A beautiful sunset over the mountains",
                "The cat sat on the mat",
                "In the beginning was the word"
            ]

            generated_texts = []

            for i, text in enumerate(sample_texts[:num_samples]):
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=50
                ).to(self.device)

                # Generate
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=self.hparams.max_generation_length,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                # Decode
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )

                generated_texts.append(
                    f"Input: {text} -> Generated: {generated_text}")

            # Log generated samples to W&B
            for i, gen_text in enumerate(generated_texts):
                if hasattr(self.logger, 'experiment'):
                    # For W&B logger
                    if hasattr(self.logger.experiment, 'log'):
                        self.logger.experiment.log({
                            f"Generated_Sample_{i}": gen_text,
                            "epoch": self.current_epoch
                        })
                    else:
                        print(f"Generated Sample {i}: {gen_text}")

        except Exception as e:
            print(f"Error generating samples: {e}")

        self.model.train()

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Ensure learning_rate is float for mathematical operations
        learning_rate = float(self.hparams.learning_rate)
        weight_decay = float(self.hparams.weight_decay)
        warmup_steps = int(self.hparams.warmup_steps)
        
        # Separate parameters for different learning rates
        encoder_params = []
        decoder_params = []
        memory_params = []
        fusion_params = []

        for name, param in self.model.named_parameters():
            if 'text_encoder' in name or 'vision_encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            elif 'memory' in name:
                memory_params.append(param)
            else:
                fusion_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = [
            # Lower LR for encoders
            {'params': encoder_params, 'lr': learning_rate * 0.5},
            {'params': decoder_params, 'lr': learning_rate},
            # Higher LR for memory
            {'params': memory_params, 'lr': learning_rate * 1.5},
            {'params': fusion_params, 'lr': learning_rate}
        ]

        # Choose optimizer
        if self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
                eps=1e-8
            )
        elif self.hparams.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.hparams.optimizer_type}")

        # Choose scheduler
        if self.hparams.scheduler_type == "linear":
            # Safely estimate total steps - handle small datasets
            try:
                total_steps = self.trainer.estimated_stepping_batches
            except (ValueError, AttributeError):
                # Fallback: estimate manually for small datasets
                if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
                    try:
                        train_loader = self.trainer.datamodule.train_dataloader()
                        total_steps = len(train_loader) * self.hparams.max_epochs
                    except:
                        total_steps = 1000  # Conservative fallback
                else:
                    total_steps = 1000  # Conservative fallback
                
            # Adjust warmup steps for small datasets
            warmup_steps = min(warmup_steps, total_steps // 4)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric=None):
        """Step the learning rate scheduler"""
        if self.hparams.scheduler_type == "linear":
            scheduler.step()
        else:
            if metric is not None:
                scheduler.step(metric)
            else:
                scheduler.step()


def create_larimar_babylm_model(
    config_path: Optional[str] = None,
    **kwargs
) -> LarimarBabyLMLightningModel:
    """
    Factory function to create a LarimarBabyLMLightningModel

    Args:
        config_path: Path to configuration file (optional)
        **kwargs: Additional arguments to override config

    Returns:
        Initialized Lightning model
    """
    # Load config if provided
    if config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = LarimarMultiModalConfig.from_dict(config_dict)
    else:
        config = LarimarMultiModalConfig()

    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create model
    model = LarimarBabyLMLightningModel(config=config, **kwargs)

    return model


def create_data_module(
    data_path: str = "../babylm_dataset",
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 12,
    max_length: int = 512,
    dataset_type: str = "cc_3M",
    **kwargs
) -> BabyLMMultiModalDataModule:
    """
    Factory function to create a BabyLMMultiModalDataModule

    Returns:
        Initialized data module
    """
    return BabyLMMultiModalDataModule(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        max_length=max_length,
        dataset_type=dataset_type,
        **kwargs
    )
