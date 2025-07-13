#!/usr/bin/env python3
"""
Training script for Tiny-MultiModal-Larimar using authentic Larimar architecture.
This script trains the model on BabyLM multimodal data with DiNOv2 vision features.
"""

from src.modules.larimar_multimodal_vae import LarimarMultiModalConfig
from src.modules.larimar_babylm_lightning import (
    LarimarBabyLMLightningModel,
    create_larimar_babylm_model,
    create_data_module
)
import os
import argparse
import yaml
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Tiny-MultiModal-Larimar model")

    # Data arguments
    parser.add_argument("--data_path", type=str, default="../babylm_dataset",
                        help="Path to BabyLM multimodal data")
    parser.add_argument("--dataset_type", type=str, default="cc_3M",
                        choices=["cc_3M", "local_narr"],
                        help="Dataset type to use")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")

    # Model arguments
    parser.add_argument("--text_model", type=str, default="bert-base-uncased",
                        help="Text encoder model name")
    parser.add_argument("--vision_model", type=str, default="facebook/dinov2-base",
                        help="Vision encoder model name")
    parser.add_argument("--decoder_model", type=str, default="gpt2-medium",
                        help="Decoder model name")
    parser.add_argument("--text_latent_size", type=int, default=384,
                        help="Text latent dimension")
    parser.add_argument("--vision_latent_size", type=int, default=384,
                        help="Vision latent dimension")
    parser.add_argument("--memory_size", type=int, default=512,
                        help="Memory size")
    parser.add_argument("--use_memory", action="store_true", default=True,
                        help="Use episodic memory")
    parser.add_argument("--fusion_type", type=str, default="cross_attention",
                        choices=["cross_attention", "concat", "add"],
                        help="Multimodal fusion type")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--kl_warmup_steps", type=int, default=5000,
                        help="KL loss warmup steps")
    parser.add_argument("--memory_warmup_steps", type=int, default=3000,
                        help="Memory loss warmup steps")

    # Loss weights
    parser.add_argument("--kl_weight", type=float, default=1.0,
                        help="KL divergence loss weight")
    parser.add_argument("--memory_weight", type=float, default=1.0,
                        help="Memory loss weight")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0,
                        help="Reconstruction loss weight")

    # Optimization
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adam"],
                        help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Learning rate scheduler")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Gradient accumulation steps")

    # Hardware
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices (GPUs)")
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="Accelerator type")
    parser.add_argument("--strategy", type=str, default="auto",
                        help="Training strategy")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")

    # Checkpointing and logging
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="larimar_babylm",
                        help="Experiment name")
    parser.add_argument("--save_top_k", type=int, default=3,
                        help="Number of best checkpoints to save")
    parser.add_argument("--every_n_epochs", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every_n_steps", type=int, default=100,
                        help="Log every N steps")

    # Logging backend
    parser.add_argument("--logger", type=str, default="wandb",
                        choices=["tensorboard", "wandb"],
                        help="Logger backend")
    parser.add_argument("--wandb_project", type=str, default="tiny-multimodal-larimar",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="babylm-ntust",
                        help="Wandb entity/team name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (auto-generated if not provided)")
    parser.add_argument("--wandb_key", type=str, default="5fba3726e4e32540d9fcba403f880dfaad983051",
                        help="Wandb API key")

    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Validation and testing
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="Validation check interval")
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit training batches (for debugging)")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Limit validation batches (for debugging)")

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--early_stopping_monitor", type=str, default="val/loss",
                        help="Metric to monitor for early stopping")

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(args) -> LarimarMultiModalConfig:
    """Create model configuration from arguments"""
    return LarimarMultiModalConfig(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        decoder_model_name=args.decoder_model,
        text_latent_size=args.text_latent_size,
        vision_latent_size=args.vision_latent_size,
        memory_size=args.memory_size,
        use_memory=args.use_memory,
        max_length=args.max_length,
        kl_weight=args.kl_weight,
        memory_weight=args.memory_weight,
        reconstruction_weight=args.reconstruction_weight
    )


def setup_logging(args):
    """Setup logging backend"""
    if args.logger == "wandb":
        # Set up W&B with your credentials
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_key

        # Auto-generate run name if not provided
        if args.run_name is None:
            # Find the next available run number
            run_number = 1
            try:
                api = wandb.Api()
                runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}")
                existing_names = [
                    run.name for run in runs if run.name and run.name.startswith("baby-larimar")]
                while f"baby-larimar{run_number}" in existing_names:
                    run_number += 1
            except:
                pass  # If API fails, start with 1
            args.run_name = f"baby-larimar{run_number}"

        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            save_dir=args.output_dir
        )
        print(
            f"🔗 W&B logging setup: {args.wandb_entity}/{args.wandb_project} - {args.run_name}")
    else:
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name=args.experiment_name,
            log_graph=True
        )

    return logger


def setup_callbacks(args):
    """Setup training callbacks"""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            args.output_dir, args.experiment_name, "checkpoints"),
        filename="larimar-{epoch:02d}-{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=args.save_top_k,
        every_n_epochs=args.every_n_epochs,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Early stopping
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor=args.early_stopping_monitor,
            patience=args.early_stopping_patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)

    return callbacks


def main():
    """Main training function"""
    args = parse_args()

    # Load config from file if provided
    if args.config is not None:
        config_dict = load_config_from_yaml(args.config)
        # Update args with config values
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(
        args.output_dir, args.experiment_name, "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    print("=" * 80)
    print(f"Training Tiny-MultiModal-Larimar")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output dir: {args.output_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Text model: {args.text_model}")
    print(f"Vision model: {args.vision_model}")
    print(f"Decoder model: {args.decoder_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Memory size: {args.memory_size}")
    print(f"Use memory: {args.use_memory}")
    print("=" * 80)

    # Set random seeds for reproducibility
    L.seed_everything(42, workers=True)

    # Create model configuration
    model_config = create_model_config(args)

    # Create model
    model = LarimarBabyLMLightningModel(
        config=model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        kl_warmup_steps=args.kl_warmup_steps,
        memory_warmup_steps=args.memory_warmup_steps,
        max_generation_length=50,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        log_every_n_steps=args.log_every_n_steps
    )

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create data module
    data_module = create_data_module(
        data_path=args.data_path,
        tokenizer_name=args.text_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dataset_type=args.dataset_type,
        num_workers=args.num_workers
    )

    # Setup logging and callbacks
    logger = setup_logging(args)
    callbacks = setup_callbacks(args)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Log model summary
    print("\nModel Summary:")
    print(f"Text Encoder: {model.model.text_encoder.__class__.__name__}")
    print(f"Vision Encoder: {model.model.vision_encoder.__class__.__name__}")
    print(f"Decoder: {model.model.decoder.__class__.__name__}")
    if model.model.memory is not None:
        print(f"Memory: {model.model.memory.__class__.__name__}")
    print(f"Fusion: {model.model.fusion_type}")

    # Start training
    print(f"\nStarting training...")
    trainer.fit(
        model,
        data_module,
        ckpt_path=args.resume_from_checkpoint
    )

    print(f"\nTraining completed!")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(
        f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
