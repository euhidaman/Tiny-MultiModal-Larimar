#!/usr/bin/env python3

from modules.data import MultiModalDataModule, download_babylm_data, create_dummy_multimodal_data
from modules.lightning_model import TinyMultiModalLitModel
import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Tiny MultiModal Larimar")

    # Configuration
    parser.add_argument('--config', type=str, default='configs/config_tiny_multimodal.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Data
    parser.add_argument('--train_data', type=str, default='data/train',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data')
    parser.add_argument('--data_type', type=str, default='multimodal',
                        choices=['multimodal', 'babylm', 'conceptual'],
                        help='Type of data to use')
    parser.add_argument('--download_data', action='store_true',
                        help='Download BabyLM data automatically')
    parser.add_argument('--create_dummy', action='store_true',
                        help='Create dummy data for testing')

    # Model
    parser.add_argument('--model_name', type=str, default='tiny-multimodal-larimar',
                        help='Model name for logging')
    parser.add_argument('--latent_size', type=int, default=384,
                        help='Latent space size')
    parser.add_argument('--memory_size', type=int, default=128,
                        help='Memory size')
    parser.add_argument('--no_memory', action='store_true',
                        help='Disable memory module')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Maximum number of epochs')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum number of steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Gradient accumulation steps')

    # Hardware
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'tpu'],
                        help='Accelerator to use')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Training strategy')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['16', '32', 'bf16'],
                        help='Training precision')

    # Logging
    parser.add_argument('--logger', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb', 'none'],
                        help='Logger to use')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--wandb_project', type=str, default='tiny-multimodal-larimar',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity name')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_top_k', type=int, default=3,
                        help='Number of best checkpoints to save')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run fast development run')
    parser.add_argument('--overfit_batches', type=int, default=0,
                        help='Number of batches to overfit on')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        return {}


def setup_logging(args):
    """Setup logging"""
    if args.logger == 'wandb':
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.model_name,
            save_dir=args.log_dir
        )
    elif args.logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.model_name
        )
    else:
        logger = None

    return logger


def setup_callbacks(args):
    """Setup training callbacks"""
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_epochs=args.save_every_n_epochs
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val/total_loss',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks


def main():
    """Main training function"""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Set random seed
    L.seed_everything(args.seed)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Download or create data if needed
    if args.download_data:
        download_babylm_data(args.train_data)

    if args.create_dummy:
        create_dummy_multimodal_data(args.train_data, num_samples=1000)

    # Setup data module
    data_module = MultiModalDataModule(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=config.get('max_length', 512),
        image_size=config.get('image_size', 224),
        mode=config.get('mode', 'multimodal'),
        tokenizer_name=config.get(
            'text_model_name', 'distilbert-base-uncased'),
        image_processor_name=config.get(
            'vision_model_name', 'facebook/dinov2-base')
    )

    # Setup model
    model = TinyMultiModalLitModel(
        latent_size=args.latent_size,
        memory_size=args.memory_size,
        use_memory=not args.no_memory,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        vision_model_name=config.get(
            'vision_model_name', 'facebook/dinov2-base'),
        text_model_name=config.get(
            'text_model_name', 'distilbert-base-uncased'),
        decoder_model_name=config.get('decoder_model_name', 'distilgpt2'),
        beta=config.get('beta', 0.5),
        beta_schedule=config.get('beta_schedule', 'linear'),
        reconstruction_strength=config.get('reconstruction_strength', 1.0),
        memory_strength=config.get('memory_strength', 1.0),
        freeze_vision=config.get('freeze_vision', False),
        freeze_text=config.get('freeze_text', False),
        use_cross_attention=config.get('use_cross_attention', True),
        num_attention_heads=config.get('num_attention_heads', 8)
    )

    # Print model information
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    model.print_model_info()
    print("="*50 + "\n")

    # Setup logger
    logger = setup_logging(args)

    # Setup callbacks
    callbacks = setup_callbacks(args)

    # Setup strategy
    strategy = 'auto'
    if args.strategy != 'auto':
        if args.strategy == 'ddp':
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = args.strategy

    # Setup trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        overfit_batches=args.overfit_batches,
        log_every_n_steps=50,
        val_check_interval=0.25,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train model
    print("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)

    # Test model
    if not args.fast_dev_run:
        print("Testing model...")
        trainer.test(model, data_module)

    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model')
    model.model.save_pretrained(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
