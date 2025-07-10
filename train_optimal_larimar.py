#!/usr/bin/env python3
"""
Optimal training script for Tiny-MultiModal-Larimar to beat original Larimar.
Uses the optimized configuration and comprehensive logging.
"""

import os
import torch
import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

# Model imports
from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
from src.modules.babylm_data import BabyLMMultiModalDataModule


def load_optimal_config(config_path: str) -> dict:
    """Load the optimal configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb_logging(config: dict) -> WandbLogger:
    """Setup W&B logging with optimal configuration"""

    # Auto-increment run name
    api = wandb.Api()
    runs = api.runs(
        f"{config['logging']['wandb_entity']}/{config['logging']['wandb_project']}")
    existing_names = [
        run.name for run in runs if run.name.startswith("optimal-larimar")]

    if existing_names:
        # Extract numbers and find next
        numbers = []
        for name in existing_names:
            try:
                num = int(name.split("-")[-1])
                numbers.append(num)
            except:
                continue
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    run_name = f"optimal-larimar{next_num}"

    logger = WandbLogger(
        project=config['logging']['wandb_project'],
        entity=config['logging']['wandb_entity'],
        name=run_name,
        tags=config['logging']['wandb_tags'],
        log_model=True,
        save_code=True
    )

    return logger


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks"""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='optimal-larimar-{epoch:02d}-{val_loss:.2f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=config['logging']['save_top_k'],
        save_last=config['logging']['save_last'],
        every_n_epochs=config['logging']['save_every_n_epochs'],
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val/total_loss',
        patience=config['callbacks']['early_stopping']['patience'],
        min_delta=config['callbacks']['early_stopping']['min_delta'],
        mode='min',
        verbose=True
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    return callbacks


def create_trainer(config: dict, logger, callbacks: list) -> pl.Trainer:
    """Create optimized PyTorch Lightning trainer"""

    trainer = pl.Trainer(
        # Hardware configuration
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],

        # Training configuration
        max_epochs=config['trainer']['max_epochs'],
        max_steps=config['trainer']['max_steps'],
        gradient_clip_val=config['trainer']['gradient_clip_val'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],

        # Validation configuration
        val_check_interval=config['trainer']['val_check_interval'],
        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
        limit_val_batches=config['trainer']['limit_val_batches'],

        # Logging and callbacks
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        enable_progress_bar=config['trainer']['enable_progress_bar'],

        # Optimization
        enable_model_summary=True,
        sync_batchnorm=True,

        # Debugging
        fast_dev_run=config['trainer']['fast_dev_run'],
        overfit_batches=config['trainer']['overfit_batches']
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train optimal Tiny-MultiModal-Larimar")
    parser.add_argument("--config", type=str, default="configs/config_tiny_multimodal.yaml",
                        help="Path to optimal configuration file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    # Load optimal configuration
    config = load_optimal_config(args.config)

    print("="*80)
    print("TRAINING TINY-MULTIMODAL-LARIMAR TO BEAT ORIGINAL LARIMAR")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Target: Outperform original Larimar on all benchmarks")
    print(f"Novel capabilities: Multimodal understanding + Enhanced memory")
    print("="*80)

    # Setup logging
    logger = setup_wandb_logging(config)
    print(f"W&B Run: {logger.experiment.name}")

    # Log configuration
    logger.experiment.config.update(config)

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Create trainer
    trainer = create_trainer(config, logger, callbacks)

    # Setup data
    print("Setting up BabyLM dataset...")
    data_module = BabyLMMultiModalDataModule(
        data_path=config['data']['train_data_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        persistent_workers=config['data']['persistent_workers']
    )

    # Setup model
    print("Initializing optimal Larimar model...")
    model_config = {
        # Core architecture
        'text_model': config['model']['text_model_name'],
        'decoder_model': config['model']['decoder_model_name'],
        'vision_model': config['model']['vision_model_name'],

        # Dimensions
        'text_latent_size': config['model']['latent_size'],
        'vision_latent_size': config['model']['latent_size'],
        'hidden_size': config['model']['hidden_size'],

        # Memory configuration
        'memory_size': config['memory']['memory_size'],
        'use_memory': config['model']['use_memory'],
        'direct_writing': config['memory']['direct_writing'],
        'identity_init': config['memory']['identity_init'],
        'observation_noise_std': config['memory']['observation_noise_std'],

        # Training parameters
        'learning_rate': config['optimization']['optimizer']['lr'],
        'weight_decay': config['optimization']['optimizer']['weight_decay'],
        'warmup_steps': config['optimization']['scheduler']['warmup_steps'],
        'max_steps': config['optimization']['scheduler']['max_steps'],

        # Loss weights
        'kl_weight': config['model']['kl_weight'],
        'memory_weight': config['model']['memory_strength'],
        'reconstruction_weight': config['model']['reconstruction_strength'],

        # Warmup schedules
        'kl_warmup_steps': config['model']['kl_warmup_steps'],
        'memory_warmup_steps': config['model']['memory_warmup_steps'],

        # Multimodal fusion
        'fusion_type': config['model']['fusion_type'],
        'use_cross_attention': config['model']['use_cross_attention'],
        'num_attention_heads': config['model']['num_attention_heads']
    }

    model = LarimarBabyLMLightningModel(**model_config)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Log model info
    logger.experiment.log({
        "model/total_parameters": sum(p.numel() for p in model.parameters()),
        "model/trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model/memory_size": config['memory']['memory_size'],
        "model/latent_size": config['model']['latent_size']
    })

    # Start training
    print("Starting optimal training to beat Larimar...")
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint
    )

    print("="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(
        f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(
        f"W&B Run: https://wandb.ai/{config['logging']['wandb_entity']}/{config['logging']['wandb_project']}/runs/{logger.experiment.id}")
    print("\nNext steps:")
    print("1. Run comprehensive evaluation: python evaluate_against_larimar.py")
    print("2. Compare results with original Larimar benchmarks")
    print("3. Analyze novel multimodal capabilities")
    print("="*80)


if __name__ == "__main__":
    main()
