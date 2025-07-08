#!/usr/bin/env python3

from src.modules.babylm_lightning_model import BabyLMLightningModel
from src.modules.babylm_data import BabyLMMultiModalDataModule
import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train Tiny MultiModal Larimar on BabyLM data")

    parser.add_argument('--config', type=str, default='configs/config_tiny_multimodal.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data/babylm',
                        help='Path to BabyLM data directory')
    parser.add_argument('--dataset_type', type=str, default='cc_3M', choices=['cc_3M', 'local_narr'],
                        help='Which dataset to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--fast_dev_run', action='store_true',
                        help='Run a fast development run')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.fast_dev_run:
        config['trainer']['fast_dev_run'] = True

    print("=== Training Configuration ===")
    print(f"Data path: {args.data_path}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Model config: {config['model']}")
    print(f"Data config: {config['data']}")
    print(f"Trainer config: {config['trainer']}")

    # Setup data module
    data_module = BabyLMMultiModalDataModule(
        data_path=args.data_path,
        tokenizer_name=config['model']['text_model_name'],
        max_length=config['data']['max_length'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        dataset_type=args.dataset_type
    )

    # Setup model
    model = BabyLMLightningModel(config)

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='tiny-multimodal-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name='tiny_multimodal_larimar'
    )

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        max_steps=config['trainer']['max_steps'],
        accelerator=config['trainer']['accelerator'],
        devices=config['trainer']['devices'],
        precision=config['trainer']['precision'],
        gradient_clip_val=config['trainer']['gradient_clip_val'],
        accumulate_grad_batches=config['trainer']['accumulate_grad_batches'],
        val_check_interval=config['trainer']['val_check_interval'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        fast_dev_run=config['trainer'].get('fast_dev_run', False),
        callbacks=callbacks,
        logger=logger
    )

    # Train
    print("\n=== Starting Training ===")
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume
    )

    print("\n=== Training Complete ===")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
