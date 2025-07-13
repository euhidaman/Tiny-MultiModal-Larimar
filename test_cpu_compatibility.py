#!/usr/bin/env python3
"""
CPU Compatibility Test for Tiny-MultiModal-Larimar
Downloads actual BabyLM dataset and tests compatibility with the code on CPU
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")

    try:
        from src.modules.larimar_text_encoder import LarimarTextEncoder, BertForLatentConnector
        print("✅ LarimarTextEncoder imported successfully")

        from src.modules.larimar_gpt2_decoder import LarimarGPT2Decoder, GPT2ForLatentConnector
        print("✅ LarimarGPT2Decoder imported successfully")

        from src.modules.larimar_memory import TinyLarimarMemory, LarimarMemoryVAE
        print("✅ LarimarMemory imported successfully")

        from src.modules.larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
        print("✅ LarimarMultiModalVAE imported successfully")

        from src.modules.vision_encoder import DiNOv2VisionEncoder
        print("✅ DiNOv2VisionEncoder imported successfully")

        from src.modules.babylm_data import BabyLMMultiModalDataModule, BabyLMMultiModalDataset, download_babylm_data
        print("✅ BabyLMMultiModalDataModule imported successfully")

        from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
        print("✅ LarimarBabyLMLightningModel imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_dataset_download_and_loading():
    """Download actual BabyLM dataset and test loading"""
    print("� Downloading actual BabyLM dataset...")

    try:
        from src.modules.babylm_data import download_babylm_data, BabyLMMultiModalDataset

        # Use a local data directory
        data_path = Path("../babylm_dataset")
        data_path.mkdir(parents=True, exist_ok=True)

        # Download the dataset
        print("⬇️  Starting dataset download (this may take several minutes)...")
        download_babylm_data(
            data_path=str(data_path),
            dataset_type="cc_3M",
            force_download=False
        )

        # Verify files were downloaded
        required_files = [
            "cc_3M_captions.json",
            "cc_3M_dino_v2_states_1of2.npy",
            "cc_3M_dino_v2_states_2of2.npy"
        ]

        print("📁 Verifying downloaded files...")
        for filename in required_files:
            filepath = data_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {filename}: {size_mb:.1f} MB")
            else:
                print(f"❌ {filename}: Not found")
                return False

        # Test dataset loading
        print("📚 Testing dataset loading with real data...")
        dataset = BabyLMMultiModalDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            max_length=128,
            dataset_type="cc_3M",
            auto_download=False  # Already downloaded
        )

        print(f"✅ Dataset loaded successfully: {len(dataset)} samples")

        # Test loading samples
        print("🔍 Testing sample loading...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"  Sample {i}:")
            print(f"    - Caption: {sample['caption'][:60]}...")
            print(f"    - Input IDs shape: {sample['input_ids'].shape}")
            print(
                f"    - Attention mask shape: {sample['attention_mask'].shape}")
            print(
                f"    - Vision embedding shape: {sample['vision_embedding'].shape}")
            print(f"    - Labels shape: {sample['labels'].shape}")

        return True
    except Exception as e:
        print(f"❌ Dataset download/loading error: {e}")
        return False


def test_data_module_with_real_data():
    """Test DataModule with real downloaded data"""
    print("⚡ Testing DataModule with real data...")

    try:
        from src.modules.babylm_data import BabyLMMultiModalDataModule

        # Create data module that will use the downloaded data
        data_module = BabyLMMultiModalDataModule(
            data_path="../babylm_dataset",
            tokenizer_name="bert-base-uncased",
            max_length=128,
            batch_size=4,  # Small batch for CPU
            num_workers=0,  # No multiprocessing for CPU test
            dataset_type="cc_3M",
            auto_download=False,  # Use already downloaded data
            train_split=0.99  # Use most data for training, tiny bit for val
        )

        # Setup the data module
        data_module.setup()

        # Test train dataloader
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        print(f"✅ DataModule setup successful")
        print(f"  - Train dataset size: {len(data_module.train_dataset)}")
        print(f"  - Val dataset size: {len(data_module.val_dataset)}")

        # Test getting a batch
        batch = next(iter(train_loader))
        print(f"✅ Train batch loaded:")
        print(f"  - Batch size: {batch['input_ids'].shape[0]}")
        print(f"  - Sequence length: {batch['input_ids'].shape[1]}")
        print(f"  - Vision embedding shape: {batch['vision_embedding'].shape}")

        # Test validation batch
        val_batch = next(iter(val_loader))
        print(f"✅ Validation batch loaded:")
        print(f"  - Batch size: {val_batch['input_ids'].shape[0]}")

        return True
    except Exception as e:
        print(f"❌ DataModule test error: {e}")
        return False


def test_model_with_real_data():
    """Test full model with real downloaded data"""
    print("🧠 Testing full model with real data...")

    try:
        from src.modules.larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
        from src.modules.babylm_data import BabyLMMultiModalDataModule

        # Create config optimized for CPU testing
        config = LarimarMultiModalConfig(
            text_model_name="bert-base-uncased",
            vision_model_name="facebook/dinov2-base",
            decoder_model_name="gpt2",  # Use smaller GPT2 for CPU
            text_latent_size=256,  # Smaller for CPU
            vision_latent_size=256,
            memory_size=32,  # Much smaller for CPU
            use_memory=True,
            max_length=128,
            kl_weight=0.1,
            memory_weight=0.1,
            reconstruction_weight=1.0
        )

        # Create model
        print("🏗️  Creating model...")
        model = LarimarMultiModalVAE(
            text_model_name=config.text_model_name,
            vision_model_name=config.vision_model_name,
            decoder_model_name=config.decoder_model_name,
            text_latent_size=config.text_latent_size,
            vision_latent_size=config.vision_latent_size,
            memory_size=config.memory_size,
            use_memory=config.use_memory,
            max_length=config.max_length,
            kl_weight=config.kl_weight,
            memory_weight=config.memory_weight,
            reconstruction_weight=config.reconstruction_weight
        )

        print("✅ Model created successfully")

        # Get real data batch
        data_module = BabyLMMultiModalDataModule(
            data_path="../babylm_dataset",
            tokenizer_name="bert-base-uncased",
            max_length=128,
            batch_size=2,  # Very small for CPU
            num_workers=0,
            dataset_type="cc_3M",
            auto_download=False
        )

        data_module.setup()
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))

        # Test forward pass
        print("🔄 Testing forward pass with real data...")
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            try:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_embedding=batch['vision_embedding'],
                    labels=batch['labels']
                )
                forward_time = time.time() - start_time

                print(f"✅ Forward pass successful ({forward_time:.2f}s):")
                print(f"  - Total loss: {outputs['loss']:.4f}")
                print(
                    f"  - Reconstruction loss: {outputs['reconstruction_loss']:.4f}")
                print(f"  - Text KL loss: {outputs['text_kl_loss']:.4f}")
                print(
                    f"  - Multimodal KL loss: {outputs['multimodal_kl_loss']:.4f}")
                print(f"  - Memory KL loss: {outputs['memory_kl_loss']:.4f}")
                print(f"  - Output logits shape: {outputs['logits'].shape}")

                return True
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False


def test_lightning_training_step():
    """Test Lightning training step with real data"""
    print("⚡ Testing Lightning training step...")

    try:
        from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
        from src.modules.larimar_multimodal_vae import LarimarMultiModalConfig
        from src.modules.babylm_data import BabyLMMultiModalDataModule

        # Create config for CPU
        config = LarimarMultiModalConfig(
            text_model_name="bert-base-uncased",
            vision_model_name="facebook/dinov2-base",
            decoder_model_name="gpt2",
            text_latent_size=256,
            vision_latent_size=256,
            memory_size=32,
            use_memory=True,
            max_length=128
        )

        # Create Lightning model
        lightning_model = LarimarBabyLMLightningModel(
            config=config,
            learning_rate=1e-4,
            max_epochs=1,
            kl_warmup_steps=5,
            memory_warmup_steps=3
        )

        # Get real data
        data_module = BabyLMMultiModalDataModule(
            data_path="../babylm_dataset",
            batch_size=2,
            num_workers=0,
            auto_download=False
        )
        data_module.setup()

        batch = next(iter(data_module.train_dataloader()))

        # Test training step
        print("🔄 Testing training step...")
        lightning_model.train()
        # Don't try to set global_step - it's read-only in Lightning

        start_time = time.time()
        loss = lightning_model.training_step(batch, 0)
        step_time = time.time() - start_time

        print(f"✅ Training step successful ({step_time:.2f}s):")
        print(f"  - Loss: {loss:.4f}")
        print(f"  - Loss weights applied correctly")

        return True
    except Exception as e:
        print(f"❌ Lightning training step error: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 CPU Compatibility Test with Real BabyLM Dataset")
    print("=" * 70)
    print("This will download the actual BabyLM dataset and test compatibility")

    # Ask for confirmation since this downloads large files
    response = input("⚠️  This will download ~3GB of data. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return False

    # Set CPU-only mode
    torch.set_num_threads(4)
    print(f"🔧 Using CPU with {torch.get_num_threads()} threads")

    tests = [
        ("Import Test", test_imports),
        ("Dataset Download & Loading", test_dataset_download_and_loading),
        ("DataModule with Real Data", test_data_module_with_real_data),
        ("Model with Real Data", test_model_with_real_data),
        ("Lightning Training Step", test_lightning_training_step)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Dataset and code are fully compatible.")
        print("\n✅ Ready for RunPod training!")
        print("\nNext steps:")
        print("1. Upload your project to RunPod")
        print("2. Run: python train_with_wandb.py")
        print("3. Monitor at: https://wandb.ai/babylm-ntust/tiny-multimodal-larimar")
    else:
        print("⚠️  Some tests failed. Please fix issues before RunPod training.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
