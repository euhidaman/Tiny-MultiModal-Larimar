#!/usr/bin/env python3
"""
Quick test script to validate the Tiny-MultiModal-Larimar model setup.
Run this script to ensure all components are working correctly.
"""

import torch
import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.modules.multimodal_vae import TinyMultiModalVAE
        from src.modules.lightning_model import TinyMultiModalLitModel
        from src.modules.data import MultiModalDataModule
        from src.modules.memory import TinyMemory
        from src.modules.vision_encoder import DiNOv2VisionEncoder
        from src.modules.text_encoder import DistilBERTTextEncoder
        from src.modules.decoder import DistilGPT2Decoder
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test that configuration file is valid."""
    try:
        with open('configs/config_tiny_multimodal.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ['model', 'training', 'data', 'logging']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False

        print("✓ Configuration file is valid")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_model_creation():
    """Test that the model can be created and run."""
    try:
        from src.modules.multimodal_vae import TinyMultiModalVAE

        # Load config
        with open('configs/config_tiny_multimodal.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create model
        model = TinyMultiModalVAE(config['model'])
        model.eval()

        # Test shapes
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        text_ids = torch.randint(0, 1000, (batch_size, 20))
        text_mask = torch.ones(batch_size, 20)

        # Forward pass
        with torch.no_grad():
            outputs = model(images, text_ids, text_mask)

        # Check outputs
        if 'reconstruction_loss' in outputs and 'kl_loss' in outputs:
            print("✓ Model creation and forward pass successful")
            print(
                f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(
                f"  - Reconstruction loss: {outputs['reconstruction_loss'].item():.4f}")
            print(f"  - KL loss: {outputs['kl_loss'].item():.4f}")
            return True
        else:
            print("✗ Model outputs missing expected keys")
            return False

    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False


def test_lightning_model():
    """Test PyTorch Lightning model wrapper."""
    try:
        from src.modules.lightning_model import TinyMultiModalLitModel

        with open('configs/config_tiny_multimodal.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create lightning model
        lit_model = TinyMultiModalLitModel(config)

        # Test training step
        batch_size = 2
        batch = {
            'image': torch.randn(batch_size, 3, 224, 224),
            'text_ids': torch.randint(0, 1000, (batch_size, 20)),
            'text_mask': torch.ones(batch_size, 20)
        }

        with torch.no_grad():
            loss = lit_model.training_step(batch, 0)

        if isinstance(loss, torch.Tensor):
            print("✓ Lightning model training step successful")
            print(f"  - Training loss: {loss.item():.4f}")
            return True
        else:
            print("✗ Lightning model training step failed")
            return False

    except Exception as e:
        print(f"✗ Lightning model error: {e}")
        return False


def test_data_module():
    """Test data module setup."""
    try:
        from src.modules.data import MultiModalDataModule

        # Create data module with dummy data
        data_module = MultiModalDataModule(
            data_dir='data',
            batch_size=4,
            num_workers=0  # No multiprocessing for testing
        )

        # Test setup
        data_module.setup('fit')

        print("✓ Data module setup successful")
        return True

    except Exception as e:
        print(f"✗ Data module error: {e}")
        return False


def test_memory_system():
    """Test the memory system specifically."""
    try:
        from src.modules.memory import TinyMemory

        memory = TinyMemory(
            memory_dim=256,
            memory_slots=64,
            gating_dim=128
        )

        # Test memory forward pass
        batch_size = 2
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 256)

        with torch.no_grad():
            output = memory(input_tensor)

        if output.shape == input_tensor.shape:
            print("✓ Memory system test successful")
            print(f"  - Memory slots: {memory.memory_slots}")
            print(f"  - Memory dimension: {memory.memory_dim}")
            return True
        else:
            print("✗ Memory system output shape mismatch")
            return False

    except Exception as e:
        print(f"✗ Memory system error: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Tiny-MultiModal-Larimar setup...")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Lightning Model", test_lightning_model),
        ("Data Module", test_data_module),
        ("Memory System", test_memory_system)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  Failed: {test_name}")

    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The model is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python scripts/prepare_data.py' to download data")
        print("2. Run 'python train.py' to start training")
        print("3. Run 'python inference.py --help' for inference options")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
