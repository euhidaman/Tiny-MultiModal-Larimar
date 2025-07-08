#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_data_compatibility():
    """Test if the BabyLM data format is compatible with the training code"""
    data_path = Path("data/babylm")

    print("=== Testing BabyLM Data Compatibility ===")

    try:
        # Test data module import and setup
        from src.modules.babylm_data import BabyLMMultiModalDataModule

        print("✓ Data module import successful")

        # Test data loading
        dm = BabyLMMultiModalDataModule(
            data_path=str(data_path),
            dataset_type='cc_3M',
            batch_size=2,
            num_workers=0  # Avoid multiprocessing issues in testing
        )

        print("✓ Data module creation successful")

        # Setup data
        dm.setup()
        print("✓ Data setup successful")

        # Test data loader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        print("✓ Data loading successful")
        print(f"  - Batch keys: {list(batch.keys())}")
        print(f"  - Input IDs shape: {batch['input_ids'].shape}")
        print(f"  - Vision embedding shape: {batch['vision_embedding'].shape}")
        print(f"  - Labels shape: {batch['labels'].shape}")
        print(f"  - Sample caption: {batch['caption'][0][:100]}...")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_data_files():
    """Test if the actual data files exist and are readable"""
    data_path = Path("data/babylm")

    print("\n=== Testing Data Files ===")

    # Test Conceptual Captions
    print("\n--- Conceptual Captions ---")
    cc_captions = data_path / "cc_3M_captions.json"
    cc_embed1 = data_path / "cc_3M_dino_v2_states_1of2.npy"
    cc_embed2 = data_path / "cc_3M_dino_v2_states_2of2.npy"

    files_exist = True

    if cc_captions.exists():
        print(f"✓ Captions file exists: {cc_captions}")
        try:
            with open(cc_captions, 'r', encoding='utf-8') as f:
                captions = json.load(f)
            print(f"  - Captions count: {len(captions)}")
            print(f"  - Sample caption: {captions[0] if captions else 'None'}")
        except Exception as e:
            print(f"  - Error reading captions: {e}")
            files_exist = False
    else:
        print(f"✗ Captions file missing: {cc_captions}")
        files_exist = False

    if cc_embed1.exists() and cc_embed2.exists():
        print(f"✓ Embedding files exist")
        try:
            embed1 = np.load(cc_embed1)
            embed2 = np.load(cc_embed2)
            print(f"  - Embeddings 1 shape: {embed1.shape}")
            print(f"  - Embeddings 2 shape: {embed2.shape}")
            print(f"  - Total embeddings: {embed1.shape[0] + embed2.shape[0]}")
        except Exception as e:
            print(f"  - Error reading embeddings: {e}")
            files_exist = False
    else:
        print(f"✗ Embedding files missing")
        files_exist = False

    return files_exist


if __name__ == "__main__":
    print("Testing BabyLM dataset compatibility...")

    files_ok = test_data_files()
    if files_ok:
        compat_ok = test_data_compatibility()
        if compat_ok:
            print("\n🎉 All tests passed! Dataset is compatible and ready for training.")
        else:
            print("\n❌ Compatibility test failed. Check dependencies and code.")
    else:
        print("\n❌ Data files test failed. Check data paths and files.")
