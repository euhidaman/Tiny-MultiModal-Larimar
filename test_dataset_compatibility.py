#!/usr/bin/env python3
"""
Quick Dataset Download Test for Tiny-MultiModal-Larimar
Tests actual dataset download and compatibility with a small subset
"""

import os
import sys
import torch
from pathlib import Path
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_dataset_download():
    """Test downloading a small subset of actual data"""
    print("🌐 Testing dataset download functionality...")
    
    try:
        from src.modules.babylm_data import download_babylm_data, BabyLMMultiModalDataset
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_path = Path(temp_dir) / "test_download"
            
            print(f"📁 Test data path: {test_data_path}")
            
            # Test download function (will download actual files)
            print("⬇️  Attempting to download dataset files...")
            download_babylm_data(
                data_path=str(test_data_path),
                dataset_type="cc_3M",
                force_download=False
            )
            
            print("✅ Download completed successfully")
            
            # Verify files exist
            required_files = [
                "cc_3M_captions.json",
                "cc_3M_dino_v2_states_1of2.npy", 
                "cc_3M_dino_v2_states_2of2.npy"
            ]
            
            for filename in required_files:
                filepath = test_data_path / filename
                if filepath.exists():
                    size_mb = filepath.stat().st_size / (1024 * 1024)
                    print(f"✅ {filename}: {size_mb:.1f} MB")
                else:
                    print(f"❌ {filename}: Not found")
                    return False
            
            # Test dataset loading with actual data
            print("📚 Testing dataset loading with downloaded data...")
            dataset = BabyLMMultiModalDataset(
                data_path=str(test_data_path),
                tokenizer_name="bert-base-uncased",
                max_length=128,
                dataset_type="cc_3M",
                auto_download=False  # Already downloaded
            )
            
            print(f"✅ Dataset loaded: {len(dataset)} samples")
            
            # Test loading a few samples
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"  Sample {i}:")
                print(f"    - Text: {sample['caption'][:50]}...")
                print(f"    - Input IDs shape: {sample['input_ids'].shape}")
                print(f"    - Vision embedding shape: {sample['vision_embedding'].shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ Dataset download test failed: {e}")
        return False

def test_model_with_real_data():
    """Test model with actual downloaded data"""
    print("🔗 Testing model with real dataset...")
    
    try:
        from src.modules.babylm_data import BabyLMMultiModalDataModule
        from src.modules.larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data_path = Path(temp_dir) / "model_test"
            
            # Create small dataset for testing
            data_module = BabyLMMultiModalDataModule(
                data_path=str(test_data_path),
                tokenizer_name="bert-base-uncased",
                max_length=128,
                batch_size=2,
                dataset_type="cc_3M",
                auto_download=True,  # Download automatically
                num_workers=0  # No multiprocessing for CPU test
            )
            
            # Setup data
            data_module.setup()
            
            # Get a small batch
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            
            print(f"✅ Real data batch loaded:")
            print(f"  - Batch size: {batch['input_ids'].shape[0]}")
            print(f"  - Sequence length: {batch['input_ids'].shape[1]}")
            print(f"  - Vision embedding shape: {batch['vision_embedding'].shape}")
            
            # Test model with real data
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
            
            model = LarimarMultiModalVAE(
                text_model_name=config.text_model_name,
                vision_model_name=config.vision_model_name,
                decoder_model_name=config.decoder_model_name,
                text_latent_size=config.text_latent_size,
                vision_latent_size=config.vision_latent_size,
                memory_size=config.memory_size,
                use_memory=config.use_memory,
                max_length=config.max_length
            )
            
            # Forward pass with real data
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    vision_embedding=batch['vision_embedding'],
                    labels=batch['labels']
                )
            
            print(f"✅ Model forward pass with real data successful:")
            print(f"  - Loss: {outputs['loss']:.4f}")
            print(f"  - Logits shape: {outputs['logits'].shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ Model with real data test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Real Dataset Compatibility Test")
    print("=" * 50)
    print("⚠️  Note: This will download actual dataset files (may take time)")
    
    response = input("Continue with download test? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    # Set CPU mode
    torch.set_num_threads(4)
    print(f"🔧 Using CPU with {torch.get_num_threads()} threads")
    
    tests = [
        ("Dataset Download Test", test_dataset_download),
        ("Model with Real Data Test", test_model_with_real_data)
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
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 Dataset compatibility verified! Ready for RunPod training.")
    else:
        print("⚠️  Issues found. Please check the errors above.")

if __name__ == "__main__":
    main()
