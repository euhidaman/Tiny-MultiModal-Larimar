#!/usr/bin/env python3
"""
Test script for Tiny-MultiModal-Larimar with authentic Larimar architecture.
This script tests the integration of Larimar text components with DiNOv2 vision.
"""

from transformers import AutoTokenizer
from src.modules.vision_encoder import DiNOv2VisionEncoder
from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
from src.modules.larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
from src.modules.larimar_memory import TinyLarimarMemory, LarimarMemoryVAE
from src.modules.larimar_gpt2_decoder import LarimarGPT2Decoder, GPT2ForLatentConnector
from src.modules.larimar_text_encoder import LarimarTextEncoder, BertForLatentConnector
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_larimar_text_encoder():
    """Test the Larimar text encoder"""
    print("Testing LarimarTextEncoder...")

    encoder = LarimarTextEncoder(
        model_name="bert-base-uncased",
        latent_size=384
    )

    # Create sample input
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "This is a test sentence for the Larimar encoder."
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)

    # Test encoding
    with torch.no_grad():
        text_features, latent_params = encoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_latent_params=True
        )

        latent_z, mu, logvar = encoder.encode_to_latent(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

    print(f"  Text features shape: {text_features.shape}")
    print(f"  Latent z shape: {latent_z.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")
    print("  ✓ LarimarTextEncoder test passed!")


def test_larimar_gpt2_decoder():
    """Test the Larimar GPT2 decoder"""
    print("\nTesting LarimarGPT2Decoder...")

    decoder = LarimarGPT2Decoder(
        model_name="gpt2",  # Use smaller model for testing
        latent_size=384
    )

    # Create sample input
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "This is a test"
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)

    # Create sample latent conditioning
    batch_size = inputs['input_ids'].shape[0]
    latent_conditioning = torch.randn(batch_size, 384)

    # Test decoding
    with torch.no_grad():
        outputs = decoder(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            latent_conditioning=latent_conditioning,
            labels=inputs['input_ids']
        )

    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss']:.4f}")
    print("  ✓ LarimarGPT2Decoder test passed!")


def test_larimar_memory():
    """Test the Larimar memory module"""
    print("\nTesting TinyLarimarMemory...")

    memory = TinyLarimarMemory(
        code_size=384,
        memory_size=512
    )

    # Create sample input
    batch_size = 2
    episode_size = 3
    input_encoded = torch.randn(episode_size, batch_size, 384)

    # Test memory writing and reading
    with torch.no_grad():
        # Write to memory
        memory_state, memory_kl = memory.write_to_memory(input_encoded)
        print(f"  Memory state shape: {memory_state[0].shape}")
        print(f"  Memory KL: {memory_kl:.4f}")

        # Read from memory
        retrieved_z, attention_weights = memory.read_from_memory(
            input_encoded, memory_state)
        print(f"  Retrieved z shape: {retrieved_z.shape}")
        print(f"  Attention weights shape: {attention_weights.shape}")

    print("  ✓ TinyLarimarMemory test passed!")


def test_vision_encoder():
    """Test the DiNOv2 vision encoder"""
    print("\nTesting DiNOv2VisionEncoder...")

    vision_encoder = DiNOv2VisionEncoder(
        model_name="facebook/dinov2-base",
        latent_size=384
    )

    # Create sample DiNOv2 embedding (as if pre-computed)
    batch_size = 2
    dinov2_dim = 768  # DiNOv2-base embedding dimension
    vision_embedding = torch.randn(batch_size, dinov2_dim)

    # Test encoding
    with torch.no_grad():
        vision_features = vision_encoder(vision_embedding)

    print(f"  Vision features shape: {vision_features.shape}")
    print("  ✓ DiNOv2VisionEncoder test passed!")


def test_multimodal_vae():
    """Test the complete multimodal VAE"""
    print("\nTesting LarimarMultiModalVAE...")

    config = LarimarMultiModalConfig(
        text_model_name="bert-base-uncased",
        vision_model_name="facebook/dinov2-base",
        decoder_model_name="gpt2",  # Use smaller model for testing
        text_latent_size=384,
        vision_latent_size=384,
        memory_size=256,  # Smaller for testing
        use_memory=True
    )

    model = LarimarMultiModalVAE(
        text_model_name=config.text_model_name,
        vision_model_name=config.vision_model_name,
        decoder_model_name=config.decoder_model_name,
        text_latent_size=config.text_latent_size,
        vision_latent_size=config.vision_latent_size,
        memory_size=config.memory_size,
        use_memory=config.use_memory
    )

    # Create sample inputs
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "This is a test sentence for multimodal learning."
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=50)

    # Create sample vision embedding
    batch_size = inputs['input_ids'].shape[0]
    vision_embedding = torch.randn(batch_size, 768)  # DiNOv2-base dimension

    # Test forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            vision_embedding=vision_embedding,
            labels=inputs['input_ids']
        )

    print(f"  Total loss: {outputs['loss']:.4f}")
    print(f"  Reconstruction loss: {outputs['reconstruction_loss']:.4f}")
    print(f"  Text KL loss: {outputs['text_kl_loss']:.4f}")
    print(f"  Multimodal KL loss: {outputs['multimodal_kl_loss']:.4f}")
    print(f"  Memory KL loss: {outputs['memory_kl_loss']:.4f}")
    print(f"  Output logits shape: {outputs['logits'].shape}")

    # Test generation
    generated = model.generate(
        input_ids=inputs['input_ids'][:, :10],  # Shorter prompt
        vision_embedding=vision_embedding,
        max_length=20,
        do_sample=False
    )
    print(f"  Generated sequence shape: {generated.shape}")

    print("  ✓ LarimarMultiModalVAE test passed!")


def test_lightning_model():
    """Test the Lightning model"""
    print("\nTesting LarimarBabyLMLightningModel...")

    config = LarimarMultiModalConfig(
        text_model_name="bert-base-uncased",
        vision_model_name="facebook/dinov2-base",
        decoder_model_name="gpt2",
        text_latent_size=256,  # Smaller for testing
        vision_latent_size=256,
        memory_size=128,
        use_memory=True
    )

    lightning_model = LarimarBabyLMLightningModel(
        config=config,
        learning_rate=1e-4,
        kl_warmup_steps=100,
        memory_warmup_steps=50
    )

    # Create sample batch
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'

    texts = ["This is a test.", "Another test sentence."]
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True, max_length=50)

    batch = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': inputs['input_ids'],
        'vision_embedding': torch.randn(len(texts), 768),  # DiNOv2 embeddings
        'caption': texts
    }

    # Test training step
    lightning_model.train()
    with torch.no_grad():
        loss = lightning_model.training_step(batch, 0)

    print(f"  Training loss: {loss:.4f}")

    # Test validation step
    lightning_model.eval()
    with torch.no_grad():
        val_outputs = lightning_model.validation_step(batch, 0)

    print(f"  Validation loss: {val_outputs['loss']:.4f}")
    print("  ✓ LarimarBabyLMLightningModel test passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Tiny-MultiModal-Larimar with Authentic Larimar Architecture")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Run component tests
        test_larimar_text_encoder()
        test_larimar_gpt2_decoder()
        test_larimar_memory()
        test_vision_encoder()
        test_multimodal_vae()
        test_lightning_model()

        print("\n" + "=" * 60)
        print("✅ All tests passed! The Larimar integration is working correctly.")
        print("=" * 60)

        # Print summary
        print("\nSummary:")
        print("  ✓ Larimar BERT text encoder working")
        print("  ✓ Larimar GPT2 decoder working")
        print("  ✓ Tiny Larimar memory working")
        print("  ✓ DiNOv2 vision encoder working")
        print("  ✓ Multimodal VAE integration working")
        print("  ✓ Lightning model working")
        print("\nReady to train with: python train_larimar_babylm.py --config configs/config_larimar_babylm.yaml")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
