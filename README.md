# Tiny-MultiModal-Larimar

A smaller, cognitively-inspired multimodal model based on Larimar architecture, designed for efficient vision-language understanding with episodic memory control.

## Architecture

This model combines:
- **Vision Encoder**: DiNOv2 ViT-Base for visual representation
- **Text Encoder**: DistilBERT-Base for text understanding  
- **Multimodal Fusion**: Cross-attention mechanism between vision and text
- **Episodic Memory**: Smaller memory bank (128 slots) for memory-augmented generation
- **Decoder**: DistilGPT-2 for text generation

## Key Features

- **Cognitive Inspiration**: Episodic memory system mimicking human memory processes
- **Multimodal**: Handles both images and text inputs
- **Efficient**: Smaller model size while maintaining performance
- **Memory-Augmented**: Uses generative parametric memory for context-aware generation

## Data

The model is trained on:
- **BabyLM Dataset**: https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip=
- **Image-Caption Pairs**: 
  - Localized Narratives (OpenImage + MSCOCO training sets)
  - Conceptual Captions 3M (training split)
  - Pre-computed DiNOv2 embeddings available

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Installation**
```bash
pip install -r requirements.txt
```

2. **Test Setup**
```bash
python test_setup.py
```

3. **Prepare Data**
```bash
python scripts/prepare_data.py
```

4. **Train Model**
```bash
python train.py --config configs/config_tiny_multimodal.yaml
```

5. **Run Inference**
```bash
python inference.py --image_path path/to/image.jpg --text_prompt "Describe this image"
```

## Usage

### Training
```bash
python train.py --config configs/config_tiny_multimodal.yaml
```

### Inference
```bash
python inference.py --image_path path/to/image.jpg --text_prompt "Describe this image"
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint_path path/to/checkpoint.ckpt
```

### Example Usage
See `example_usage.ipynb` for detailed examples of how to use the model components.

## Model Configuration

- **Vision Encoder**: DiNOv2 ViT-Base (768 dim)
- **Text Encoder**: DistilBERT-Base (768 dim)
- **Latent Size**: 384 (smaller than original 768)
- **Memory Size**: 128 slots (vs 512 in original)
- **Decoder**: DistilGPT-2 (768 dim)
- **Episode Length**: 8 (vs 16 in original)

This design achieves a good balance between model capacity and computational efficiency while maintaining the cognitive principles of the original Larimar architecture.
