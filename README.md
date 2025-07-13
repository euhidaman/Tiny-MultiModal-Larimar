# Tiny-MultiModal-Larimar with Authentic Larimar Architecture

A compact multimodal AI model combining **authentic Larimar** text processing with **DiNOv2** vision encoding and **episodic memory**. Designed for efficient CPU testing and GPU training.

## 🏗️ Architecture

**Authentic Larimar Components + DiNOv2 Vision:**

```text
Text Input → BERT Encoder (Larimar) → Text Latent (384D)
                                           ↓
DiNOv2 Features → Vision Encoder → Vision Latent (384D)  
                                           ↓
                               Cross-Attention Fusion
                                           ↓
                               Multimodal Latent (384D)
                                           ↓
                               Episodic Memory (512 slots)
                                           ↓
                               GPT2 Decoder (Larimar) → Generated Text
```

### Key Components

- **Text Processing**: BERT encoder + GPT2 decoder (authentic Larimar)
- **Vision Processing**: DiNOv2 with pre-computed embeddings
- **Memory System**: 512-slot episodic memory with attention
- **Fusion**: Cross-attention between text and vision features
- **Parameters**: ~300M total (CPU testable, GPU trainable)

## 📦 Installation

```bash
# Clone repository and install dependencies
git clone <repo-url>
cd Tiny-MultiModal-Larimar
pip install torch torchvision lightning transformers
pip install datasets accelerate wandb tensorboard
```

## 🗂️ Data Structure

Expected data format:

```text
../babylm_dataset/
├── cc_3M_captions.json              # Text captions
├── cc_3M_dino_v2_states_1of2.npy    # DiNOv2 embeddings (part 1)
├── cc_3M_dino_v2_states_2of2.npy    # DiNOv2 embeddings (part 2)
└── local_narr_captions.json         # Optional: Local Narratives
```

## 🚀 Quick Start

### 1. Test Integration (CPU)

```bash
python test_larimar_integration.py
```

### 2. Train Model (GPU/CPU)

```bash
# CPU testing (small config)
python train_larimar_babylm.py --devices 0 --accelerator cpu --batch_size 2 --max_epochs 1

# GPU training (full config)  
python train_larimar_babylm.py --config configs/config_larimar_babylm.yaml
```

### 3. Monitor Training

```bash
tensorboard --logdir outputs/
```

## ⚙️ Configuration

Key settings in `configs/config_larimar_babylm.yaml`:

```yaml
# Model (Authentic Larimar)
text_model: "bert-base-uncased"      # BERT encoder
decoder_model: "gpt2-medium"         # GPT2 decoder  
vision_model: "facebook/dinov2-base" # DiNOv2 vision

# Memory & Latent
memory_size: 512
text_latent_size: 384
vision_latent_size: 384
use_memory: true

# Training
batch_size: 12
learning_rate: 1e-4
max_epochs: 10
```

## 🔧 Model Components

### Authentic Larimar Text Processing

- **BertForLatentConnector**: BERT with latent projection (mu, logvar)
- **GPT2ForLatentConnector**: GPT2 with latent conditioning
- **Special tokens**: `<PAD>`, `<BOS>`, `<EOS>` for generation

### DiNOv2 Vision Processing  

- Pre-computed embeddings (768D → 384D projection)
- Efficient processing without raw image loading
- Compatible with `cc_3M_dino_v2_states_*.npy` files

### Episodic Memory System

- **512 memory slots** for long-term context
- **Attention-based reading/writing** 
- **KL regularization** for stable training
- **Direct writing** with pseudoinverse for efficiency

### Multimodal Fusion

- **Cross-attention** between text and vision
- **VAE framework** with reparameterization trick
- **Progressive loss scheduling** (KL annealing)

## 📊 Performance

| Component         | Size      | Description                 |
| ----------------- | --------- | --------------------------- |
| BERT Encoder      | ~110M     | Text → Latent (384D)        |
| GPT2 Decoder      | ~124M     | Latent → Text               |
| DiNOv2 Projection | ~1M       | Vision → Latent (384D)      |
| Memory Module     | ~2M       | 512 slots × 384D            |
| **Total**         | **~240M** | CPU testable, GPU trainable |

## 🛠️ Hardware Requirements

### CPU Testing

- **RAM**: 8GB+ 
- **CPU**: 4+ cores
- **Time**: ~10min per epoch (small batch)

### GPU Training  

- **GPU**: 8GB+ VRAM (RTX 3070/4060+)
- **RAM**: 16GB+
- **Time**: ~2-5min per epoch

### Memory Optimization

```bash
# Reduce memory usage
--batch_size 4 --accumulate_grad_batches 4
--precision 16-mixed
--memory_size 256
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Use smaller models
--decoder_model gpt2  # instead of gpt2-medium
--memory_size 256     # instead of 512
--batch_size 4        # reduce batch size
```

**2. Missing Data Files**

```text
# Check data structure
../babylm_dataset/
├── cc_3M_captions.json              ✓
├── cc_3M_dino_v2_states_1of2.npy    ✓ 
├── cc_3M_dino_v2_states_2of2.npy    ✓
```

**3. Slow Training**

```bash
# Disable memory for faster training  
--use_memory false
--num_workers 2
```

**4. Poor Generation Quality**

```yaml
# Adjust in config
kl_warmup_steps: 10000
memory_warmup_steps: 5000
kl_weight: 0.5
```

## 📁 Project Structure

```text
Tiny-MultiModal-Larimar/
├── src/modules/
│   ├── larimar_text_encoder.py      # BERT encoder (Larimar)
│   ├── larimar_gpt2_decoder.py      # GPT2 decoder (Larimar)  
│   ├── larimar_memory.py            # Episodic memory
│   ├── larimar_multimodal_vae.py    # Complete model
│   ├── larimar_babylm_lightning.py  # Training module
│   ├── babylm_data.py               # Data loading
│   └── vision_encoder.py            # DiNOv2 processing
├── configs/
│   └── config_larimar_babylm.yaml   # Training config
├── train_larimar_babylm.py          # Training script
├── test_larimar_integration.py      # Integration test
└── README.md                        # This file
```

## 🧪 Testing

Run integration tests to verify everything works:

```bash
# Test all components
python test_larimar_integration.py

# Expected output:
# ✓ Larimar BERT text encoder working
# ✓ Larimar GPT2 decoder working  
# ✓ Tiny Larimar memory working
# ✓ DiNOv2 vision encoder working
# ✓ Multimodal VAE integration working
# ✓ Lightning model working
```

## 🎯 Training Tips

### For CPU Testing

```bash
python train_larimar_babylm.py \
  --accelerator cpu \
  --devices 1 \
  --batch_size 2 \
  --max_epochs 1 \
  --limit_train_batches 0.1 \
  --limit_val_batches 0.1
```

### For GPU Training

```bash
python train_larimar_babylm.py \
  --config configs/config_larimar_babylm.yaml \
  --accelerator gpu \
  --devices 1 \
  --precision 16-mixed
```

### Memory Scheduling

- **KL Annealing**: Gradual increase of KL loss weight
- **Memory Warmup**: Progressive memory loss integration  
- **Learning Rate**: Different rates for encoder/decoder/memory

## 🔗 Data Sources

- **BabyLM Dataset**: https://babylm.github.io/
- **Conceptual Captions**: Pre-computed DiNOv2 embeddings
- **Local Narratives**: Optional additional data

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **Larimar**: Original episodic memory architecture
- **Meta AI**: DiNOv2 vision encoder
- **BabyLM**: Multimodal dataset and challenge

---

**Ready to train authentic Larimar with DiNOv2 vision!** 🚀
