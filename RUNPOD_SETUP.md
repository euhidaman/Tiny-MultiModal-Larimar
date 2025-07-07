# RunPod Optimization Configurations for Tiny-MultiModal-Larimar

**Updated for Better Models:**
- Text Encoder: `microsoft/deberta-v3-base` (184M parameters)
- Text Decoder: `gpt2-medium` (355M parameters)  
- Vision Encoder: `facebook/dinov2-base` (86M parameters)
- **Total Model Size: ~300M parameters** (upgraded from ~150M)

## For RTX 4090 (24GB) - Budget Option
```yaml
# Modify these settings in config_tiny_multimodal.yaml
trainer:
  precision: "16"              # Essential for memory savings
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2   # Effective batch size = 8 * 2 = 16

data:
  batch_size: 8                # Reduced due to larger model
  num_workers: 2               # Reduce for stability
  pin_memory: true

model:
  latent_size: 512
  hidden_size: 1024
  memory_size: 192             # Slightly reduced for memory
```

## For RTX A6000 (48GB) - Recommended  
```yaml
# Optimal settings for A6000 with upgraded models
trainer:
  precision: "16"              # Recommended for efficiency
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1   

data:
  batch_size: 12               # Balanced for larger model
  num_workers: 4
  pin_memory: true

model:
  latent_size: 512
  hidden_size: 1024
  memory_size: 256             # Full memory size
```

## For A100 40GB/80GB - High Performance
```yaml
# High performance settings for larger models
trainer:
  precision: "16"              # Essential for A100 efficiency
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  strategy: "ddp"              # If using multiple GPUs

data:
  batch_size: 16               # Can go higher on A100 80GB
  num_workers: 8
  pin_memory: true

model:
  latent_size: 512
  hidden_size: 1024
  memory_size: 256             # Full memory size
```

## Memory Monitoring Commands
```bash
# Monitor GPU memory during training
nvidia-smi -l 1

# Check memory usage in Python
python -c "import torch; print(f'GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

## RunPod Setup Commands
```bash
# Install requirements
pip install -r requirements.txt

# Test setup
python test_setup.py

# Start training with monitoring
python train.py --config configs/config_tiny_multimodal.yaml
```
