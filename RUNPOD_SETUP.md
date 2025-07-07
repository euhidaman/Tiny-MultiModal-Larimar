# RunPod Optimization Configurations for Tiny-MultiModal-Larimar

## For RTX 4090 (24GB) - Budget Option
```yaml
# Modify these settings in config_tiny_multimodal.yaml
trainer:
  precision: "16"              # Use FP16 to save memory
  gradient_clip_val: 1.0
  accumulate_grad_batches: 2   # Effective batch size = 16 * 2 = 32

data:
  batch_size: 12               # Reduce if memory issues
  num_workers: 2               # Reduce for stability
  pin_memory: true
```

## For RTX A6000 (48GB) - Recommended
```yaml
# Optimal settings for A6000
trainer:
  precision: "16"              # Optional, can use "32" too
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1   # Can increase batch_size instead

data:
  batch_size: 24               # Increase for better training
  num_workers: 4
  pin_memory: true
```

## For A100 40GB/80GB - High Performance
```yaml
# High performance settings
trainer:
  precision: "16"              # Recommended for A100
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  strategy: "ddp"              # If using multiple GPUs

data:
  batch_size: 32               # Can go higher on A100 80GB
  num_workers: 8
  pin_memory: true
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
