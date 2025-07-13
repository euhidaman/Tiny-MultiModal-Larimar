# RunPod Training Guide for Tiny-MultiModal-Larimar

## 1. RunPod Setup

### GPU Recommendation
- **RTX 4090 (24GB VRAM)** - Best price/performance
- **RTX A6000 (48GB VRAM)** - For larger batches
- **RTX 3090 (24GB VRAM)** - Alternative option

### Template Selection
- Choose **PyTorch 2.0+** with **CUDA 11.8+**
- Or **runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04**

## 2. Connect VS Code to RunPod

### Method A: SSH Connection (Recommended)

1. **Install VS Code Extensions**:
   - Remote - SSH
   - Remote - SSH: Editing Configuration Files

2. **Get SSH Connection Details**:
   - In RunPod dashboard, click "Connect" → "SSH"
   - Copy the SSH command (e.g., `ssh root@<ip> -p <port>`)

3. **Connect via VS Code**:
   ```bash
   # In VS Code: Ctrl+Shift+P → "Remote-SSH: Connect to Host"
   # Enter: root@<runpod-ip> -p <port>
   ```

### Method B: Direct Upload

1. **Zip your project locally**:
   ```powershell
   cd d:\BabyLM
   Compress-Archive -Path Tiny-MultiModal-Larimar -DestinationPath tiny-larimar.zip
   ```

2. **Upload to RunPod**:
   - Use RunPod web interface file upload
   - Or use SCP: `scp -P <port> tiny-larimar.zip root@<ip>:/workspace/`

## 3. Environment Setup on RunPod

```bash
# Update system
apt update && apt install -y git wget curl

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets lightning accelerate
pip install numpy pandas tqdm wandb tensorboard
pip install PyYAML scikit-learn

# Navigate to workspace
cd /workspace

# Extract project (if uploaded as zip)
unzip tiny-larimar.zip
cd Tiny-MultiModal-Larimar

# Or clone from GitHub (if you have a repo)
# git clone <your-repo-url>
# cd Tiny-MultiModal-Larimar
```

## 4. Prepare Data

```bash
# Create data directory
mkdir -p ../babylm_dataset

# Upload your BabyLM data files:
# - cc_3M_captions.json
# - cc_3M_dino_v2_states_1of2.npy  
# - cc_3M_dino_v2_states_2of2.npy
# - local_narr_captions.json (optional)

# You can use RunPod's file upload or wget/curl to download
```

## 5. Test Integration (CPU first)

```bash
# Test basic functionality
python test_larimar_integration.py

# Test with small batch on CPU
python train_larimar_babylm.py --devices 0 --accelerator cpu --batch_size 2 --max_epochs 1
```

## 6. Full GPU Training

### Option A: Command Line
```bash
# Basic training
python train_larimar_babylm.py --config configs/config_larimar_babylm.yaml

# Custom settings
python train_larimar_babylm.py \
    --batch_size 16 \
    --max_epochs 10 \
    --learning_rate 1e-4 \
    --devices 1 \
    --accelerator gpu

# With Weights & Biases logging
python train_larimar_babylm.py \
    --config configs/config_larimar_babylm.yaml \
    --logger wandb \
    --project_name "tiny-larimar" \
    --run_name "runpod-training"
```

### Option B: VS Code Integration
1. **Open project in VS Code** (connected to RunPod)
2. **Install Python extension** in VS Code
3. **Select Python interpreter**: `/usr/bin/python3`
4. **Use integrated terminal** in VS Code for training commands

## 7. Monitor Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006

# Access via: http://<runpod-ip>:6006
```

### Weights & Biases
```bash
# Login to W&B
wandb login

# Training will automatically log to W&B dashboard
```

## 8. Save and Download Models

```bash
# Models are saved in outputs/
ls outputs/

# Download specific checkpoint
# Use RunPod file manager or SCP:
scp -P <port> root@<ip>:/workspace/Tiny-MultiModal-Larimar/outputs/model.ckpt ./
```

## 9. Resume Training

```bash
# Resume from checkpoint
python train_larimar_babylm.py \
    --config configs/config_larimar_babylm.yaml \
    --resume_from_checkpoint outputs/lightning_logs/version_0/checkpoints/last.ckpt
```

## 10. Troubleshooting

### Common Issues:
- **CUDA out of memory**: Reduce batch_size in config
- **Import errors**: Check Python path and dependencies
- **Data loading errors**: Verify data files are in correct location

### Debug Commands:
```bash
# Check GPU
nvidia-smi

# Check Python packages
pip list | grep torch

# Test individual components
python -c "from src.modules.larimar_multimodal_vae import LarimarMultiModalVAE; print('✓ Import successful')"
```

## Example Full Training Session

```bash
# 1. Connect to RunPod via VS Code SSH
# 2. Set up environment (as above)
# 3. Upload data to ../babylm_dataset/
# 4. Test integration
python test_larimar_integration.py

# 5. Start training
python train_larimar_babylm.py \
    --config configs/config_larimar_babylm.yaml \
    --batch_size 12 \
    --max_epochs 10 \
    --devices 1 \
    --accelerator gpu \
    --logger wandb

# 6. Monitor via TensorBoard
tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006
```

This setup gives you a complete development environment on RunPod with GPU acceleration, accessible through VS Code!
