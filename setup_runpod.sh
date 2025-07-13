#!/bin/bash
# setup_runpod.sh - Setup script for RunPod environment

echo "🚀 Setting up Tiny-MultiModal-Larimar on RunPod..."

# Update system
echo "📦 Updating system packages..."
apt update && apt install -y git wget curl zip unzip

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets lightning accelerate
pip install numpy pandas tqdm wandb weave
pip install PyYAML scikit-learn requests

# Setup W&B with your credentials
echo "🔧 Setting up Weights & Biases..."
export WANDB_API_KEY="5fba3726e4e32540d9fcba403f880dfaad983051"
wandb login --relogin 5fba3726e4e32540d9fcba403f880dfaad983051

# Check GPU
echo "🔧 Checking GPU availability..."
nvidia-smi

# Test PyTorch GPU
echo "🧪 Testing PyTorch GPU support..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}' if torch.cuda.is_available() else 'No GPU')"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p /workspace/babylm_dataset
mkdir -p /workspace/Tiny-MultiModal-Larimar/outputs

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your project files to /workspace/Tiny-MultiModal-Larimar/"
echo "2. Upload BabyLM data to /workspace/babylm_dataset/"
echo "3. Run: python test_larimar_integration.py"
echo "4. Start training: python train_larimar_babylm.py"
echo ""
echo "W&B is configured with team: babylm-ntust"
echo "Run names will auto-increment: baby-larimar1, baby-larimar2, etc."
