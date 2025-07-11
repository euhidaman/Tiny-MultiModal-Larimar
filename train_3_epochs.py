from src.modules.larimar_multimodal_vae import LarimarMultiModalConfig
from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
from src.modules.babylm_data import BabyLMMultiModalDataModule
import pytorch_lightning as pl
import torch
import sys
sys.path.append("src")


# Quick setup
torch.set_float32_matmul_precision('medium')

# Small model config
config = LarimarMultiModalConfig(
    text_model_name="bert-base-uncased",
    decoder_model_name="gpt2",
    vision_model_name="facebook/dinov2-base",
    text_latent_size=256,
    vision_latent_size=256,
    memory_size=128,
    max_length=128,
    kl_weight=1.0,
    memory_weight=0.1
)

# Data
data_module = BabyLMMultiModalDataModule(
    data_path="data/babylm",
    batch_size=16,
    max_length=128,
    num_workers=2
)

# Model
model = LarimarBabyLMLightningModel(
    config=config,
    learning_rate=1e-4,
    max_epochs=3
)

# Simple trainer - NO CHECKPOINTS
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    precision="32",
    enable_checkpointing=False,  # No checkpoints to save space!
    logger=False,  # No logging to save space
    enable_progress_bar=True,
    val_check_interval=0.5
)

print("Training for 3 epochs with NO checkpoints...")
trainer.fit(model, data_module)
print("3-epoch training completed!!")
