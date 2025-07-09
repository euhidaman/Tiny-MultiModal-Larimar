"""
Tiny MultiModal Larimar - A smaller, cognitively-inspired multimodal model
based on Larimar architecture with episodic memory control.
"""

from .modules.larimar_multimodal_vae import LarimarMultiModalVAE, LarimarMultiModalConfig
from .modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
from .modules.babylm_data import BabyLMMultiModalDataModule, BabyLMMultiModalDataset
from .modules.vision_encoder import DiNOv2VisionEncoder
from .modules.larimar_text_encoder import LarimarTextEncoder, BertForLatentConnector
from .modules.larimar_gpt2_decoder import LarimarGPT2Decoder, GPT2ForLatentConnector
from .modules.larimar_memory import TinyLarimarMemory, LarimarMemoryVAE

__version__ = "0.1.0"
__author__ = "Tiny-MultiModal-Larimar Team"

__all__ = [
    "LarimarMultiModalVAE",
    "LarimarMultiModalConfig",
    "LarimarBabyLMLightningModel",
    "BabyLMMultiModalDataModule",
    "BabyLMMultiModalDataset",
    "DiNOv2VisionEncoder",
    "LarimarTextEncoder",
    "BertForLatentConnector",
    "LarimarGPT2Decoder",
    "GPT2ForLatentConnector",
    "TinyLarimarMemory",
    "LarimarMemoryVAE"
]
