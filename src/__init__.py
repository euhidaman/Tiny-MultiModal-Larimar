"""
Tiny MultiModal Larimar - A smaller, cognitively-inspired multimodal model
based on Larimar architecture with episodic memory control.
"""

from .modules.multimodal_vae import TinyMultiModalVAE
from .modules.lightning_model import TinyMultiModalLitModel
from .modules.data import MultiModalDataModule
from .modules.vision_encoder import DiNOv2VisionEncoder
from .modules.text_encoder import DistilBERTTextEncoder
from .modules.decoder import DistilGPT2Decoder
from .modules.memory import TinyMemory

__version__ = "0.1.0"
__author__ = "Tiny-MultiModal-Larimar Team"

__all__ = [
    "TinyMultiModalVAE",
    "TinyMultiModalLitModel",
    "MultiModalDataModule",
    "DiNOv2VisionEncoder",
    "DistilBERTTextEncoder",
    "DistilGPT2Decoder",
    "TinyMemory"
]
