import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lightning as L
from pathlib import Path
import os
import requests
import zipfile
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BabyLM Dataset URLs
BABYLM_URLS = {
    "cc_3M_captions.json": "https://data.babylm.github.io/multimodal/cc_3M_captions.json",
    "cc_3M_dino_v2_states_1of2.npy": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_1of2.npy",
    "cc_3M_dino_v2_states_2of2.npy": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_2of2.npy",
    "local_narr_captions.json": "https://data.babylm.github.io/multimodal/local_narr_captions.json",
    "local_narr_dino_v2_states.npy": "https://data.babylm.github.io/multimodal/local_narr_dino_v2_states.npy"
}


def download_file(url: str, filepath: Path, force_download: bool = False):
    """Download a file with progress bar"""
    if filepath.exists() and not force_download:
        logger.info(f"File {filepath.name} already exists, skipping download")
        return

    logger.info(f"Downloading {filepath.name} from {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            with tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.info(f"Successfully downloaded {filepath.name}")

    except Exception as e:
        logger.error(f"Error downloading {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial file
        raise


def download_babylm_data(data_path: str, dataset_type: str = "cc_3M", force_download: bool = False):
    """Download BabyLM multimodal dataset"""
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    if dataset_type == "cc_3M":
        required_files = [
            "cc_3M_captions.json",
            "cc_3M_dino_v2_states_1of2.npy",
            "cc_3M_dino_v2_states_2of2.npy"
        ]
    elif dataset_type == "local_narr":
        required_files = [
            "local_narr_captions.json",
            "local_narr_dino_v2_states.npy"
        ]
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    for filename in required_files:
        url = BABYLM_URLS[filename]
        filepath = data_path / filename
        download_file(url, filepath, force_download)

    logger.info(f"Dataset download complete for {dataset_type}")


class BabyLMMultiModalDataset(Dataset):
    """
    Dataset for BabyLM multimodal data with pre-computed DiNOv2 embeddings
    """

    def __init__(self,
                 data_path: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 dataset_type: str = "cc_3M",  # "cc_3M" or "local_narr"
                 auto_download: bool = True,
                 force_download: bool = False):

        self.data_path = Path(data_path)
        self.max_length = max_length
        self.dataset_type = dataset_type
        self.auto_download = auto_download
        self.force_download = force_download

        # Load tokenizer (BERT-based for Larimar compatibility)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add padding token if not present (BERT usually has [PAD])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'

        # Download data if needed
        if self.auto_download:
            download_babylm_data(
                self.data_path, self.dataset_type, self.force_download)

        # Load data
        self.captions, self.embeddings = self._load_data()

    def _load_data(self) -> Tuple[List[Dict], np.ndarray]:
        """Load captions and embeddings"""

        if self.dataset_type == "cc_3M":
            # Load Conceptual Captions data
            caption_file = self.data_path / "cc_3M_captions.json"
            embeddings_file1 = self.data_path / "cc_3M_dino_v2_states_1of2.npy"
            embeddings_file2 = self.data_path / "cc_3M_dino_v2_states_2of2.npy"

            # Load captions
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)

            # Load and concatenate embeddings
            embeddings1 = np.load(embeddings_file1)
            embeddings2 = np.load(embeddings_file2)
            embeddings = np.concatenate([embeddings1, embeddings2], axis=0)

        elif self.dataset_type == "local_narr":
            # Load Local Narratives data
            caption_file = self.data_path / "local_narr_captions.json"
            embeddings_file = self.data_path / "local_narr_dino_v2_states.npy"

            # Load captions
            with open(caption_file, 'r', encoding='utf-8') as f:
                captions = json.load(f)

            # Load embeddings
            embeddings = np.load(embeddings_file)

        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        print(
            f"Loaded {len(captions)} captions and {embeddings.shape[0]} embeddings")

        # Ensure we have matching counts
        min_count = min(len(captions), embeddings.shape[0])
        captions = captions[:min_count]
        embeddings = embeddings[:min_count]

        print(f"Using {len(captions)} samples")

        return captions, embeddings

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        caption_data = self.captions[idx]

        # Extract caption text (handle different possible formats)
        if isinstance(caption_data, dict):
            caption = caption_data.get(
                'caption', caption_data.get('text', str(caption_data)))
        else:
            caption = str(caption_data)

        # Tokenize caption
        tokens = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get vision embedding
        vision_embedding = torch.from_numpy(self.embeddings[idx]).float()

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': tokens['input_ids'].squeeze(0),  # For language modeling
            'vision_embedding': vision_embedding,  # Pre-computed DiNOv2 features
            'caption': caption
        }


class BabyLMMultiModalDataModule(L.LightningDataModule):
    """Lightning DataModule for BabyLM multimodal data"""

    def __init__(self,
                 data_path: str = "data/babylm",
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 512,
                 batch_size: int = 12,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 dataset_type: str = "cc_3M",
                 train_split: float = 0.9,
                 auto_download: bool = True,
                 force_download: bool = False):

        super().__init__()
        self.data_path = data_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_type = dataset_type
        self.train_split = train_split
        self.auto_download = auto_download
        self.force_download = force_download

    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""

        # Load full dataset
        full_dataset = BabyLMMultiModalDataset(
            data_path=self.data_path,
            tokenizer_name=self.tokenizer_name,
            max_length=self.max_length,
            dataset_type=self.dataset_type,
            auto_download=self.auto_download,
            force_download=self.force_download
        )

        # Split into train/val
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = total_size - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
