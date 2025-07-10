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
import ssl
import urllib.request

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
    """Download a file with progress bar, handling SSL issues"""
    if filepath.exists() and not force_download:
        logger.info(f"File {filepath.name} already exists, skipping download")
        return

    logger.info(f"Downloading {filepath.name} from {url}")

    try:
        # First try with requests
        # Disable SSL verification
        response = requests.get(url, stream=True, verify=False)
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

    except Exception as e1:
        logger.warning(
            f"Failed with requests: {e1}, trying urllib with SSL context")
        try:
            # Try with urllib and custom SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with urllib.request.urlopen(url, context=ssl_context) as response:
                total_size = int(response.headers.get('content-length', 0))

                with open(filepath, 'wb') as f:
                    with tqdm(
                        desc=filepath.name,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded {filepath.name}")

        except Exception as e2:
            logger.error(f"Error downloading {filepath.name}: {e2}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file

            # Create a warning but don't fail completely
            logger.warning(
                f"Could not download {filepath.name}. You may need to download manually from {url}")

            # Create a dummy file with instructions
            with open(filepath.with_suffix('.download_instructions.txt'), 'w') as f:
                f.write(f"Failed to download {filepath.name}\n")
                f.write(f"Please download manually from: {url}\n")
                f.write(f"Save to: {filepath}\n")

            raise RuntimeError(
                f"Could not download {filepath.name}. Manual download required from {url}")


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

    download_errors = []
    for filename in required_files:
        url = BABYLM_URLS[filename]
        filepath = data_path / filename
        try:
            download_file(url, filepath, force_download)
        except Exception as e:
            download_errors.append((filename, str(e)))

    if download_errors:
        logger.warning(
            "Some files failed to download. Checking if we can create dummy data...")

        # Check if we already have some files
        existing_files = [
            f for f in required_files if (data_path / f).exists()]

        if len(existing_files) == 0:
            logger.warning(
                "No files downloaded successfully. Creating minimal dummy data for testing...")
            create_dummy_data(data_path, dataset_type)
        else:
            logger.info(
                f"Found {len(existing_files)} existing files, proceeding with available data")

    logger.info(f"Dataset setup complete for {dataset_type}")


def create_dummy_data(data_path: Path, dataset_type: str = "cc_3M"):
    """Create minimal dummy data for testing when downloads fail"""
    logger.info("Creating dummy data for testing purposes")

    if dataset_type == "cc_3M":
        # Create dummy captions
        dummy_captions = [
            {"caption": "A small brown dog sits on green grass"},
            {"caption": "A red car drives down a busy street"},
            {"caption": "Children play in a colorful playground"},
            {"caption": "A white cat sleeps on a soft couch"},
            {"caption": "Mountains rise behind a clear blue lake"}
        ] * 100  # 500 samples

        with open(data_path / "cc_3M_captions.json", 'w') as f:
            json.dump(dummy_captions, f)

        # Create dummy DiNOv2 embeddings (768-dimensional for DiNOv2-base)
        dummy_embeddings_1 = np.random.normal(
            0, 0.5, (250, 768)).astype(np.float32)
        dummy_embeddings_2 = np.random.normal(
            0, 0.5, (250, 768)).astype(np.float32)

        np.save(data_path / "cc_3M_dino_v2_states_1of2.npy", dummy_embeddings_1)
        np.save(data_path / "cc_3M_dino_v2_states_2of2.npy", dummy_embeddings_2)

        logger.info("Created dummy cc_3M dataset with 500 samples")

    elif dataset_type == "local_narr":
        # Create dummy captions for local narratives
        dummy_captions = [
            {"caption": "Person walks across the room slowly"},
            {"caption": "Hand reaches for the blue object"},
            {"caption": "Camera pans left to show the window"}
        ] * 50  # 150 samples

        with open(data_path / "local_narr_captions.json", 'w') as f:
            json.dump(dummy_captions, f)

        # Create dummy DiNOv2 embeddings (768-dimensional)
        dummy_embeddings = np.random.normal(
            0, 0.5, (150, 768)).astype(np.float32)
        np.save(data_path / "local_narr_dino_v2_states.npy", dummy_embeddings)

        logger.info("Created dummy local_narr dataset with 150 samples")

    # Create a readme file explaining the dummy data
    readme_content = f"""
# Dummy Dataset Notice

This directory contains dummy data created automatically because the original
BabyLM multimodal dataset could not be downloaded.

## What's in the dummy data:
- Synthetic captions (repeated patterns)
- Random DiNOv2 embeddings (384-dimensional)
- Compatible with the training pipeline for testing

## To use real data:
1. Manually download the files from:
   - https://data.babylm.github.io/multimodal/
2. Replace the dummy files with real ones
3. Re-run training

Dataset type: {dataset_type}
Created automatically by Tiny-MultiModal-Larimar
"""

    with open(data_path / "README_DUMMY_DATA.txt", 'w') as f:
        f.write(readme_content)

    logger.warning(
        "IMPORTANT: Using dummy data for testing. Download real data for actual training!")


def create_dummy_babylm_data(data_path: Path, dataset_type: str = "cc_3M"):
    """Create dummy BabyLM data for testing when real data isn't available (e.g., during website maintenance)"""
    logger.info(
        f"Creating dummy {dataset_type} data for testing/development...")
    logger.info(
        "This allows you to test the training pipeline while the BabyLM website is under maintenance")

    data_path.mkdir(parents=True, exist_ok=True)

    if dataset_type == "cc_3M":
        # Create dummy captions with realistic content
        dummy_captions = []
        sample_captions = [
            "A red car driving down a busy street in the city",
            "Children playing soccer in a green park on a sunny day",
            "A cat sitting on a windowsill looking outside",
            "People walking across a bridge over a river",
            "A dog running on the beach near ocean waves",
            "A person reading a book under a tree in summer",
            "Birds flying over mountains covered in snow",
            "A train traveling through a countryside landscape",
            "Students studying together in a library",
            "A chef cooking delicious food in a restaurant kitchen",
            "Flowers blooming in a beautiful garden during spring",
            "A lighthouse standing tall on a rocky coastline",
            "Children building sandcastles on a sandy beach",
            "A cyclist riding through a forest trail",
            "Musicians performing on stage at a concert"
        ]

        # Create 1000 dummy samples by repeating and varying the base captions
        for i in range(1000):
            base_caption = sample_captions[i % len(sample_captions)]
            # Add some variation to make it more realistic
            variations = [
                f"Photo of {base_caption.lower()}",
                f"Image showing {base_caption.lower()}",
                f"Picture of {base_caption.lower()}",
                base_caption,
                f"Scene with {base_caption.lower()}",
                f"View of {base_caption.lower()}"
            ]
            dummy_captions.append({"caption": variations[i % len(variations)]})

        # Save captions
        captions_file = data_path / "cc_3M_captions.json"
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_captions, f, indent=2)
        logger.info(f"Created {len(dummy_captions)} dummy captions")

        # Create dummy DiNOv2 embeddings (768 dimensions for DiNOv2-base)
        embedding_dim = 768  # DiNOv2-base embedding dimension
        # Use a more realistic distribution (closer to actual DiNOv2 outputs)
        embeddings1 = np.random.normal(
            0, 0.5, (500, embedding_dim)).astype(np.float32)
        embeddings2 = np.random.normal(
            0, 0.5, (500, embedding_dim)).astype(np.float32)

        np.save(data_path / "cc_3M_dino_v2_states_1of2.npy", embeddings1)
        np.save(data_path / "cc_3M_dino_v2_states_2of2.npy", embeddings2)
        logger.info(
            f"Created dummy embeddings: {embeddings1.shape} + {embeddings2.shape}")

    elif dataset_type == "local_narr":
        # Create dummy local narratives data
        dummy_captions = []
        for i in range(500):
            dummy_captions.append({
                "caption": f"Local narrative {i+1}: A detailed scene description with various objects, people, and activities taking place in different environments"
            })

        captions_file = data_path / "local_narr_captions.json"
        with open(captions_file, 'w', encoding='utf-8') as f:
            json.dump(dummy_captions, f, indent=2)

        # Create dummy embeddings with realistic distribution
        embedding_dim = 768
        embeddings = np.random.normal(
            0, 0.5, (500, embedding_dim)).astype(np.float32)
        np.save(data_path / "local_narr_dino_v2_states.npy", embeddings)
        logger.info(
            f"Created dummy local narratives data: {len(dummy_captions)} captions")

    logger.info(
        "Dummy data creation completed! You can now test the training pipeline.")
    logger.info(
        "Note: Replace with real data when the BabyLM website is back online.")


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
