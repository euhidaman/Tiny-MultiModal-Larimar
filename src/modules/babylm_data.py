import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pytorch_lightning as pl
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

# BabyLM Dataset URLs - Updated for 2025 OSF repository
# The actual dataset is available as a zip file from OSF
BABYLM_ZIP_URL = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="


def download_babylm_data(data_path: str, dataset_type: str = "cc_3M", force_download: bool = False):
    """Download BabyLM multimodal dataset from OSF zip file"""
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

    # Check if all files already exist
    all_exist = all((data_path / filename).exists() for filename in required_files)
    
    if all_exist and not force_download:
        logger.info("All required files already exist, skipping download")
        for filename in required_files:
            filepath = data_path / filename
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  {filename}: {size_mb:.1f} MB")
        return

    logger.info("Downloading BabyLM dataset from OSF...")
    logger.info("This will download and extract the full dataset (~3GB)")
    
    zip_path = data_path / "babylm_dataset.zip"
    
    try:
        # Download the zip file
        logger.info(f"Downloading from: {BABYLM_ZIP_URL}")
        response = requests.get(BABYLM_ZIP_URL, stream=True, verify=False)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(
                desc="Downloading BabyLM dataset",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info("Download completed, extracting files...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in the zip
            file_list = zip_ref.namelist()
            logger.info(f"Zip contains {len(file_list)} files")
            
            # Extract only the files we need
            extracted_count = 0
            for file_info in zip_ref.filelist:
                # Check if this file is one we need
                filename = Path(file_info.filename).name
                if filename in required_files:
                    # Extract to the data directory
                    target_path = data_path / filename
                    with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    logger.info(f"Extracted: {filename}")
                    extracted_count += 1
            
            if extracted_count == 0:
                # If no direct matches, try to find files in subdirectories
                logger.info("Direct file matches not found, searching in subdirectories...")
                for file_info in zip_ref.filelist:
                    for required_file in required_files:
                        if file_info.filename.endswith(required_file):
                            target_path = data_path / required_file
                            with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                                target.write(source.read())
                            logger.info(f"Extracted: {required_file} from {file_info.filename}")
                            extracted_count += 1
                            break
        
        # Clean up zip file
        zip_path.unlink()
        
        # Verify extraction
        missing_files = []
        for filename in required_files:
            filepath = data_path / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"✅ {filename}: {size_mb:.1f} MB")
            else:
                missing_files.append(filename)
                logger.error(f"❌ {filename}: Not found after extraction")
        
        if missing_files:
            raise RuntimeError(f"Failed to extract required files: {missing_files}")
        
        logger.info(f"Dataset download and extraction completed successfully!")
        logger.info(f"Dataset setup complete for {dataset_type}")
        
    except Exception as e:
        logger.error(f"Failed to download/extract dataset: {e}")
        # Clean up partial files
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Dataset download failed: {e}")


def test_babylm_connectivity():
    """Test if BabyLM OSF repository is accessible"""
    try:
        response = requests.head(BABYLM_ZIP_URL, timeout=10, verify=False)
        return response.status_code == 200
    except:
        return False


# Dummy data creation functions removed - no fallback to dummy data
# Real dataset download is required for training


# All dummy data creation functions have been removed
# Only real dataset downloads are supported


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


class BabyLMMultiModalDataModule(pl.LightningDataModule):
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
