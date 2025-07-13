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
import tarfile
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
        # Download the zip file with proper headers for OSF
        logger.info(f"Downloading from: {BABYLM_ZIP_URL}")
        
        # Add headers to handle OSF properly
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/zip, application/octet-stream, */*'
        }
        
        response = requests.get(BABYLM_ZIP_URL, stream=True, verify=False, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Check if we actually got a zip file
        content_type = response.headers.get('content-type', '')
        logger.info(f"Content-Type: {content_type}")
        
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
        
        logger.info("Download completed, checking file type...")
        
        # Check file size and first bytes for debugging
        file_size = zip_path.stat().st_size
        logger.info(f"Downloaded file size: {file_size / (1024*1024):.1f} MB")
        
        with open(zip_path, 'rb') as f:
            first_bytes = f.read(50)
            logger.info(f"File header (first 50 bytes): {first_bytes}")
        
        # Check if the downloaded file is actually a zip
        try:
            with zipfile.ZipFile(zip_path, 'r') as test_zip:
                file_list = test_zip.namelist()
                file_count = len(file_list)
                logger.info(f"Valid zip file with {file_count} files")
                
                # Log first few files for debugging
                logger.info("First 10 files in zip:")
                for i, filename in enumerate(file_list[:10]):
                    info = test_zip.getinfo(filename)
                    logger.info(f"  {i+1}. {filename} ({info.file_size} bytes)")
                
        except zipfile.BadZipFile as e:
            logger.error(f"BadZipFile error: {e}")
            
            # Try to detect if it's HTML (redirect page)
            try:
                with open(zip_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline()
                    if 'html' in first_line.lower():
                        logger.error("Downloaded file appears to be HTML, not data")
                        raise RuntimeError("Downloaded HTML instead of zip file - URL may be incorrect")
            except:
                pass
            
            # Try different extraction methods
            logger.info("Trying alternative extraction methods...")
            
            # Try with tarfile (in case it's a tar.gz)
            import tarfile
            try:
                with tarfile.open(zip_path, 'r:*') as tar:
                    tar_files = tar.getnames()
                    logger.info(f"Valid tar file with {len(tar_files)} files")
                    
                    # Extract required files from tar
                    extracted_count = 0
                    for member in tar.getmembers():
                        if member.isfile():
                            filename = Path(member.name).name
                            if filename in required_files:
                                target_path = data_path / filename
                                with tar.extractfile(member) as source:
                                    with open(target_path, 'wb') as target:
                                        target.write(source.read())
                                logger.info(f"Extracted from tar: {filename}")
                                extracted_count += 1
                    
                    if extracted_count > 0:
                        logger.info(f"Successfully extracted {extracted_count} files from tar")
                        zip_path.unlink()  # Clean up
                        # Skip to verification
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
                        return
                        
            except Exception as tar_e:
                logger.warning(f"Tar extraction also failed: {tar_e}")
            
            raise RuntimeError(f"Downloaded file is not a valid zip or tar file: {e}")
        
        logger.info("Extracting files from zip...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Processing {len(file_list)} files from zip")
            
            # Extract only the files we need
            extracted_count = 0
            
            # First try: exact filename matches
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():  # Skip directories
                    filename = Path(file_info.filename).name
                    if filename in required_files:
                        target_path = data_path / filename
                        logger.info(f"Extracting {filename} from {file_info.filename}")
                        with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                            target.write(source.read())
                        logger.info(f"✅ Extracted: {filename} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
                        extracted_count += 1
            
            # Second try: if no direct matches, search in subdirectories
            if extracted_count == 0:
                logger.info("No direct filename matches, searching in subdirectories...")
                for file_info in zip_ref.filelist:
                    if not file_info.is_dir():
                        for required_file in required_files:
                            if file_info.filename.endswith(required_file):
                                target_path = data_path / required_file
                                logger.info(f"Extracting {required_file} from {file_info.filename}")
                                with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                                    target.write(source.read())
                                logger.info(f"✅ Extracted: {required_file} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
                                extracted_count += 1
                                break
            
            # Third try: case-insensitive search
            if extracted_count == 0:
                logger.info("No matches found, trying case-insensitive search...")
                for file_info in zip_ref.filelist:
                    if not file_info.is_dir():
                        filename_lower = Path(file_info.filename).name.lower()
                        for required_file in required_files:
                            if filename_lower == required_file.lower():
                                target_path = data_path / required_file
                                logger.info(f"Extracting {required_file} from {file_info.filename} (case-insensitive match)")
                                with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                                    target.write(source.read())
                                logger.info(f"✅ Extracted: {required_file} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
                                extracted_count += 1
                                break
            
            if extracted_count == 0:
                # Log all files for debugging
                logger.error("No required files found in zip. Contents:")
                for i, filename in enumerate(file_list):
                    if i < 20:  # Show first 20 files
                        logger.error(f"  {filename}")
                    elif i == 20:
                        logger.error(f"  ... and {len(file_list)-20} more files")
                        break
                
                raise RuntimeError(f"None of the required files found in zip: {required_files}")
            
            logger.info(f"Successfully extracted {extracted_count}/{len(required_files)} required files")
        
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
        
        # Provide helpful troubleshooting information
        logger.error("Dataset download failed. Troubleshooting options:")
        logger.error("1. Check internet connection")
        logger.error("2. Try manual download from: https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip=")
        logger.error("3. Extract manually and place files in: ../babylm_dataset/")
        logger.error(f"4. Required files: {required_files}")
        logger.error("5. Or use force_download=True to retry")
        
        raise RuntimeError(f"Dataset download failed: {e}")


def manual_download_instructions():
    """Print manual download instructions"""
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("If automatic download fails, please download manually:")
    print("\n1. Download from: https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip=")
    print("2. Extract the zip file")
    print("3. Find these files and copy to ../babylm_dataset/:")
    print("   - cc_3M_captions.json")
    print("   - cc_3M_dino_v2_states_1of2.npy") 
    print("   - cc_3M_dino_v2_states_2of2.npy")
    print("4. Run the script again")
    print("="*60)


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
            try:
                download_babylm_data(
                    self.data_path, self.dataset_type, self.force_download)
            except Exception as e:
                logger.error(f"Automatic download failed: {e}")
                manual_download_instructions()
                raise

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
                 data_path: str = "../babylm_dataset",
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
