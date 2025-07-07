import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import lightning as L
from datasets import load_dataset
import requests
from io import BytesIO
import base64


class MultiModalDataset(Dataset):
    """
    Dataset for multimodal vision-language data
    """

    def __init__(self,
                 data_path: str,
                 tokenizer_name: str = "distilbert-base-uncased",
                 image_processor_name: str = "facebook/dinov2-base",
                 max_length: int = 512,
                 image_size: int = 224,
                 mode: str = "multimodal"):  # "vision", "text", "multimodal"

        self.data_path = data_path
        self.max_length = max_length
        self.image_size = image_size
        self.mode = mode

        # Load tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from various sources"""
        if self.data_path.endswith('.json'):
            return self._load_json_data()
        elif self.data_path.endswith('.jsonl'):
            return self._load_jsonl_data()
        elif os.path.isdir(self.data_path):
            return self._load_directory_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")

    def _load_json_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _load_jsonl_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _load_directory_data(self) -> List[Dict[str, Any]]:
        """Load data from directory structure"""
        data = []

        # Look for image-caption pairs
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    # Look for corresponding caption file
                    caption_path = os.path.join(root, file.replace(
                        '.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))

                    if os.path.exists(caption_path):
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()

                        data.append({
                            'image_path': image_path,
                            'caption': caption
                        })

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        result = {}

        # Process text
        if 'caption' in item or 'text' in item:
            text = item.get('caption', item.get('text', ''))

            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)

            # Create labels (same as input_ids for language modeling)
            result['labels'] = result['input_ids'].clone()
            # Set pad tokens to -100 to ignore in loss
            result['labels'][result['attention_mask'] == 0] = -100

        # Process image
        if 'image_path' in item or 'image' in item:
            if 'image_path' in item:
                image = Image.open(item['image_path']).convert('RGB')
            elif 'image' in item:
                # Handle different image formats
                if isinstance(item['image'], str):
                    # Base64 encoded image
                    image_data = base64.b64decode(item['image'])
                    image = Image.open(BytesIO(image_data)).convert('RGB')
                elif isinstance(item['image'], np.ndarray):
                    # NumPy array
                    image = Image.fromarray(item['image']).convert('RGB')
                else:
                    # Assume PIL Image
                    image = item['image'].convert('RGB')

            # Process image
            image_inputs = self.image_processor(image, return_tensors='pt')
            result['pixel_values'] = image_inputs['pixel_values'].squeeze(0)

        # Handle precomputed visual embeddings
        if 'visual_embedding' in item:
            result['visual_embedding'] = torch.tensor(
                item['visual_embedding'], dtype=torch.float32)

        return result


class BabyLMDataset(Dataset):
    """
    Dataset for BabyLM text data
    """

    def __init__(self,
                 data_path: str,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 stride: int = 256):

        self.data_path = data_path
        self.max_length = max_length
        self.stride = stride

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and process data
        self.examples = self._load_and_process_data()

    def _load_and_process_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load and process text data"""
        examples = []

        # Read text file
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize entire text
        tokens = self.tokenizer(text, return_tensors='pt')[
            'input_ids'].squeeze(0)

        # Create sliding window examples
        for i in range(0, len(tokens) - self.max_length + 1, self.stride):
            input_ids = tokens[i:i + self.max_length]
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()

            examples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class ConceptualCaptionsDataset(Dataset):
    """
    Dataset for Conceptual Captions data
    """

    def __init__(self,
                 embeddings_path: str,
                 captions_path: str,
                 tokenizer_name: str = "distilbert-base-uncased",
                 max_length: int = 512,
                 max_samples: Optional[int] = None):

        self.embeddings_path = embeddings_path
        self.captions_path = captions_path
        self.max_length = max_length
        self.max_samples = max_samples

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load data
        self.visual_embeddings = np.load(embeddings_path)
        with open(captions_path, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        # Ensure same length
        assert len(self.visual_embeddings) == len(self.captions), \
            f"Mismatch: {len(self.visual_embeddings)} embeddings vs {len(self.captions)} captions"

        # Limit samples if specified
        if max_samples is not None:
            self.visual_embeddings = self.visual_embeddings[:max_samples]
            self.captions = self.captions[:max_samples]

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        caption = self.captions[idx]
        visual_embedding = self.visual_embeddings[idx]

        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),
            'visual_embedding': torch.tensor(visual_embedding, dtype=torch.float32)
        }


class MultiModalDataModule(L.LightningDataModule):
    """
    Lightning data module for multimodal data
    """

    def __init__(self,
                 # Data paths
                 train_data_path: str,
                 val_data_path: Optional[str] = None,
                 test_data_path: Optional[str] = None,

                 # Model parameters
                 tokenizer_name: str = "distilbert-base-uncased",
                 image_processor_name: str = "facebook/dinov2-base",

                 # Data parameters
                 max_length: int = 512,
                 image_size: int = 224,
                 mode: str = "multimodal",

                 # DataLoader parameters
                 batch_size: int = 16,
                 num_workers: int = 4,
                 pin_memory: bool = True,

                 # Optional parameters
                 max_samples: Optional[int] = None,
                 data_type: str = "multimodal"):  # "multimodal", "babylm", "conceptual"

        super().__init__()

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path

        self.tokenizer_name = tokenizer_name
        self.image_processor_name = image_processor_name

        self.max_length = max_length
        self.image_size = image_size
        self.mode = mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.max_samples = max_samples
        self.data_type = data_type

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = self._create_dataset(self.train_data_path)

            # Validation dataset
            if self.val_data_path:
                self.val_dataset = self._create_dataset(self.val_data_path)
            else:
                # Split training data
                train_size = int(0.9 * len(self.train_dataset))
                val_size = len(self.train_dataset) - train_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_size, val_size]
                )

        if stage == "test" or stage is None:
            # Test dataset
            if self.test_data_path:
                self.test_dataset = self._create_dataset(self.test_data_path)
            else:
                self.test_dataset = self.val_dataset

    def _create_dataset(self, data_path: str) -> Dataset:
        """Create dataset based on data type"""
        if self.data_type == "multimodal":
            return MultiModalDataset(
                data_path=data_path,
                tokenizer_name=self.tokenizer_name,
                image_processor_name=self.image_processor_name,
                max_length=self.max_length,
                image_size=self.image_size,
                mode=self.mode
            )
        elif self.data_type == "babylm":
            return BabyLMDataset(
                data_path=data_path,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length
            )
        elif self.data_type == "conceptual":
            # Assume data_path contains both embeddings and captions
            embeddings_path = data_path.replace('.json', '.npy')
            return ConceptualCaptionsDataset(
                embeddings_path=embeddings_path,
                captions_path=data_path,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_length,
                max_samples=self.max_samples
            )
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")

    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Determine which keys are present
        keys = set()
        for item in batch:
            keys.update(item.keys())

        # Create batched dict
        batched = {}
        for key in keys:
            # Only include items that have this key
            values = [item[key] for item in batch if key in item]
            if values:
                batched[key] = torch.stack(values)

        return batched


def download_babylm_data(save_path: str):
    """Download BabyLM data"""
    import zipfile
    import urllib.request

    url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="

    print(f"Downloading BabyLM data from {url}")

    # Create directory
    os.makedirs(save_path, exist_ok=True)

    # Download zip file
    zip_path = os.path.join(save_path, "babylm_data.zip")
    urllib.request.urlretrieve(url, zip_path)

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    # Remove zip file
    os.remove(zip_path)

    print(f"BabyLM data downloaded and extracted to {save_path}")


def create_dummy_multimodal_data(save_path: str, num_samples: int = 100):
    """Create dummy multimodal data for testing"""
    import random

    os.makedirs(save_path, exist_ok=True)

    # Create dummy data
    dummy_data = []
    captions = [
        "A person walking down the street",
        "A cat sitting on a chair",
        "A beautiful sunset over the ocean",
        "Children playing in the park",
        "A car driving on the highway",
        "A dog running through the grass",
        "A bird flying in the sky",
        "A flower blooming in the garden",
        "A mountain covered in snow",
        "A river flowing through the forest"
    ]

    for i in range(num_samples):
        # Create dummy visual embedding (DiNOv2 size)
        visual_embedding = np.random.randn(768).astype(np.float32)

        # Random caption
        caption = random.choice(captions)

        dummy_data.append({
            'visual_embedding': visual_embedding.tolist(),
            'caption': caption
        })

    # Save data
    with open(os.path.join(save_path, 'dummy_data.json'), 'w') as f:
        json.dump(dummy_data, f, indent=2)

    print(f"Created {num_samples} dummy multimodal samples in {save_path}")


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Create dummy data
    create_dummy_multimodal_data("./dummy_data", num_samples=50)

    # Test dataset
    dataset = MultiModalDataset(
        data_path="./dummy_data/dummy_data.json",
        mode="multimodal"
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Test data module
    data_module = MultiModalDataModule(
        train_data_path="./dummy_data/dummy_data.json",
        data_type="multimodal",
        batch_size=4
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()

    print(f"Training batches: {len(train_loader)}")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print("Data loading test completed!")
