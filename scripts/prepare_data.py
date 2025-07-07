#!/usr/bin/env python3

"""
Utility script to download and prepare data for Tiny MultiModal Larimar
"""

import os
import argparse
import json
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import requests
from PIL import Image
from tqdm import tqdm


def download_babylm_data(save_path: str):
    """Download BabyLM data"""
    url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="

    print(f"Downloading BabyLM data from {url}")

    # Create directory
    os.makedirs(save_path, exist_ok=True)

    # Download zip file
    zip_path = os.path.join(save_path, "babylm_data.zip")

    print("Downloading zip file...")
    urllib.request.urlretrieve(url, zip_path)

    # Extract zip file
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    # Remove zip file
    os.remove(zip_path)

    print(f"BabyLM data downloaded and extracted to {save_path}")


def download_conceptual_captions_embeddings():
    """
    Download precomputed DiNOv2 embeddings for Conceptual Captions
    Note: This is a placeholder - you would need to implement actual download
    """
    print("Conceptual Captions embeddings download not implemented.")
    print("Please refer to the original Larimar paper for data sources:")
    print("1. Localized Narratives: https://google.github.io/localized-narratives/")
    print("2. Conceptual Captions: https://ai.google.com/research/ConceptualCaptions/download")
    print("3. Precomputed embeddings: Check the original repository")


def create_dummy_multimodal_data(save_path: str, num_samples: int = 1000):
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
        "A river flowing through the forest",
        "A book lying on a table",
        "A cup of coffee steaming hot",
        "A bicycle parked by the fence",
        "Rain falling on the window",
        "Stars shining in the night sky"
    ]

    print(f"Creating {num_samples} dummy multimodal samples...")

    for i in tqdm(range(num_samples)):
        # Create dummy visual embedding (DiNOv2 size: 768)
        visual_embedding = np.random.randn(768).astype(np.float32)

        # Random caption
        caption = random.choice(captions)

        # Add some variation to captions
        variations = [
            f"A photo of {caption.lower()}",
            f"An image showing {caption.lower()}",
            f"This picture depicts {caption.lower()}",
            caption,
            f"{caption} in high resolution"
        ]

        final_caption = random.choice(variations)

        dummy_data.append({
            'visual_embedding': visual_embedding.tolist(),
            'caption': final_caption,
            'id': i
        })

    # Save training data (80%)
    train_size = int(0.8 * num_samples)
    train_data = dummy_data[:train_size]
    val_data = dummy_data[train_size:]

    train_path = os.path.join(save_path, 'train_data.json')
    val_path = os.path.join(save_path, 'val_data.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f)

    with open(val_path, 'w') as f:
        json.dump(val_data, f)

    print(f"Created {len(train_data)} training samples in {train_path}")
    print(f"Created {len(val_data)} validation samples in {val_path}")


def create_dummy_image_caption_pairs(save_path: str, num_samples: int = 100):
    """Create dummy image-caption pairs with actual dummy images"""
    import random
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(save_path, exist_ok=True)
    images_dir = os.path.join(save_path, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Colors and shapes for dummy images
    colors = ['red', 'blue', 'green', 'yellow',
              'purple', 'orange', 'pink', 'brown']
    shapes = ['circle', 'rectangle', 'triangle', 'star']

    data = []

    print(f"Creating {num_samples} dummy image-caption pairs...")

    for i in tqdm(range(num_samples)):
        # Create dummy image
        img = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(img)

        # Draw random shape
        color = random.choice(colors)
        shape = random.choice(shapes)

        if shape == 'circle':
            draw.ellipse([50, 50, 174, 174], fill=color)
            caption = f"A {color} circle"
        elif shape == 'rectangle':
            draw.rectangle([50, 50, 174, 174], fill=color)
            caption = f"A {color} rectangle"
        elif shape == 'triangle':
            draw.polygon([(112, 50), (50, 174), (174, 174)], fill=color)
            caption = f"A {color} triangle"
        else:  # star
            # Simple star shape
            points = [(112, 50), (125, 85), (160, 85), (135, 110), (145, 145),
                      (112, 125), (80, 145), (90, 110), (65, 85), (100, 85)]
            draw.polygon(points, fill=color)
            caption = f"A {color} star"

        # Save image
        img_path = os.path.join(images_dir, f'image_{i:04d}.png')
        img.save(img_path)

        data.append({
            'image_path': img_path,
            'caption': caption,
            'id': i
        })

    # Save metadata
    metadata_path = os.path.join(save_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Created {num_samples} image-caption pairs in {save_path}")
    print(f"Images saved in {images_dir}")
    print(f"Metadata saved in {metadata_path}")


def prepare_babylm_text_data(data_path: str, output_path: str, max_files: int = 10):
    """Prepare BabyLM text data by concatenating files"""
    import glob

    os.makedirs(output_path, exist_ok=True)

    # Find text files
    text_files = glob.glob(os.path.join(data_path, "**/*.txt"), recursive=True)

    if not text_files:
        print(f"No text files found in {data_path}")
        return

    print(f"Found {len(text_files)} text files")

    # Limit number of files if specified
    if max_files > 0:
        text_files = text_files[:max_files]
        print(f"Using first {len(text_files)} files")

    # Concatenate all text
    all_text = []

    for file_path in tqdm(text_files, desc="Reading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_text.append(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Split into train and validation
    total_text = '\n\n'.join(all_text)
    split_point = int(0.9 * len(total_text))

    train_text = total_text[:split_point]
    val_text = total_text[split_point:]

    # Save train and validation files
    train_path = os.path.join(output_path, 'train.txt')
    val_path = os.path.join(output_path, 'val.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f"Train text saved to {train_path} ({len(train_text)} characters)")
    print(f"Validation text saved to {val_path} ({len(val_text)} characters)")


def validate_data(data_path: str, data_type: str = "multimodal"):
    """Validate data format and content"""
    print(f"Validating {data_type} data in {data_path}")

    if data_type == "multimodal":
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            print(f"Expected JSON file for multimodal data, got {data_path}")
            return False

        # Check data format
        required_keys = ['caption']
        optional_keys = ['visual_embedding', 'image_path']

        for i, item in enumerate(data[:5]):  # Check first 5 items
            print(f"Item {i}: {list(item.keys())}")

            if not any(key in item for key in required_keys):
                print(
                    f"Warning: Item {i} missing required keys: {required_keys}")

            if 'visual_embedding' in item:
                emb = np.array(item['visual_embedding'])
                print(f"  Visual embedding shape: {emb.shape}")

            if 'image_path' in item:
                img_path = item['image_path']
                if os.path.exists(img_path):
                    print(f"  Image exists: {img_path}")
                else:
                    print(f"  Image missing: {img_path}")

        print(f"Total samples: {len(data)}")

    elif data_type == "babylm":
        if not data_path.endswith('.txt'):
            print(f"Expected text file for BabyLM data, got {data_path}")
            return False

        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Text length: {len(content)} characters")
        print(f"First 200 characters: {content[:200]}...")

    else:
        print(f"Unknown data type: {data_type}")
        return False

    print("Data validation completed")
    return True


def main():
    """Main function for data preparation"""
    parser = argparse.ArgumentParser(
        description="Prepare data for Tiny MultiModal Larimar")

    parser.add_argument('--task', type=str, required=True,
                        choices=['download_babylm', 'download_conceptual', 'create_dummy_multimodal',
                                 'create_dummy_images', 'prepare_babylm', 'validate'],
                        help='Data preparation task')

    parser.add_argument('--output_path', type=str, default='data',
                        help='Output directory for prepared data')
    parser.add_argument('--input_path', type=str, default=None,
                        help='Input directory for data processing')

    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to create for dummy data')
    parser.add_argument('--max_files', type=int, default=10,
                        help='Maximum number of files to process')

    parser.add_argument('--data_type', type=str, default='multimodal',
                        choices=['multimodal', 'babylm', 'conceptual'],
                        help='Type of data for validation')

    args = parser.parse_args()

    if args.task == 'download_babylm':
        download_babylm_data(args.output_path)

    elif args.task == 'download_conceptual':
        download_conceptual_captions_embeddings()

    elif args.task == 'create_dummy_multimodal':
        create_dummy_multimodal_data(args.output_path, args.num_samples)

    elif args.task == 'create_dummy_images':
        create_dummy_image_caption_pairs(args.output_path, args.num_samples)

    elif args.task == 'prepare_babylm':
        if not args.input_path:
            raise ValueError("Input path required for prepare_babylm task")
        prepare_babylm_text_data(
            args.input_path, args.output_path, args.max_files)

    elif args.task == 'validate':
        if not args.input_path:
            raise ValueError("Input path required for validation task")
        validate_data(args.input_path, args.data_type)

    else:
        print(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
