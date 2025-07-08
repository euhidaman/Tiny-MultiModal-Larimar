#!/usr/bin/env python3
"""
Dataset Exploration Script for Tiny-MultiModal-Larimar
This script downloads and explores the structure of datasets on CPU.
Run this separately from model training to understand data formats.
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Create data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def log_message(message: str):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def download_file(url: str, filepath: Path, chunk_size: int = 8192):
    """Download a file with progress indication"""
    log_message(f"Downloading {url} to {filepath}")

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%",
                              end='', flush=True)

            print()  # New line after progress
            log_message(f"Download completed: {filepath}")
            return True

    except Exception as e:
        log_message(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file"""
    log_message(f"Extracting {zip_path} to {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        log_message(f"Extraction completed")
        return True
    except Exception as e:
        log_message(f"Error extracting {zip_path}: {e}")
        return False


def explore_directory_structure(path: Path, max_depth: int = 3, current_depth: int = 0):
    """Explore directory structure recursively"""
    items = []

    if current_depth >= max_depth:
        return items

    try:
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"{'  ' * current_depth}📁 {item.name}/")
                items.extend(explore_directory_structure(
                    item, max_depth, current_depth + 1))
            else:
                size = item.stat().st_size
                size_str = format_file_size(size)
                items.append(
                    f"{'  ' * current_depth}📄 {item.name} ({size_str})")
    except PermissionError:
        items.append(f"{'  ' * current_depth}❌ Permission denied")

    return items


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f}TB"


def analyze_text_file(filepath: Path, max_lines: int = 10) -> Dict[str, Any]:
    """Analyze a text file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            total_lines = 0
            total_chars = 0

            for line in f:
                total_lines += 1
                total_chars += len(line)

                if len(lines) < max_lines:
                    lines.append(line.strip())

        return {
            'total_lines': total_lines,
            'total_chars': total_chars,
            'sample_lines': lines,
            'avg_line_length': total_chars / total_lines if total_lines > 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_json_file(filepath: Path, max_items: int = 5) -> Dict[str, Any]:
    """Analyze a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            return {
                'type': 'dict',
                'keys': list(data.keys())[:max_items],
                'total_keys': len(data.keys()),
                'sample_data': {k: v for k, v in list(data.items())[:max_items]}
            }
        elif isinstance(data, list):
            return {
                'type': 'list',
                'length': len(data),
                'sample_items': data[:max_items],
                'item_types': [type(item).__name__ for item in data[:max_items]]
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:200]
            }
    except Exception as e:
        return {'error': str(e)}


def download_babylm_dataset():
    """Download BabyLM dataset"""
    log_message("=== Downloading BabyLM Dataset ===")

    babylm_url = "https://files.osf.io/v1/resources/ad7qg/providers/osfstorage/6603014bb3a1e301127dfa59/?zip="
    babylm_zip = DATA_DIR / "babylm_data.zip"
    babylm_dir = DATA_DIR / "babylm"

    # Download if not exists
    if not babylm_zip.exists():
        if not download_file(babylm_url, babylm_zip):
            log_message("Failed to download BabyLM dataset")
            return False
    else:
        log_message(f"BabyLM dataset already downloaded: {babylm_zip}")

    # Extract if not exists
    if not babylm_dir.exists():
        if not extract_zip(babylm_zip, babylm_dir):
            log_message("Failed to extract BabyLM dataset")
            return False
    else:
        log_message(f"BabyLM dataset already extracted: {babylm_dir}")

    return True


def explore_babylm_dataset():
    """Explore BabyLM dataset structure"""
    log_message("=== Exploring BabyLM Dataset ===")

    babylm_dir = DATA_DIR / "babylm"
    if not babylm_dir.exists():
        log_message("BabyLM dataset not found. Please download first.")
        return

    # Show directory structure
    log_message("Directory structure:")
    structure = explore_directory_structure(babylm_dir, max_depth=4)
    for item in structure[:50]:  # Limit output
        print(item)

    if len(structure) > 50:
        print(f"... and {len(structure) - 50} more items")

    # Analyze sample files
    log_message("\nAnalyzing sample files:")
    for filepath in babylm_dir.rglob("*"):
        if filepath.is_file() and filepath.suffix in ['.txt', '.json']:
            log_message(f"\nAnalyzing: {filepath.relative_to(babylm_dir)}")

            if filepath.suffix == '.txt':
                analysis = analyze_text_file(filepath)
                print(f"  Lines: {analysis.get('total_lines', 'N/A')}")
                print(f"  Characters: {analysis.get('total_chars', 'N/A')}")
                print(
                    f"  Avg line length: {analysis.get('avg_line_length', 'N/A'):.1f}")
                print("  Sample lines:")
                for i, line in enumerate(analysis.get('sample_lines', [])[:3]):
                    print(f"    {i+1}: {line[:100]}...")

            elif filepath.suffix == '.json':
                analysis = analyze_json_file(filepath)
                print(f"  Type: {analysis.get('type', 'N/A')}")
                if 'keys' in analysis:
                    print(f"  Keys: {analysis['keys']}")
                if 'length' in analysis:
                    print(f"  Length: {analysis['length']}")
                if 'sample_data' in analysis:
                    print(f"  Sample: {analysis['sample_data']}")

            # Only analyze first few files to avoid spam
            break


def download_conceptual_captions():
    """Download Conceptual Captions dataset info"""
    log_message("=== Preparing Conceptual Captions Dataset Info ===")

    # Create conceptual captions directory
    cc_dir = DATA_DIR / "conceptual_captions"
    cc_dir.mkdir(exist_ok=True)

    # Create info file about the dataset
    info_file = cc_dir / "dataset_info.json"

    cc_info = {
        "name": "Conceptual Captions 3M",
        "description": "Large-scale dataset of image-caption pairs",
        "url": "https://ai.google.com/research/ConceptualCaptions/",
        "download_instructions": [
            "The dataset is too large to download automatically",
            "Please visit the official website to download",
            "Expected structure:",
            "  conceptual_captions/",
            "  ├── images/",
            "  │   ├── train/",
            "  │   └── val/",
            "  ├── annotations/",
            "  │   ├── train_captions.json",
            "  │   └── val_captions.json"
        ],
        "format": {
            "images": "JPEG files",
            "captions": "JSON format with image_id and caption fields"
        },
        "size": "~100GB for images, ~500MB for captions",
        "alternative": "Use pre-computed DiNOv2 embeddings instead"
    }

    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(cc_info, f, indent=2)

    log_message(f"Conceptual Captions info saved to: {info_file}")


def create_dummy_multimodal_data():
    """Create dummy multimodal data for testing"""
    log_message("=== Creating Dummy Multimodal Data ===")

    dummy_dir = DATA_DIR / "dummy_multimodal"
    dummy_dir.mkdir(exist_ok=True)

    # Create dummy image embeddings (DiNOv2 style)
    embeddings_dir = dummy_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Create sample embeddings and captions
    num_samples = 100
    embedding_dim = 768  # DiNOv2 base dimension

    for i in range(num_samples):
        # Create dummy embedding
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding_file = embeddings_dir / f"image_{i:04d}.npy"
        np.save(embedding_file, embedding)

        # Create dummy caption
        captions = [
            "A beautiful landscape with mountains and trees",
            "A dog playing in the park on a sunny day",
            "People walking on a busy street in the city",
            "A colorful sunset over the ocean",
            "Children playing soccer in a green field"
        ]

        caption_data = {
            "image_id": f"image_{i:04d}",
            "caption": captions[i % len(captions)],
            "source": "dummy_data"
        }

        caption_file = embeddings_dir / f"image_{i:04d}.json"
        with open(caption_file, 'w', encoding='utf-8') as f:
            json.dump(caption_data, f, indent=2)

    log_message(
        f"Created {num_samples} dummy multimodal samples in {dummy_dir}")


def generate_dataset_report():
    """Generate a comprehensive dataset report"""
    log_message("=== Generating Dataset Report ===")

    report_file = DATA_DIR / "dataset_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Dataset Exploration Report\n\n")
        f.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall structure
        f.write("## Overall Data Directory Structure\n\n")
        f.write("```\n")
        structure = explore_directory_structure(DATA_DIR, max_depth=3)
        for item in structure:
            f.write(f"{item}\n")
        f.write("```\n\n")

        # Dataset summaries
        f.write("## Dataset Summaries\n\n")

        # BabyLM
        babylm_dir = DATA_DIR / "babylm"
        if babylm_dir.exists():
            f.write("### BabyLM Dataset\n")
            f.write(f"- Location: `{babylm_dir}`\n")
            f.write(f"- Status: Downloaded and extracted\n")
            f.write(f"- Purpose: Language learning corpus\n\n")
        else:
            f.write("### BabyLM Dataset\n")
            f.write("- Status: Not downloaded\n\n")

        # Conceptual Captions
        cc_dir = DATA_DIR / "conceptual_captions"
        if cc_dir.exists():
            f.write("### Conceptual Captions Dataset\n")
            f.write(f"- Location: `{cc_dir}`\n")
            f.write(f"- Status: Info file created, requires manual download\n")
            f.write(f"- Purpose: Image-caption pairs for multimodal learning\n\n")

        # Dummy data
        dummy_dir = DATA_DIR / "dummy_multimodal"
        if dummy_dir.exists():
            f.write("### Dummy Multimodal Data\n")
            f.write(f"- Location: `{dummy_dir}`\n")
            f.write(f"- Status: Generated for testing\n")
            f.write(f"- Purpose: Test multimodal data pipeline\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Review BabyLM dataset structure\n")
        f.write("2. Decide on image-caption dataset approach\n")
        f.write("3. Update data loading pipeline accordingly\n")
        f.write("4. Test with dummy data first\n")

    log_message(f"Dataset report saved to: {report_file}")


def main():
    """Main exploration function"""
    parser = argparse.ArgumentParser(
        description="Explore datasets for Tiny-MultiModal-Larimar")
    parser.add_argument("--download-babylm",
                        action="store_true", help="Download BabyLM dataset")
    parser.add_argument("--explore-babylm", action="store_true",
                        help="Explore BabyLM dataset")
    parser.add_argument("--create-dummy", action="store_true",
                        help="Create dummy multimodal data")
    parser.add_argument("--generate-report",
                        action="store_true", help="Generate dataset report")
    parser.add_argument("--all", action="store_true",
                        help="Run all exploration steps")

    args = parser.parse_args()

    if args.all:
        args.download_babylm = True
        args.explore_babylm = True
        args.create_dummy = True
        args.generate_report = True

    log_message("Starting dataset exploration...")

    # Download BabyLM dataset
    if args.download_babylm:
        if not download_babylm_dataset():
            log_message("Failed to download BabyLM dataset")
            return

    # Explore BabyLM dataset
    if args.explore_babylm:
        explore_babylm_dataset()

    # Create dummy multimodal data
    if args.create_dummy:
        create_dummy_multimodal_data()

    # Download conceptual captions info
    download_conceptual_captions()

    # Generate report
    if args.generate_report:
        generate_dataset_report()

    log_message("Dataset exploration completed!")


if __name__ == "__main__":
    main()
