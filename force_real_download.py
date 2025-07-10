#!/usr/bin/env python3
"""
Force download real BabyLM dataset by bypassing connectivity checks
"""

import requests
import json
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm


def download_file_with_progress(url: str, filepath: Path) -> bool:
    """Download a file with progress bar"""
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True, timeout=30, verify=False)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(
            f"✅ Downloaded: {filepath} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        return True

    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False


def force_download_real_dataset():
    """Force download the real BabyLM cc_3M dataset"""

    # Create data directory
    data_path = Path("data/babylm")
    data_path.mkdir(parents=True, exist_ok=True)

    # Remove any existing dummy files
    dummy_files = [
        "cc_3M_captions.json",
        "cc_3M_dino_v2_states_1of2.npy",
        "cc_3M_dino_v2_states_2of2.npy"
    ]

    print("🗑️  Removing any existing dummy files...")
    for filename in dummy_files:
        filepath = data_path / filename
        if filepath.exists():
            print(f"Removing: {filepath}")
            filepath.unlink()

    # Real BabyLM URLs - bypassing connectivity check
    urls = {
        "cc_3M_captions.json": "https://data.babylm.github.io/multimodal/cc_3M_captions.json",
        "cc_3M_dino_v2_states_1of2.npy": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_1of2.npy",
        "cc_3M_dino_v2_states_2of2.npy": "https://data.babylm.github.io/multimodal/cc_3M_dino_v2_states_2of2.npy"
    }

    print("🌐 Force downloading real BabyLM dataset...")
    print("This will download several GB of data. Please be patient.")

    success_count = 0

    for filename, url in urls.items():
        filepath = data_path / filename
        if download_file_with_progress(url, filepath):
            success_count += 1
        else:
            print(f"❌ Failed to download {filename}")

    if success_count == len(urls):
        print("\n🎉 SUCCESS! Real BabyLM dataset downloaded!")

        # Verify the files are real (not dummy)
        print("\n📊 Verifying dataset integrity...")

        # Check caption file
        caption_file = data_path / "cc_3M_captions.json"
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                captions = json.load(f)
            print(f"✅ Captions: {len(captions):,} samples")

            if len(captions) < 1000:
                print("⚠️  WARNING: Caption count seems too low!")

        # Check embedding files
        for i, filename in enumerate(["cc_3M_dino_v2_states_1of2.npy", "cc_3M_dino_v2_states_2of2.npy"]):
            embed_file = data_path / filename
            if embed_file.exists():
                embeddings = np.load(embed_file)
                print(
                    f"✅ Embeddings {i+1}: {embeddings.shape[0]:,} samples × {embeddings.shape[1]}D")

                if embeddings.shape[0] < 1000:
                    print(f"⚠️  WARNING: {filename} seems too small!")

        print("\n🚀 Ready for training with REAL dataset!")
        return True
    else:
        print(
            f"\n❌ Download failed: {success_count}/{len(urls)} files downloaded")
        return False


if __name__ == "__main__":
    print("🔥 FORCE DOWNLOAD REAL BABYLM DATASET")
    print("="*50)

    response = input("This will download several GB. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(1)

    success = force_download_real_dataset()
    sys.exit(0 if success else 1)
