#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path


def test_data_loading():
    """Test if the BabyLM data can be loaded"""
    data_path = Path("data/babylm")

    print("=== Testing BabyLM Data Loading ===")

    # Test Conceptual Captions
    print("\n--- Conceptual Captions ---")
    cc_captions = data_path / "cc_3M_captions.json"
    cc_embed1 = data_path / "cc_3M_dino_v2_states_1of2.npy"
    cc_embed2 = data_path / "cc_3M_dino_v2_states_2of2.npy"

    if cc_captions.exists():
        print(f"✓ Captions file exists: {cc_captions}")
        with open(cc_captions, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"  - Captions count: {len(captions)}")
        print(f"  - Sample caption: {captions[0] if captions else 'None'}")
    else:
        print(f"✗ Captions file missing: {cc_captions}")

    if cc_embed1.exists() and cc_embed2.exists():
        print(f"✓ Embedding files exist")
        embed1 = np.load(cc_embed1)
        embed2 = np.load(cc_embed2)
        print(f"  - Embeddings 1 shape: {embed1.shape}")
        print(f"  - Embeddings 2 shape: {embed2.shape}")
        print(f"  - Total embeddings: {embed1.shape[0] + embed2.shape[0]}")
    else:
        print(f"✗ Embedding files missing")

    # Test Local Narratives
    print("\n--- Local Narratives ---")
    ln_captions = data_path / "local_narr_captions.json"
    ln_embed = data_path / "local_narr_dino_v2_states.npy"

    if ln_captions.exists():
        print(f"✓ Captions file exists: {ln_captions}")
        with open(ln_captions, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"  - Captions count: {len(captions)}")
        print(f"  - Sample caption: {captions[0] if captions else 'None'}")
    else:
        print(f"✗ Captions file missing: {ln_captions}")

    if ln_embed.exists():
        print(f"✓ Embedding file exists: {ln_embed}")
        embed = np.load(ln_embed)
        print(f"  - Embeddings shape: {embed.shape}")
    else:
        print(f"✗ Embedding file missing: {ln_embed}")


if __name__ == "__main__":
    test_data_loading()
