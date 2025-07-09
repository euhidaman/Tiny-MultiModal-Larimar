#!/usr/bin/env python3
"""
Download BabyLM multimodal dataset
"""

from src.modules.babylm_data import download_babylm_data
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Download BabyLM multimodal dataset")
    parser.add_argument("--data_path", type=str, default="data/babylm",
                        help="Directory to download data to")
    parser.add_argument("--dataset_type", type=str, default="cc_3M",
                        choices=["cc_3M", "local_narr"],
                        help="Dataset type to download")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if files exist")

    args = parser.parse_args()

    print(f" Downloading BabyLM multimodal dataset ({args.dataset_type})...")
    print(f" Data path: {args.data_path}")

    try:
        download_babylm_data(
            data_path=args.data_path,
            dataset_type=args.dataset_type,
            force_download=args.force
        )
        print(" Download complete!")
    except Exception as e:
        print(f" Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
