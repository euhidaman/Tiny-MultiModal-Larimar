#!/usr/bin/env python3
"""
Quick training script for RunPod with W&B auto-naming
"""

import os
import subprocess
import sys


def get_next_run_number():
    """Get the next available run number"""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs("babylm-ntust/tiny-multimodal-larimar")
        existing_names = [
            run.name for run in runs if run.name and run.name.startswith("baby-larimar")]

        run_number = 1
        while f"baby-larimar{run_number}" in existing_names:
            run_number += 1
        return run_number
    except Exception as e:
        print(f"  Could not connect to W&B API: {e}")
        print("Using default run number: 1")
        return 1


def main():
    # Get next run number
    run_number = get_next_run_number()
    run_name = f"baby-larimar{run_number}"

    print(f" Starting training run: {run_name}")

    # Build command
    cmd = [
        "python", "train_larimar_babylm.py",
        "--config", "configs/config_larimar_babylm.yaml",
        "--logger", "wandb",
        "--wandb_project", "tiny-multimodal-larimar",
        "--wandb_entity", "babylm-ntust",
        "--run_name", run_name,
        "--devices", "1",
        "--accelerator", "gpu",
        "--precision", "16-mixed"
    ]

    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])

    print(f"💻 Command: {' '.join(cmd)}")

    # Run training
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
