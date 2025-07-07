# Dataset Exploration Guide

This guide helps you explore the datasets on your local CPU machine before training on RunPod.

## Quick Start (CPU Only)

1. **Install minimal dependencies:**
```bash
pip install -r requirements_exploration.txt
```

2. **Run complete exploration:**
```bash
python explore_datasets.py --all
```

3. **Or run specific steps:**
```bash
# Download BabyLM dataset
python explore_datasets.py --download-babylm

# Explore BabyLM structure
python explore_datasets.py --explore-babylm

# Create dummy multimodal data for testing
python explore_datasets.py --create-dummy

# Generate comprehensive report
python explore_datasets.py --generate-report
```

## What This Does

### 1. Downloads BabyLM Dataset
- Downloads the zip file from OSF
- Extracts and explores the structure
- Analyzes sample files to understand format

### 2. Prepares Multimodal Data Info
- Creates info about Conceptual Captions dataset
- Generates dummy multimodal data for testing
- Shows expected data structure

### 3. Generates Report
- Creates `data/dataset_report.md` with findings
- Documents data structure and formats
- Provides recommendations for data pipeline

## Expected Output Structure

```
data/
├── babylm/                     # BabyLM dataset
│   ├── (extracted files)
│   └── ...
├── conceptual_captions/        # Image-caption dataset info
│   └── dataset_info.json
├── dummy_multimodal/          # Test multimodal data
│   └── embeddings/
│       ├── image_0000.npy     # Dummy DiNOv2 embeddings
│       ├── image_0000.json    # Corresponding captions
│       └── ...
└── dataset_report.md          # Comprehensive report
```

## Key Benefits

✅ **CPU Only**: No GPU dependencies, runs on any machine
✅ **Separate Process**: Independent of model training
✅ **Data Structure Analysis**: Understand formats before training
✅ **Dummy Data**: Test pipeline without large downloads
✅ **Comprehensive Report**: Document findings for training setup

## Next Steps After Exploration

1. Review the generated `data/dataset_report.md`
2. Update the data loading pipeline based on findings
3. Test with dummy data first
4. Move to RunPod for actual training

## File Sizes

- BabyLM dataset: ~100MB (compressed)
- Conceptual Captions: Info only (large dataset requires manual download)
- Dummy data: ~100MB (for testing)

This exploration helps you understand the data structure before committing to expensive GPU training time on RunPod.
