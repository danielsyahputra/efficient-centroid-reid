# ReID with EfficientNet-V2 and Augmentations

A Person Re-Identification (ReID) project built with PyTorch Lightning that explores the use of EfficientNet-V2 models and specialized ReID augmentations to improve identification performance.

## Overview

This project implements a person re-identification system with the following key features:

- **Multiple backbone architectures**: ResNet, ResNet-IBN-A, EfficientNet, and EfficientNet-V2
- **ReID-specific augmentations**: Local Grayscale Transformation (LGT), Local Grayscale Patch Replacement (LGPR), Global Grayscale Patch Replacement (GGPR), and RGB-Gray-Sketch fusion
- **Centroid-based evaluation**: Optional use of centroids during validation for improved performance
- **Mixed precision training**: Support for faster training with automatic mixed precision

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch Lightning 1.1.4
- PyTorch 1.7.1+cu101
- EfficientNet PyTorch
- YACS for configuration management

## Dataset Setup

1. **Prepare your dataset in COCO format** with the following structure:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── query/
   │   └── gallery/
   └── annotations/
       ├── train_reid.json
       ├── query_reid.json
       └── gallery_reid.json
   ```

2. **For DeepFashion Consumer-to-Shop dataset**, use the conversion script:
   ```bash
   python scripts/deep_fashion2reid.py --root-dir-path /path/to/deepfashion --target-image-size 320 320
   ```

## Configuration

Create or modify configuration files in the `configs/` directory. Example configurations are provided:

- `configs/256_resnet50.yml` - ResNet50 with 256x128 input
- `configs/320_efficientnet_b0.yml` - EfficientNet-B0 with 320x320 input
- `configs/320_efficientnet_b1.yml` - EfficientNet-B1 with 320x320 input

### Key Configuration Options

```yaml
MODEL:
  NAME: 'efficientnet-b0'  # Model architecture
  USE_CENTROIDS: True      # Use centroids for evaluation
  
INPUT:
  SIZE_TRAIN: [320, 320]   # Training image size
  USE_LGT: False           # Local Grayscale Transformation
  USE_LGPR: True           # Local Grayscale Patch Replacement  
  USE_GGPR: True           # Global Grayscale Patch Replacement
  USE_FUSE_RGB: False      # RGB-Gray-Sketch fusion

SOLVER:
  BASE_LR: 0.0001         # Learning rate
  MAX_EPOCHS: 50          # Training epochs
  IMS_PER_BATCH: 32       # Batch size

DATASETS:
  NAMES: 'df1'
  ROOT_DIR: '/path/to/dataset'
  JSON_TRAIN_PATH: '/path/to/train_annotations.json'
```

## Training

### Create a Training Script

Create a training script (e.g., `train_efficientnet_v2.sh`):

```bash
#!/bin/bash
python train_ctl_model.py \
--config_file="configs/320_efficientnet_b0.yml" \
MODEL.NAME 'efficientnet-b0' \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'dataset/train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'dataset/320_320_images' \
SOLVER.IMS_PER_BATCH 32 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 1e-4 \
SOLVER.MAX_EPOCHS 50 \
OUTPUT_DIR './logs/efficientnet_b0_augmented' \
DATALOADER.USE_RESAMPLING False \
MODEL.KEEP_CAMID_CENTROIDS False \
INPUT.USE_LGT False \
INPUT.USE_LGPR True \
INPUT.USE_GGPR True \
INPUT.USE_FUSE_RGB True \
MODEL.USE_CENTROIDS True \
EXPERIMENT_NAME 'efficientnet_b0_with_augmentations'
```

### Run Training

```bash
chmod +x train_efficientnet_v2.sh
./train_efficientnet_v2.sh
```

Or run directly:
```bash
python train_ctl_model.py --config_file="configs/320_efficientnet_b0.yml" [additional parameters]
```

## Project Novelty

This project introduces several key innovations:

1. **EfficientNet-V2 Integration**: First implementation using EfficientNet-V2 models for person ReID, providing better efficiency and accuracy trade-offs.

2. **ReID-Specific Augmentations**:
   - **LGT (Local Grayscale Transformation)**: Randomly converts local patches to grayscale
   - **LGPR (Local Grayscale Patch Replacement)**: Replaces random patches with grayscale versions
   - **GGPR (Global Grayscale Patch Replacement)**: Global grayscale transformations
   - **RGB-Gray-Sketch Fusion**: Combines RGB, grayscale, and sketch representations

3. **Comparative Analysis**: Systematic evaluation of augmentation effects on different backbone architectures.

## Evaluation

The model automatically evaluates on validation data during training. Key metrics include:
- **mAP (mean Average Precision)**: Primary evaluation metric
- **CMC (Cumulative Matching Characteristics)**: Rank-1, Rank-5, Rank-10 accuracy
- **Top-K accuracy**: For various K values

## Model Checkpoints

Models are automatically saved during training:
- Best model based on mAP is saved automatically
- Periodic checkpoints every N epochs (configurable)
- Checkpoints include full model state for resuming training

## Testing Only

To run evaluation on a trained model:

```bash
python train_ctl_model.py \
--config_file="configs/your_config.yml" \
TEST.ONLY_TEST True \
TEST.WEIGHT "/path/to/checkpoint.pth" \
[other parameters]
```

## Logs and Monitoring

Training logs and metrics are automatically saved to:
- TensorBoard logs in the output directory
- MLflow tracking (if configured)
- Console output with detailed metrics

## Tips for Best Results

1. **Batch Size**: Adjust based on GPU memory (32 works well for most setups)
2. **Learning Rate**: Start with 1e-4, adjust based on convergence
3. **Augmentations**: Try different combinations of LGPR, GGPR, and RGB fusion
4. **Image Size**: 320x320 generally works better than 256x128 for EfficientNet models
5. **Centroids**: Enable `USE_CENTROIDS` for better evaluation performance

## Example Training Commands

### EfficientNet-B0 with All Augmentations
```bash
python train_ctl_model.py \
--config_file="configs/320_efficientnet_b0.yml" \
MODEL.NAME 'efficientnet-b0' \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'dataset/train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'dataset/320_320_images' \
SOLVER.IMS_PER_BATCH 32 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 1e-4 \
SOLVER.MAX_EPOCHS 50 \
OUTPUT_DIR './logs/efficientnet_b0_all_aug' \
INPUT.USE_LGPR True \
INPUT.USE_GGPR True \
INPUT.USE_FUSE_RGB True \
MODEL.USE_CENTROIDS True \
EXPERIMENT_NAME 'efficientnet_b0_full_augmentation'
```

### EfficientNet-V2-S with Selective Augmentations
```bash
python train_ctl_model.py \
--config_file="configs/320_efficientnet_v2_s.yml" \
MODEL.NAME 'efficientnet_v2-s' \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'dataset/train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'dataset/320_320_images' \
SOLVER.IMS_PER_BATCH 24 \
TEST.IMS_PER_BATCH 24 \
SOLVER.BASE_LR 1e-4 \
SOLVER.MAX_EPOCHS 50 \
OUTPUT_DIR './logs/efficientnet_v2_s_selective' \
INPUT.USE_LGPR True \
INPUT.USE_GGPR False \
INPUT.USE_FUSE_RGB False \
MODEL.USE_CENTROIDS True \
EXPERIMENT_NAME 'efficientnet_v2_s_lgpr_only'
```

### ResNet50 Baseline
```bash
python train_ctl_model.py \
--config_file="configs/256_resnet50.yml" \
MODEL.NAME 'resnet50' \
GPU_IDS [0] \
DATASETS.NAMES 'df1' \
DATASETS.JSON_TRAIN_PATH 'dataset/train_reid_cropped_320_320.json' \
DATASETS.ROOT_DIR 'dataset/320_320_images' \
SOLVER.IMS_PER_BATCH 32 \
TEST.IMS_PER_BATCH 32 \
SOLVER.BASE_LR 1e-4 \
SOLVER.MAX_EPOCHS 50 \
OUTPUT_DIR './logs/resnet50_baseline' \
INPUT.USE_LGPR False \
INPUT.USE_GGPR False \
INPUT.USE_FUSE_RGB False \
MODEL.USE_CENTROIDS True \
EXPERIMENT_NAME 'resnet50_baseline'
```
