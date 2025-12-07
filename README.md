# Alzheimer's Disease Severity Classification from MRI Images

This project implements a machine learning model to predict Alzheimer's disease severity from MRI images using the OASIS dataset.

## Dataset Structure

The dataset contains MRI images organized into 4 classes:
- **Non Demented** (Class 0)
- **Very mild Dementia** (Class 1)
- **Mild Dementia** (Class 2)
- **Moderate Dementia** (Class 3)

Images are named in the format: `OAS1_####_MR1_mpr-1_###.jpg` where `####` is the patient number.

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset Split

The dataset is split by **patient number** (not individual images) to avoid data leakage. This ensures that all images from the same patient are in either the training or validation set, but never both.

```bash
python prepare_dataset.py
```

This script will:
- Extract patient numbers from all image filenames
- Split patients into train (80%) and validation (20%) sets
- Organize images into `dataset_split/train/` and `dataset_split/val/` folders
- Print detailed statistics about the split

**Output:**
- `dataset_split/train/` - Training images organized by class
- `dataset_split/val/` - Validation images organized by class

### Step 2: Train the Model

```bash
python train_model.py
```

This script will:
- Load the prepared dataset
- Create a ResNet50-based model (pretrained on ImageNet)
- Train the model with data augmentation
- Save the best model as `best_model.pth`
- Display training/validation metrics and classification report

**Training Parameters:**
- Epochs: 20
- Batch size: 32
- Learning rate: 0.001 (with step scheduler)
- Optimizer: Adam
- Loss: CrossEntropyLoss

## Model Architecture

The model uses a **ResNet50** backbone pretrained on ImageNet, with the final fully connected layer replaced to output 4 classes.

**Data Augmentation (Training only):**
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness and contrast)
- Resize to 224x224
- Normalization (ImageNet statistics)

## Files

- `prepare_dataset.py` - Creates train/validation split based on patient numbers
- `train_model.py` - Trains the Alzheimer's severity classification model
- `analyze_data.py` - Original data exploration script
- `requirements.txt` - Python dependencies
- `best_model.pth` - Saved model weights (created after training)

## Notes

- The split is based on **patient numbers** to prevent data leakage
- Class imbalance exists in the dataset (more "Non Demented" images than others)
- The validation set may not contain all classes if certain patients are only in the training set
- GPU is recommended for training but CPU will work (slower)

