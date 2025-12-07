# Alzheimer's Disease Severity Classification from MRI Images

This project implements machine learning models to predict Alzheimer's disease severity from MRI images using the OASIS dataset. It trains multiple convolutional neural networks to detect four different classes of dementia from MRI images.

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
- Create a model (supports multiple architectures: ResNet, EfficientNet, DenseNet, VGG, MobileNet, Inception, AlexNet)
- Train the model with data augmentation and regularization options
- Save the best model as `best_models/best_model_{model_name}.pth`
- Display training/validation metrics, classification report, and accuracy plots

**Training Parameters:**
- Configurable epochs, batch size, and learning rate
- Multiple model architectures available
- Regularization options: L2 (weight decay), L1, dropout, label smoothing
- Class weighting for handling imbalanced data
- Patient-based and adaptive sampling options

## Model Architecture

The project supports multiple architectures:
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet_b0 through efficientnet_b7
- **DenseNet**: densenet121, densenet169, densenet201
- **VGG**: vgg11, vgg13, vgg16, vgg19
- **MobileNet**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **Inception**: inception_v3
- **AlexNet**: alexnet

All models use pretrained ImageNet weights with the final layer replaced to output 4 classes.

**Data Augmentation (Training only):**
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness and contrast)
- Resize to 224x224
- Normalization (ImageNet statistics)

## Files

- `prepare_dataset.py` - Creates train/validation split based on patient numbers
- `train_model.py` - Trains the Alzheimer's severity classification model (4-class)
- `train_model_binary.py` - Binary classification model (demented vs non-demented)
- `test_model.py` - Script to test trained models
- `analyze_data.py` - Data exploration script
- `requirements.txt` - Python dependencies
- `best_models/` - Directory containing saved model weights

## Notes

- The split is based on **patient numbers** to prevent data leakage
- Class imbalance exists in the dataset (more "Non Demented" images than others)
- The validation set may not contain all classes if certain patients are only in the training set
- GPU is recommended for training but CPU will work (slower)
- Model checkpoints and large datasets are excluded from git via `.gitignore`
