import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import random
import re
from collections import defaultdict

# Dataset configuration - BINARY CLASSIFICATION
# Class 0: Non Demented
# Class 1: All dementia classes combined (Very mild, Mild, Moderate)
DATA_DIR = "dataset_split"
ORIGINAL_CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
CLASSES = ["Non Demented", "Demented"]  # Binary classification
NUM_CLASSES = 2

# Mapping: original class name -> binary class index
CLASS_MAPPING = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 1,
    "Moderate Dementia": 1
}

def calculate_adaptive_fractions(data_dir, split='train', base_fraction=1.0, random_seed=42):
    """Calculate adaptive fractions for each original class based on their sizes.
    
    Classes with more samples get reduced more, classes with fewer samples get reduced less.
    This helps balance the dataset.
    
    Args:
        data_dir: Path to dataset directory
        split: 'train' or 'val'
        base_fraction: Base fraction to use (controls overall reduction)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping original class names to their sampling fractions
    """
    data_path = Path(data_dir) / split
    class_counts = {}
    
    # Count images in each original class
    for class_name in ORIGINAL_CLASSES:
        class_dir = data_path / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*.jpg')))
        else:
            class_counts[class_name] = 0
    
    # Find min and max counts (excluding zeros)
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    if len(non_zero_counts) == 0:
        return {class_name: 1.0 for class_name in ORIGINAL_CLASSES}
    
    min_count = min(non_zero_counts)
    max_count = max(non_zero_counts)
    
    # Calculate adaptive fractions
    # Strategy: Use inverse frequency weighting
    # Larger classes get smaller fractions, smaller classes get larger fractions
    fractions = {}
    for class_name, count in class_counts.items():
        if count == 0:
            fractions[class_name] = 1.0
        else:
            # Normalize count to [0, 1] range
            if max_count > min_count:
                # Inverse frequency: (max - count) / (max - min) gives higher value for smaller classes
                normalized = (max_count - count) / (max_count - min_count)
                # Scale to range [base_fraction * min_frac, base_fraction]
                # Smaller classes get closer to base_fraction, larger classes get more reduced
                min_frac = 0.3  # Minimum fraction for largest class (controls how much to reduce large classes)
                class_fraction = base_fraction * (min_frac + (1 - min_frac) * normalized)
            else:
                # All classes have same size
                class_fraction = base_fraction
            
            fractions[class_name] = min(1.0, class_fraction)
    
    return fractions

def extract_patient_number(filename):
    """Extract patient number from filename like OAS1_0001_MR1_mpr-1_100.jpg"""
    match = re.search(r'OAS1_(\d{4})_', filename)
    if match:
        return match.group(1)
    return None

class AlzheimerBinaryDataset(Dataset):
    """Dataset class for Alzheimer's MRI images - Binary Classification
    
    Class 0: Non Demented
    Class 1: All dementia classes combined
    """
    
    def __init__(self, data_dir, split='train', transform=None, data_fraction=1.0, 
                 adaptive_sampling=True, per_class_patient_fractions=None, 
                 data_fraction_per_class=None, random_seed=42):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Set random seed once for reproducibility
        if data_fraction < 1.0 or per_class_patient_fractions is not None or data_fraction_per_class is not None:
            random.seed(random_seed)
        
        # If using per-class patient fractions, use that method
        if per_class_patient_fractions is not None:
            self._load_with_per_class_patient_sampling(
                per_class_patient_fractions, 
                data_fraction, 
                data_fraction_per_class
            )
        else:
            # Original class-based sampling
            # Calculate per-class fractions if using adaptive sampling
            if adaptive_sampling and data_fraction < 1.0:
                class_fractions = calculate_adaptive_fractions(data_dir, split, data_fraction, random_seed)
            else:
                class_fractions = {class_name: data_fraction for class_name in ORIGINAL_CLASSES}
            
            # Load all images and their labels
            for original_class_name in ORIGINAL_CLASSES:
                class_dir = self.data_dir / original_class_name
                if not class_dir.exists():
                    continue
                
                # Get all images for this original class
                class_images = list(class_dir.glob('*.jpg'))
                
                # Skip if no images found
                if len(class_images) == 0:
                    continue
                
                # Get fraction for this class
                class_fraction = class_fractions.get(original_class_name, data_fraction)
                
                # If using a fraction, randomly sample
                if class_fraction < 1.0:
                    num_samples = max(1, int(len(class_images) * class_fraction))
                    # Ensure we don't try to sample more than available
                    num_samples = min(num_samples, len(class_images))
                    class_images = random.sample(class_images, num_samples)
                
                # Map to binary class
                binary_label = CLASS_MAPPING[original_class_name]
                
                # Add to dataset
                for img_file in class_images:
                    self.images.append(str(img_file))
                    self.labels.append(binary_label)
            
            # Print sampling info if using adaptive sampling
            if adaptive_sampling and data_fraction < 1.0:
                print(f"\nAdaptive sampling fractions for {split} set:")
                for original_class_name in ORIGINAL_CLASSES:
                    if original_class_name in class_fractions:
                        class_dir = self.data_dir / original_class_name
                        if class_dir.exists():
                            total = len(list(class_dir.glob('*.jpg')))
                            if total > 0:
                                sampled = sum(1 for img_path in self.images 
                                            if Path(img_path).parent.name == original_class_name)
                                frac = class_fractions[original_class_name]
                                print(f"  {original_class_name:25s}: {frac:.3f} ({sampled:5d}/{total:5d} images)")
        
        # Print binary class distribution
        binary_counts = [0] * NUM_CLASSES
        for label in self.labels:
            binary_counts[label] += 1
        print(f"\nBinary class distribution for {split} set:")
        for i, class_name in enumerate(CLASSES):
            print(f"  {class_name:25s}: {binary_counts[i]:5d} samples")
    
    def _load_with_per_class_patient_sampling(self, per_class_patient_fractions, 
                                             image_fraction=1.0, data_fraction_per_class=None):
        """Load data by sampling a fraction of patients per class, then sampling a fraction of images from selected patients
        
        Args:
            per_class_patient_fractions: Dict mapping original class names to patient sampling fractions
                Example: {'Non Demented': 0.005, 'Mild Dementia': 0.01, ...}
            image_fraction: Fraction of images to use from each selected patient (0.0 to 1.0)
                Used if data_fraction_per_class is None. If 1.0, uses all images from selected patients.
            data_fraction_per_class: Dict mapping original class names to image sampling fractions.
                If set, overrides image_fraction with per-class values.
                Example: {'Non Demented': 0.05, 'Mild Dementia': 0.1, ...}
        """
        # Group patients by original class
        class_patients = defaultdict(lambda: defaultdict(list))  # class -> patient -> images
        
        for original_class_name in ORIGINAL_CLASSES:
            class_dir = self.data_dir / original_class_name
            if not class_dir.exists():
                continue
            
            for img_file in class_dir.glob('*.jpg'):
                patient_num = extract_patient_number(img_file.name)
                if patient_num:
                    class_patients[original_class_name][patient_num].append(img_file)
        
        # Sample patients per class and sample images from selected patients
        total_patients_before = 0
        total_patients_after = 0
        total_images_before = 0
        total_images_after = 0
        
        print(f"\nPer-class patient sampling for {self.data_dir.name} set:")
        if data_fraction_per_class is not None:
            print("Using per-class image fractions:")
            for class_name, frac in data_fraction_per_class.items():
                print(f"  {class_name:25s}: {frac*100:.1f}% of images")
        elif image_fraction < 1.0:
            print(f"Using {image_fraction*100:.1f}% of images from each selected patient")
        
        for original_class_name in ORIGINAL_CLASSES:
            if original_class_name not in class_patients:
                continue
            
            # Get all unique patients for this class
            patients = list(class_patients[original_class_name].keys())
            total_patients_before += len(patients)
            
            # Get patient fraction for this class (default to 1.0 if not specified)
            patient_fraction = per_class_patient_fractions.get(original_class_name, 1.0)
            
            if patient_fraction < 1.0:
                # Sample fraction of patients
                num_patients = max(1, int(len(patients) * patient_fraction))
                num_patients = min(num_patients, len(patients))
                selected_patients = random.sample(patients, num_patients)
            else:
                # Keep all patients
                selected_patients = patients
            
            total_patients_after += len(selected_patients)
            
            # Get image fraction for this class
            if data_fraction_per_class is not None:
                class_image_fraction = data_fraction_per_class.get(original_class_name, image_fraction)
            else:
                class_image_fraction = image_fraction
            
            # Map to binary class
            binary_label = CLASS_MAPPING[original_class_name]
            
            # Collect images from selected patients (with optional image sampling)
            class_images_before = 0
            class_images_after = 0
            for patient_num in selected_patients:
                images = class_patients[original_class_name][patient_num]
                class_images_before += len(images)
                total_images_before += len(images)
                
                # Sample fraction of images from this patient if class_image_fraction < 1.0
                if class_image_fraction < 1.0:
                    num_samples = max(1, int(len(images) * class_image_fraction))
                    num_samples = min(num_samples, len(images))
                    sampled_images = random.sample(images, num_samples)
                else:
                    sampled_images = images
                
                class_images_after += len(sampled_images)
                total_images_after += len(sampled_images)
                
                # Add sampled images from this patient with binary label
                for img_file in sampled_images:
                    self.images.append(str(img_file))
                    self.labels.append(binary_label)
            
            # Print statistics for this class
            if class_image_fraction < 1.0:
                print(f"  {original_class_name:25s}: {len(selected_patients):3d}/{len(patients):3d} patients "
                      f"({patient_fraction*100:.2f}%) - {class_images_after:5d}/{class_images_before:5d} images "
                      f"({class_image_fraction*100:.1f}%) -> Binary class {binary_label}")
            else:
                print(f"  {original_class_name:25s}: {len(selected_patients):3d}/{len(patients):3d} patients "
                      f"({patient_fraction*100:.2f}%) - {class_images_after:5d} images -> Binary class {binary_label}")
        
        print(f"\n  Total: {total_patients_after}/{total_patients_before} patients, "
              f"{total_images_after}/{total_images_before} images ({total_images_after/total_images_before*100:.2f}%)")
        
        # Print binary class distribution
        binary_counts = [0] * NUM_CLASSES
        for label in self.labels:
            binary_counts[label] += 1
        print(f"\nBinary class distribution for {self.data_dir.name} set:")
        for i, class_name in enumerate(CLASSES):
            print(f"  {class_name:25s}: {binary_counts[i]:5d} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(batch_size=32, num_workers=4, data_fraction=1.0, adaptive_sampling=True, 
                     per_class_patient_fractions=None, data_fraction_per_class=None):
    """Create train and validation data loaders for binary classification
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_fraction: Fraction of images to use (0.0 to 1.0). 
            - If per_class_patient_fractions is set: fraction of images from each selected patient (used if data_fraction_per_class is None)
            - Otherwise: base fraction of data to use (random sampling across all images)
        adaptive_sampling: If True, larger classes are reduced more than smaller classes (ignored if per_class_patient_fractions is set)
        per_class_patient_fractions: Dict mapping original class names to patient sampling fractions.
            If set, samples that fraction of patients per class, then uses image fractions from each.
            Set all values to 1.0 to keep all patients and sample images from each.
            Example: {'Non Demented': 0.005, 'Mild Dementia': 0.01, 'Very mild Dementia': 0.01, 'Moderate Dementia': 0.02}
            Or: {'Non Demented': 1.0, 'Very mild Dementia': 1.0, ...} to keep all patients
        data_fraction_per_class: Dict mapping original class names to image sampling fractions.
            If set, overrides data_fraction with per-class values.
            Only used when per_class_patient_fractions is set.
            Example: {'Non Demented': 0.05, 'Very mild Dementia': 0.1, 'Mild Dementia': 0.1, 'Moderate Dementia': 0.2}
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (use different seeds for train/val to get different samples)
    train_dataset = AlzheimerBinaryDataset(
        DATA_DIR, split='train', transform=train_transform, 
        data_fraction=data_fraction, adaptive_sampling=adaptive_sampling, 
        per_class_patient_fractions=per_class_patient_fractions,
        data_fraction_per_class=data_fraction_per_class,
        random_seed=42
    )
    val_dataset = AlzheimerBinaryDataset(
        DATA_DIR, split='val', transform=val_transform, 
        data_fraction=data_fraction, adaptive_sampling=adaptive_sampling,
        per_class_patient_fractions=per_class_patient_fractions,
        data_fraction_per_class=data_fraction_per_class,
        random_seed=123  # Different seed for validation
    )
    
    # Create data loaders
    # pin_memory only works with CUDA, not MPS or CPU
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def create_model(model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True):
    """Create a model for binary classification
    
    Args:
        model_name: Name of the model architecture. Options:
            - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            - 'efficientnet_b0' to 'efficientnet_b7'
            - 'densenet121', 'densenet169', 'densenet201'
            - 'vgg11', 'vgg13', 'vgg16', 'vgg19'
            - 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
            - 'inception_v3'
            - 'alexnet'
        num_classes: Number of output classes (should be 2 for binary)
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model with final layer replaced for num_classes
    """
    # Map model names to their creation functions and weight classes
    model_configs = {
        # ResNet family (good balance of accuracy and speed)
        'resnet18': (models.resnet18, models.ResNet18_Weights),
        'resnet34': (models.resnet34, models.ResNet34_Weights),
        'resnet50': (models.resnet50, models.ResNet50_Weights),
        'resnet101': (models.resnet101, models.ResNet101_Weights),
        'resnet152': (models.resnet152, models.ResNet152_Weights),
        
        # EfficientNet family (state-of-the-art, efficient)
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        'efficientnet_b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights),
        'efficientnet_b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights),
        'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights),
        'efficientnet_b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights),
        'efficientnet_b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights),
        'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights),
        
        # DenseNet family (memory efficient, good accuracy)
        'densenet121': (models.densenet121, models.DenseNet121_Weights),
        'densenet169': (models.densenet169, models.DenseNet169_Weights),
        'densenet201': (models.densenet201, models.DenseNet201_Weights),
        
        # VGG family (classic, but large)
        'vgg11': (models.vgg11, models.VGG11_Weights),
        'vgg13': (models.vgg13, models.VGG13_Weights),
        'vgg16': (models.vgg16, models.VGG16_Weights),
        'vgg19': (models.vgg19, models.VGG19_Weights),
        
        # MobileNet family (lightweight, fast)
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights),
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
        
        # Other architectures
        'inception_v3': (models.inception_v3, models.Inception_V3_Weights),
        'alexnet': (models.alexnet, models.AlexNet_Weights),
    }
    
    model_name = model_name.lower()
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")
    
    model_fn, weights_class = model_configs[model_name]
    
    # Get weights
    if pretrained:
        weights = weights_class.IMAGENET1K_V1
    else:
        weights = None
    
    # Create model
    model = model_fn(weights=weights)
    
    # Replace final layer based on model architecture
    if 'resnet' in model_name or 'resnext' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'efficientnet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif 'densenet' in model_name:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif 'vgg' in model_name or 'alexnet' in model_name:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif 'mobilenet' in model_name:
        if 'v2' in model_name:
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        else:  # v3
            num_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_features, num_classes)
    elif 'inception' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture type for model: {model_name}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, verbose=False):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Additional diagnostics
    if verbose:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Per-class accuracy
        print("\n  Per-class accuracy:")
        for i, class_name in enumerate(CLASSES):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = (all_preds[mask] == all_labels[mask]).mean()
                print(f"    {class_name:25s}: {class_acc:.4f} ({mask.sum()} samples)")
        
        # Prediction distribution
        print("\n  Prediction distribution:")
        for i, class_name in enumerate(CLASSES):
            count = (all_preds == i).sum()
            pct = (count / len(all_preds)) * 100
            print(f"    {class_name:25s}: {count:4d} ({pct:5.2f}%)")
        
        # Average confidence
        avg_confidence = all_probs.max(axis=1).mean()
        print(f"\n  Average prediction confidence: {avg_confidence:.4f}")
        
        # Check if model is predicting mostly one class
        unique, counts = np.unique(all_preds, return_counts=True)
        max_class_pct = (counts.max() / len(all_preds)) * 100
        if max_class_pct > 80:
            print(f"  WARNING: Model is predicting {max_class_pct:.1f}% of samples as one class!")
            print(f"           This suggests class imbalance issues.")
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(num_epochs=20, batch_size=32, learning_rate=0.001, data_fraction=1.0,
                model_name='resnet50', 
                adaptive_sampling=True, per_class_patient_fractions=None, data_fraction_per_class=None):
    """Main training function for binary classification
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        data_fraction: Fraction of data to use (0.0 to 1.0). 
            - If per_class_patient_fractions is set: fraction of images from each selected patient (used if data_fraction_per_class is None)
            - Otherwise: base fraction of data to use (random sampling)
        adaptive_sampling: If True, larger classes are reduced more than smaller classes (ignored if per_class_patient_fractions is set)
        per_class_patient_fractions: Dict mapping original class names to patient sampling fractions.
            If set, samples that fraction of patients per class, then uses image fractions from each.
            Set all values to 1.0 to keep all patients and sample images from each.
        data_fraction_per_class: Dict mapping original class names to image sampling fractions.
            If set, overrides data_fraction with per-class values.
            Only used when per_class_patient_fractions is set.
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"BINARY CLASSIFICATION: Class 0 = Non Demented, Class 1 = All Dementia Classes Combined")
    
    # Create data loaders
    print("Loading data...")
    if per_class_patient_fractions is not None:
        print("Using per-class patient sampling:")
        for class_name, frac in per_class_patient_fractions.items():
            print(f"  {class_name:25s}: {frac*100:.2f}% of patients")
        if data_fraction_per_class is not None:
            print("  Then using per-class image fractions (see details below)")
        elif data_fraction < 1.0:
            print(f"  Then using {data_fraction*100:.1f}% of images from each selected patient")
    elif data_fraction < 1.0:
        if adaptive_sampling:
            print(f"Using adaptive sampling with base fraction {data_fraction*100:.1f}%")
            print("(Larger classes will be reduced more than smaller classes)")
        else:
            print(f"Using {data_fraction*100:.1f}% of the data for faster training")
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size, 
        data_fraction=data_fraction,
        adaptive_sampling=adaptive_sampling,
        per_class_patient_fractions=per_class_patient_fractions,
        data_fraction_per_class=data_fraction_per_class
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {model_name}...")
    model = create_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # CLASS WEIGHTS CONFIGURATION (BINARY)
    # ============================================
    # Count samples per binary class in training set (for class weight calculation)
    class_counts = [0] * NUM_CLASSES
    for image, label in train_loader.dataset:
        class_counts[label] += 1
    
    # Note: Binary class distribution is already printed by AlzheimerBinaryDataset.__init__
    
    # Calculate class weights for binary classification
    # Option 1: Automatic inverse frequency weighting
    total_samples = sum(class_counts)
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = total_samples / (NUM_CLASSES * count)  # Inverse frequency
        else:
            weight = 1.0
        class_weights.append(weight)
    
    # Option 2: Manual weights (uncomment to use)
    # class_weights = [1.0, 2.0]  # Adjust as needed: [Non Demented, Demented]
    
    # Option 3: Disable weighting
    # class_weights = None
    
    # Convert to tensor
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"\nClass weights:")
        for i, class_name in enumerate(CLASSES):
            print(f"  {class_name:25s}: {class_weights[i]:.3f}")
    else:
        print("\nUsing equal class weights (no weighting)")
    
    # Loss and optimizer
    # Use weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate (show detailed metrics every 5 epochs or on last epoch)
        show_details = (epoch + 1) % 5 == 0 or epoch == num_epochs - 1
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, verbose=show_details
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Warning if loss decreasing but accuracy not improving
        if epoch > 0 and len(val_losses) >= 2:
            loss_improved = val_losses[-1] < val_losses[-2]
            acc_improved = val_accs[-1] > val_accs[-2]
            if loss_improved and not acc_improved and val_accs[-1] == val_accs[-2]:
                print("  ⚠️  WARNING: Loss decreased but accuracy unchanged!")
                print("     This may indicate class imbalance or model overconfidence.")
        
        # Save best model (with binary suffix to avoid conflicts)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create best_models directory if it doesn't exist
            os.makedirs('best_models', exist_ok=True)
            model_path = f'best_models/best_model_binary_{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (Val Acc: {best_val_acc:.4f}) to {model_path}")
    
    # Load best model and print final results
    model_path = f'best_models/best_model_binary_{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    _, final_val_acc, final_preds, final_labels = validate(model, val_loader, criterion, device)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Convert to numpy arrays
    final_preds = np.array(final_preds)
    final_labels = np.array(final_labels)
    
    # Determine which classes are actually present in the validation data
    unique_labels = sorted(set(final_labels.tolist() + final_preds.tolist()))
    present_class_names = [CLASSES[i] for i in unique_labels]
    
    # Classification report (only for present classes)
    print("\nClassification Report:")
    print(classification_report(
        final_labels, final_preds,
        labels=unique_labels,
        target_names=present_class_names
    ))
    
    # Confusion matrix (only for present classes)
    print("\nConfusion Matrix:")
    cm = confusion_matrix(final_labels, final_preds, labels=unique_labels)
    print(cm)
    
    # Print confusion matrix with labels
    print("\nConfusion Matrix (with labels):")
    print(" " * 20, end="")
    for label_idx in unique_labels:
        print(f"{CLASSES[label_idx][:15]:>15}", end="")
    print()
    for i, label_idx in enumerate(unique_labels):
        print(f"{CLASSES[label_idx][:20]:20}", end="")
        for j, pred_label_idx in enumerate(unique_labels):
            print(f"{cm[i][j]:>15}", end="")
        print()
    
    return model, train_losses, train_accs, val_losses, val_accs

if __name__ == "__main__":
    # Check if dataset is prepared
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found.")
        print("Please run prepare_dataset.py first to create the train/val split.")
        exit(1)
    
    # ============================================
    # DATA SAMPLING CONFIGURATION
    # ============================================
    # Choose ONE of the following sampling strategies:
    #
    # OPTION 1: Uniform fraction (random sampling)
    #   PER_CLASS_PATIENT_FRACTIONS = None
    #   DATA_FRACTION = 0.1  # Use 10% of all data (random sampling across all images)
    #
    # OPTION 2: Per-class patient fractions (RECOMMENDED - most flexible)
    #   PER_CLASS_PATIENT_FRACTIONS = {...}, DATA_FRACTION = 0.05
    #   Step 1: Select X% of patients per class (from PER_CLASS_PATIENT_FRACTIONS)
    #   Step 2: Use Y% of images from each selected patient (from DATA_FRACTION)
    #
    #   To keep ALL patients and sample images: set all values to 1.0
    #   Example: {'Non Demented': 1.0, 'Very mild Dementia': 1.0, ...}
    PER_CLASS_PATIENT_FRACTIONS = {
        'Non Demented': 1.0,        
        'Very mild Dementia': 1.0,  
        'Mild Dementia': 1.0,      
        'Moderate Dementia': 1.0, 
    }
    
    # OPTION A: Uniform image fraction (same for all classes)
    # Step 1: Select X% of patients per class (from PER_CLASS_PATIENT_FRACTIONS above)
    # Step 2: Use Y% of images from each selected patient (from DATA_FRACTION below)
    # DATA_FRACTION = 0.05  # 5% of images from each selected patient
    # DATA_FRACTION_PER_CLASS = None  # Set to None to use uniform DATA_FRACTION
    
    # OPTION B: Per-class image fractions (uncomment to use instead of DATA_FRACTION)
    # This allows different image sampling rates per class
    DATA_FRACTION_PER_CLASS = {
        'Non Demented': 0.2,        # 8% of images
        'Very mild Dementia': 0.2,   # 20% of images
        'Mild Dementia': 0.2,       # 20% of images
        'Moderate Dementia': 0.2,    # 100% of images
    }
    DATA_FRACTION = 1.0
    
    # ============================================
    # MODEL SELECTION
    # ============================================
    # Choose a model architecture. Popular options:
    #
    # Fast & Lightweight:
    #   'resnet18' - Fast, good for quick experiments
    #   'mobilenet_v3_small' - Very fast, mobile-friendly
    #   'efficientnet_b0' - Efficient, good accuracy
    #
    # Balanced (Recommended):
    #   'resnet50' - Good balance (current default)
    #   'efficientnet_b2' - Better accuracy, still efficient
    #   'densenet121' - Good accuracy, memory efficient
    #
    # High Accuracy:
    #   'resnet101' - Better accuracy, slower
    #   'efficientnet_b4' - State-of-the-art accuracy
    #   'densenet201' - Very good accuracy
    #
    # See create_model() function for all available models
    # ============================================
    MODEL_NAME = 'alexnet'  # Change this to try different models
    
    # Train the model
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        num_epochs=50,
        batch_size=100,
        learning_rate=0.0001,
        model_name=MODEL_NAME,
        data_fraction=DATA_FRACTION,  # Used as image fraction if per_class_patient_fractions is set and data_fraction_per_class is None
        per_class_patient_fractions=PER_CLASS_PATIENT_FRACTIONS,
        data_fraction_per_class=DATA_FRACTION_PER_CLASS
    )

