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
import matplotlib.pyplot as plt

# Dataset configuration
DATA_DIR = "dataset_split"
CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
NUM_CLASSES = len(CLASSES)

# ImageNet normalization constants (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _get_class_images(class_dir):
    """Helper to get list of image files from a class directory"""
    if not class_dir.exists():
        return []
    return list(class_dir.glob('*.jpg'))

def extract_patient_number(filename):
    """Extract patient number from filename like OAS1_0001_MR1_mpr-1_100.jpg"""
    match = re.search(r'OAS1_(\d{4})_', filename)
    if match:
        return match.group(1)
    return None

class AlzheimerDataset(Dataset):
    """Dataset class for Alzheimer's MRI images"""
    
    def __init__(self, data_dir, split='train', transform=None, data_fraction=1.0, 
                 patient_fraction=None, random_seed=42):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Set random seed once for reproducibility
        if data_fraction < 1.0 or patient_fraction is not None:
            random.seed(random_seed)
        
        # If using patient fraction sampling, use that method
        if patient_fraction is not None:
            self._load_with_patient_sampling(
                patient_fraction, 
                data_fraction
            )
        else:
            # Uniform class-based sampling
            # Load all images and their labels
            for class_idx, class_name in enumerate(CLASSES):
                class_dir = self.data_dir / class_name
                class_images = _get_class_images(class_dir)
                
                # Skip if no images found
                if not class_images:
                    continue
                
                # If using a fraction, randomly sample uniformly
                if data_fraction < 1.0:
                    num_samples = min(max(1, int(len(class_images) * data_fraction)), len(class_images))
                    class_images = random.sample(class_images, num_samples)
                
                # Add to dataset
                for img_file in class_images:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
    
    def _load_with_patient_sampling(self, patient_fraction, image_fraction=1.0):
        """Load data by sampling a fraction of patients uniformly across all classes, then sampling a fraction of images from selected patients
        
        Args:
            patient_fraction: Fraction of patients to sample uniformly across all classes (0.0 to 1.0)
            image_fraction: Fraction of images to use from each selected patient (0.0 to 1.0)
                If 1.0, uses all images from selected patients.
        """
        # Group patients by class
        class_patients = defaultdict(lambda: defaultdict(list))  # class -> patient -> images
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            for img_file in _get_class_images(class_dir):
                patient_num = extract_patient_number(img_file.name)
                if patient_num:
                    class_patients[class_name][patient_num].append(img_file)
        
        # Sample patients uniformly across all classes and sample images from selected patients
        total_patients_before = 0
        total_patients_after = 0
        total_images_before = 0
        total_images_after = 0
        
        print(f"\nPatient sampling for {self.data_dir.name} set:")
        print(f"Using {patient_fraction*100:.1f}% of patients uniformly across all classes")
        if image_fraction < 1.0:
            print(f"Using {image_fraction*100:.1f}% of images from each selected patient")
        
        for class_idx, class_name in enumerate(CLASSES):
            if class_name not in class_patients:
                continue
            
            # Get all unique patients for this class
            patients = list(class_patients[class_name].keys())
            total_patients_before += len(patients)
            
            # Sample fraction of patients uniformly
            if patient_fraction < 1.0:
                num_patients = min(max(1, int(len(patients) * patient_fraction)), len(patients))
                selected_patients = random.sample(patients, num_patients)
            else:
                selected_patients = patients
            
            total_patients_after += len(selected_patients)
            
            # Collect images from selected patients (with optional image sampling)
            class_images_before = 0
            class_images_after = 0
            for patient_num in selected_patients:
                images = class_patients[class_name][patient_num]
                class_images_before += len(images)
                total_images_before += len(images)
                
                # Sample fraction of images from this patient if image_fraction < 1.0
                if image_fraction < 1.0:
                    num_samples = min(max(1, int(len(images) * image_fraction)), len(images))
                    sampled_images = random.sample(images, num_samples)
                else:
                    sampled_images = images
                
                class_images_after += len(sampled_images)
                total_images_after += len(sampled_images)
                
                # Add sampled images from this patient
                for img_file in sampled_images:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
            
            # Print statistics for this class
            if image_fraction < 1.0:
                print(f"  {class_name:25s}: {len(selected_patients):3d}/{len(patients):3d} patients "
                      f"({patient_fraction*100:.2f}%) - {class_images_after:5d}/{class_images_before:5d} images "
                      f"({image_fraction*100:.1f}%)")
            else:
                print(f"  {class_name:25s}: {len(selected_patients):3d}/{len(patients):3d} patients "
                      f"({patient_fraction*100:.2f}%) - {class_images_after:5d} images")
        
        print(f"\n  Total: {total_patients_after}/{total_patients_before} patients, "
              f"{total_images_after}/{total_images_before} images ({total_images_after/total_images_before*100:.2f}%)")
    
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

def get_data_loaders(batch_size=32, num_workers=4, data_fraction=1.0, patient_fraction=None):
    """Create train and validation data loaders
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_fraction: Fraction of images to use (0.0 to 1.0). 
            - If patient_fraction is set: fraction of images from each selected patient
            - Otherwise: uniform fraction of data to use (random sampling across all images)
        patient_fraction: Fraction of patients to sample uniformly across all classes (0.0 to 1.0).
            If set, samples that fraction of patients per class, then uses data_fraction of images from each.
            Set to 1.0 to keep all patients and sample images from each.
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Create datasets (use different seeds for train/val to get different samples)
    dataset_kwargs = {
        'data_fraction': data_fraction,
        'patient_fraction': patient_fraction
    }
    
    train_dataset = AlzheimerDataset(
        DATA_DIR, split='train', transform=train_transform, 
        random_seed=42, **dataset_kwargs
    )
    val_dataset = AlzheimerDataset(
        DATA_DIR, split='val', transform=val_transform, 
        random_seed=123, **dataset_kwargs  # Different seed for validation
    )
    
    # Create data loaders
    # pin_memory only works with CUDA, not MPS or CPU
    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader

def _create_classifier_layer(num_features, num_classes, dropout_rate=0.0):
    """Helper function to create final classifier layer with optional dropout"""
    if dropout_rate > 0.0:
        return nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    return nn.Linear(num_features, num_classes)

def create_model(model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.0):
    """Create a model for classification
    
    Args:
        model_name: Name of the model architecture. Options:
            - 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            - 'efficientnet_b0' to 'efficientnet_b7'
            - 'densenet121', 'densenet169', 'densenet201'
            - 'vgg11', 'vgg13', 'vgg16', 'vgg19'
            - 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'
            - 'inception_v3'
            - 'alexnet'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization (0.0 to disable, typically 0.1-0.5)
    
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
    if 'resnet' in model_name or 'resnext' in model_name or 'inception' in model_name:
        num_features = model.fc.in_features
        model.fc = _create_classifier_layer(num_features, num_classes, dropout_rate)
    elif 'efficientnet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = _create_classifier_layer(num_features, num_classes, dropout_rate)
    elif 'densenet' in model_name:
        num_features = model.classifier.in_features
        model.classifier = _create_classifier_layer(num_features, num_classes, dropout_rate)
    elif 'vgg' in model_name or 'alexnet' in model_name:
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = _create_classifier_layer(num_features, num_classes, dropout_rate)
    elif 'mobilenet' in model_name:
        layer_idx = 1 if 'v2' in model_name else -1
        num_features = model.classifier[layer_idx].in_features
        model.classifier[layer_idx] = _create_classifier_layer(num_features, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown architecture type for model: {model_name}")
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, l1_lambda=0.0):
    """Train for one epoch
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        l1_lambda: L1 regularization strength (0.0 to disable)
    """
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
        
        # Add L1 regularization if specified
        if l1_lambda > 0.0:
            l1_reg = 0.0
            for param in model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss += l1_lambda * l1_reg
        
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
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(num_epochs=20, batch_size=32, learning_rate=0.001, data_fraction=1.0,
                model_name='resnet50', patient_fraction=None,
                weight_decay=0.0, l1_lambda=0.0, dropout_rate=0.0, label_smoothing=0.0):
    """Main training function
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        data_fraction: Fraction of data to use (0.0 to 1.0). 
            - If patient_fraction is set: fraction of images from each selected patient
            - Otherwise: uniform fraction of data to use (random sampling)
        patient_fraction: Fraction of patients to sample uniformly across all classes (0.0 to 1.0).
            If set, samples that fraction of patients per class, then uses data_fraction of images from each.
            Set to 1.0 to keep all patients and sample images from each.
        weight_decay: L2 regularization strength (weight decay) for optimizer (default: 0.0, typical: 1e-4 to 1e-2)
        l1_lambda: L1 regularization strength added to loss (default: 0.0, typical: 1e-5 to 1e-3)
        dropout_rate: Dropout rate for final layer (default: 0.0, typical: 0.1-0.5)
        label_smoothing: Label smoothing factor for CrossEntropyLoss (default: 0.0, typical: 0.1-0.2)
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    if patient_fraction is not None:
        print(f"Using {patient_fraction*100:.1f}% of patients uniformly across all classes")
        if data_fraction < 1.0:
            print(f"Then using {data_fraction*100:.1f}% of images from each selected patient")
    elif data_fraction < 1.0:
        print(f"Using {data_fraction*100:.1f}% of the data for faster training")
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size, 
        data_fraction=data_fraction,
        patient_fraction=patient_fraction
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {model_name}...")
    model = create_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=True, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Print regularization settings
    if weight_decay > 0.0 or l1_lambda > 0.0 or dropout_rate > 0.0 or label_smoothing > 0.0:
        print("\nRegularization settings:")
        if weight_decay > 0.0:
            print(f"  L2 Regularization (weight_decay): {weight_decay}")
        if l1_lambda > 0.0:
            print(f"  L1 Regularization (l1_lambda): {l1_lambda}")
        if dropout_rate > 0.0:
            print(f"  Dropout rate: {dropout_rate}")
        if label_smoothing > 0.0:
            print(f"  Label smoothing: {label_smoothing}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # CLASS WEIGHTS CONFIGURATION
    # ============================================
    # You can modify class weights here in several ways:
    # 
    # Option 1: Automatic inverse frequency weighting (current)
    # Option 2: Manual weights - set custom values
    # Option 3: Equal weights - disable weighting (set to None)
    # Option 4: Square root or other formulas
    # ============================================
    
    # Count samples per class in training set
    class_counts = [0] * NUM_CLASSES
    for image, label in train_loader.dataset:
        class_counts[label] += 1
    
    print(f"\nClass distribution in training set:")
    for i, class_name in enumerate(CLASSES):
        print(f"  {class_name:25s}: {class_counts[i]:5d} samples")
    
    # ============================================
    # frequency weighting for the loss function
    # ============================================
    total_samples = sum(class_counts)
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = (total_samples / (NUM_CLASSES * count))**0.6  # Inverse frequency
        else:
            weight = 1.0
        class_weights.append(weight)
    
    # ============================================
    # 
    # ============================================
    class_weights = None  # This disables class weighting
    
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
    # Note: label_smoothing is only available in PyTorch 1.10+
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    except TypeError:
        # Fallback for older PyTorch versions
        if label_smoothing > 0.0:
            print(f"  Warning: label_smoothing not supported in this PyTorch version, ignoring...")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Add weight_decay for L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, l1_lambda=l1_lambda)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate (show detailed metrics every 5 epochs or on last epoch)
        show_details = (epoch + 1) % 1 == 0 or epoch == num_epochs - 1
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Create best_models directory if it doesn't exist
            os.makedirs('best_models', exist_ok=True)
            model_path = f'best_models/best_model_{model_name}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model (Val Acc: {best_val_acc:.4f}) to {model_path}")
    
    # Load best model and print final results
    model_path = f'best_models/best_model_{model_name}.pth'
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
    
    # Plot training and validation accuracy progression
    print("\n" + "="*60)
    print("Generating accuracy plot...")
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(4, 3))
    plt.plot(epochs, train_accs, label='Training Accuracy', linewidth=2, marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', linewidth=2, marker='s')
    plt.title('ResNet101 with regularization', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.6, 1])
    plt.tight_layout()
    plt.show()

    
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
    # PATIENT_FRACTION: Fraction of patients to sample uniformly across all classes (0.0 to 1.0)
    #   Set to 1.0 to keep all patients, or a fraction (e.g., 0.5 for 50% of patients)
    #   If None, patient sampling is not used
    PATIENT_FRACTION = None  # Set to None to disable, or a value like 0.5 for 50% of patients
    
    # DATA_FRACTION: Fraction of images to use
    #   - If PATIENT_FRACTION is set: fraction of images from each selected patient
    #   - Otherwise: base fraction of data to use (random sampling across all images)
    DATA_FRACTION = 0.1  # Use 1.0 for all data, or set to a fraction (e.g., 0.1 for 10%)
    
    # ============================================
    # MODEL SELECTION
    # ============================================
    MODEL_NAME = 'resnet101'  # Change this to try different models
    
    # ============================================
    # REGULARIZATION CONFIGURATION
    # ============================================
    # Choose regularization methods to prevent overfitting:
    #
    # 1. L2 Regularization (Weight Decay) - RECOMMENDED
    #    Adds penalty to large weights. Typical values: 1e-4 to 1e-2
    #    Start with 1e-4 and increase if still overfitting
    WEIGHT_DECAY = 1e-4  # L2 regularization strength
    
    # 2. L1 Regularization
    #    Encourages sparsity (zero weights). Typical values: 1e-5 to 1e-3
    #    Use if you want feature selection. Can combine with L2.
    L1_LAMBDA = 0  # L1 regularization strength (0.0 to disable)
    
    # 3. Dropout
    #    Randomly sets some neurons to zero during training. Typical values: 0.1-0.5
    #    Higher values = more regularization. Start with 0.2-0.3.
    DROPOUT_RATE = 0.0  # Dropout rate (0.0 to disable)
    
    # 4. Label Smoothing
    #    Softens hard labels to reduce overconfidence. Typical values: 0.1-0.2
    #    Helps with calibration and generalization.
    LABEL_SMOOTHING = 0.0  # Label smoothing factor (0.0 to disable)
    # ============================================
    
    # Train the model
    model, train_losses, train_accs, val_losses, val_accs = train_model(
        num_epochs=10,
        batch_size=100,
        learning_rate=0.0001,
        model_name=MODEL_NAME,
        data_fraction=DATA_FRACTION,
        patient_fraction=PATIENT_FRACTION,
        weight_decay=WEIGHT_DECAY,
        l1_lambda=L1_LAMBDA,
        dropout_rate=DROPOUT_RATE,
        label_smoothing=LABEL_SMOOTHING
    )
