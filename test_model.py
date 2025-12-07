import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset configuration (must match train_model.py)
DATA_DIR = "dataset_split"
CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
NUM_CLASSES = len(CLASSES)

# Base model name should match MODEL_NAME used in train_model.py
DEFAULT_MODEL_NAME = "resnet18"

# Available model architectures (must match train_model.py)
AVAILABLE_MODELS = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'densenet121', 'densenet169', 'densenet201',
    'vgg11', 'vgg13', 'vgg16', 'vgg19',
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'inception_v3', 'alexnet'
]

# By default, look for weights file named best_models/best_model_{model_name}.pth
def get_model_path(model_name: str) -> str:
    return f"best_models/best_model_{model_name}.pth"

def create_model(model_name='resnet18', num_classes=NUM_CLASSES, pretrained=False):
    """Create the same model architecture as used in training
    
    Note: This should match the model_name used during training!
    """
    # Import the same function from train_model
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_model", "train_model.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    return train_module.create_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained)

def load_model(model_name: str = DEFAULT_MODEL_NAME, model_path: str | None = None, device=None):
    """Load the trained model
    
    Args:
        model_name: Model architecture name (must match training)
        model_path: Path to saved model weights. If None, uses
                    'best_model_{model_name}.pth'.
        device: Device to use (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        print(f"Warning: '{model_name}' not in known models list.")
        print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
        print("Attempting to load anyway...")
    
    # Infer model_path from model_name if not provided
    if model_path is None:
        # Try model-specific name first, then fallback to generic name
        model_specific_path = get_model_path(model_name)
        generic_path = 'best_models/best_model.pth'
        
        if os.path.exists(model_specific_path):
            model_path = model_specific_path
        elif os.path.exists(generic_path):
            model_path = generic_path
            print(f"Note: Using generic 'best_models/best_model.pth' (model-specific '{model_specific_path}' not found)")
        else:
            model_path = model_specific_path  # Will raise error below
    
    print(f"Loading model from {model_path}...")
    print(f"Model architecture: {model_name}")
    print(f"Using device: {device}")
    
    # Create model architecture (must match training)
    model = create_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    
    # Load weights
    if not os.path.exists(model_path):
        # Try alternative paths
        alternative_paths = [
            get_model_path(model_name),
            'best_models/best_model.pth',
            f'best_models/best_model_{model_name}.pth',
            # Also try old paths for backward compatibility
            f'best_model_{model_name}.pth',
            'best_model.pth'
        ]
        available = [p for p in alternative_paths if os.path.exists(p)]
        
        error_msg = (
            f"Model file '{model_path}' not found.\n"
            f"Tried: {model_path}\n"
        )
        if available:
            error_msg += f"Found these model files: {', '.join(available)}\n"
            error_msg += f"Please specify the correct path using --model or train the model first."
        else:
            error_msg += (
                f"Expected weights file named 'best_models/best_model_{model_name}.pth' or 'best_models/best_model.pth'.\n"
                f"Please train the model first or specify --model explicitly."
            )
        raise FileNotFoundError(error_msg)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    return model, device

def get_test_transform():
    """Get the same transform used for validation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, device, transform=None):
    """Predict a single image"""
    if transform is None:
        transform = get_test_transform()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities[0].cpu().numpy()

def test_class(model, class_name, split='val', device=None, max_images=None, transform=None):
    """Test model on all images from a specific class"""
    if transform is None:
        transform = get_test_transform()
    
    class_dir = Path(DATA_DIR) / split / class_name
    
    if not class_dir.exists():
        print(f"Error: Class directory '{class_dir}' does not exist.")
        return None
    
    # Get all images
    image_files = list(class_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"No images found in {class_dir}")
        return None
    
    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"\nTesting on {len(image_files)} images from class: '{class_name}'")
    print(f"Expected class index: {CLASSES.index(class_name)}")
    print("-" * 60)
    
    # Get true label
    true_label = CLASSES.index(class_name)
    
    # Predict all images
    all_predictions = []
    all_confidences = []
    all_probabilities = []
    
    for img_path in tqdm(image_files, desc="Predicting"):
        pred_class, confidence, probs = predict_single_image(model, img_path, device, transform)
        all_predictions.append(pred_class)
        all_confidences.append(confidence)
        all_probabilities.append(probs)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_probabilities = np.array(all_probabilities)
    
    accuracy = (all_predictions == true_label).mean()
    avg_confidence = all_confidences.mean()
    
    # Count predictions per class
    pred_counts = {CLASSES[i]: (all_predictions == i).sum() for i in range(NUM_CLASSES)}
    
    # Print results
    print(f"\nResults for class '{class_name}':")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"\n  Predictions distribution:")
    for class_name_pred, count in pred_counts.items():
        percentage = (count / len(all_predictions)) * 100
        marker = "✓" if class_name_pred == class_name else " "
        print(f"    {marker} {class_name_pred:25s}: {count:4d} ({percentage:5.2f}%)")
    
    return {
        'predictions': all_predictions,
        'confidences': all_confidences,
        'probabilities': all_probabilities,
        'true_label': true_label,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'pred_counts': pred_counts
    }

def test_all_classes(model, split='val', device=None, max_images_per_class=None):
    """Test model on all classes"""
    print("\n" + "=" * 60)
    print(f"TESTING MODEL ON ALL CLASSES - {split.upper()} SET")
    print("=" * 60)
    
    all_true_labels = []
    all_predictions = []
    
    for class_name in CLASSES:
        results = test_class(model, class_name, split=split, device=device, 
                           max_images=max_images_per_class)
        
        if results is not None:
            all_true_labels.extend([results['true_label']] * len(results['predictions']))
            all_predictions.extend(results['predictions'].tolist())
    
    # Overall metrics
    if len(all_predictions) > 0:
        print("\n" + "=" * 60)
        print(f"OVERALL RESULTS - {split.upper()} SET")
        print("=" * 60)
        
        overall_accuracy = accuracy_score(all_true_labels, all_predictions)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Get unique labels that are actually present
        unique_labels = sorted(set(all_true_labels + all_predictions))
        present_class_names = [CLASSES[i] for i in unique_labels]
        
        print("\nClassification Report:")
        print(classification_report(
            all_true_labels, all_predictions, 
            labels=unique_labels,
            target_names=present_class_names
        ))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_true_labels, all_predictions, labels=unique_labels)
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
        
        return {
            'accuracy': overall_accuracy,
            'confusion_matrix': cm,
            'true_labels': all_true_labels,
            'predictions': all_predictions,
            'unique_labels': unique_labels
        }
    return None

def print_confusion_matrix(cm, unique_labels):
    """Helper function to print confusion matrix"""
    print(" " * 20, end="")
    for label_idx in unique_labels:
        print(f"{CLASSES[label_idx][:15]:>15}", end="")
    print()
    for i, label_idx in enumerate(unique_labels):
        print(f"{CLASSES[label_idx][:20]:20}", end="")
        for j, pred_label_idx in enumerate(unique_labels):
            print(f"{cm[i][j]:>15}", end="")
        print()

def compare_train_val(model, device=None, max_images_per_class=None):
    """Test model on both train and validation sets and compare results"""
    print("\n" + "=" * 80)
    print("COMPARING TRAIN vs VALIDATION PERFORMANCE")
    print("=" * 80)
    
    # Test on validation set
    print("\n" + "=" * 80)
    val_results = test_all_classes(model, split='val', device=device, 
                                   max_images_per_class=max_images_per_class)
    
    # Test on training set
    print("\n" + "=" * 80)
    train_results = test_all_classes(model, split='train', device=device, 
                                     max_images_per_class=max_images_per_class)
    
    # Compare results
    if train_results is not None and val_results is not None:
        print("\n" + "=" * 80)
        print("COMPARISON: TRAIN vs VALIDATION")
        print("=" * 80)
        
        train_acc = train_results['accuracy']
        val_acc = val_results['accuracy']
        acc_diff = train_acc - val_acc
        
        print(f"\nAccuracy Comparison:")
        print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Difference:          {acc_diff:+.4f} ({acc_diff*100:+.2f}%)")
        
        if acc_diff > 0.1:
            print(f"  ⚠️  WARNING: Large gap suggests overfitting!")
        elif acc_diff < -0.05:
            print(f"  ⚠️  WARNING: Validation better than training - unusual!")
        else:
            print(f"  ✓ Gap is reasonable")
        
        # Compare confusion matrices
        print(f"\nConfusion Matrix Comparison:")
        print(f"\nTRAINING SET:")
        print_confusion_matrix(train_results['confusion_matrix'], train_results['unique_labels'])
        
        print(f"\nVALIDATION SET:")
        print_confusion_matrix(val_results['confusion_matrix'], val_results['unique_labels'])
        
        # Per-class accuracy comparison
        print(f"\nPer-Class Accuracy Comparison:")
        print(f"{'Class':<25} {'Train Acc':>12} {'Val Acc':>12} {'Difference':>12}")
        print("-" * 65)
        
        for label_idx in train_results['unique_labels']:
            class_name = CLASSES[label_idx]
            
            # Calculate per-class accuracy for train
            train_mask = np.array(train_results['true_labels']) == label_idx
            train_class_acc = (np.array(train_results['predictions'])[train_mask] == label_idx).mean() if train_mask.sum() > 0 else 0.0
            
            # Calculate per-class accuracy for val
            val_mask = np.array(val_results['true_labels']) == label_idx
            val_class_acc = (np.array(val_results['predictions'])[val_mask] == label_idx).mean() if val_mask.sum() > 0 else 0.0
            
            diff = train_class_acc - val_class_acc
            print(f"{class_name:<25} {train_class_acc:>11.2f}% {val_class_acc:>11.2f}% {diff:>+11.2f}%")

def visualize_predictions(model, class_name, split='val', num_images=9, device=None):
    """Visualize sample predictions from a class"""
    class_dir = Path(DATA_DIR) / split / class_name
    
    if not class_dir.exists():
        print(f"Error: Class directory '{class_dir}' does not exist.")
        return
    
    image_files = list(class_dir.glob('*.jpg'))[:num_images]
    
    if len(image_files) == 0:
        print(f"No images found in {class_dir}")
        return
    
    transform = get_test_transform()
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    true_label = CLASSES.index(class_name)
    
    for idx, img_path in enumerate(image_files):
        if idx >= num_images:
            break
        
        # Load and display image
        image = Image.open(img_path).convert('RGB')
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        # Predict
        pred_class, confidence, probs = predict_single_image(model, img_path, device, transform)
        
        # Format prediction text
        pred_class_name = CLASSES[pred_class]
        is_correct = "✓" if pred_class == true_label else "✗"
        color = "green" if pred_class == true_label else "red"
        
        title = f"{is_correct} Pred: {pred_class_name}\nConf: {confidence:.2f}"
        axes[idx].set_title(title, color=color, fontsize=10)
    
    plt.suptitle(f"True Class: {class_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test Alzheimer\'s disease classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Test with default model (resnet18):
  python test_model.py
  
  # Test with a specific model:
  python test_model.py --model-name resnet50
  python test_model.py --model-name efficientnet_b2
  
  # Test a specific class:
  python test_model.py --model-name resnet50 --class-name "Non Demented"
  
  # Compare train vs validation:
  python test_model.py --model-name resnet50 --compare

Available models: {', '.join(AVAILABLE_MODELS)}
        """
    )
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME,
                       choices=AVAILABLE_MODELS,
                       help=f'Model architecture name (default: {DEFAULT_MODEL_NAME}). Must match the model used during training.')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model weights file (default: best_models/best_model_{model_name}.pth). Overrides --model-name path inference.')
    parser.add_argument('--class-name', type=str, default=None, dest='class_name',
                       choices=CLASSES + ['all'],
                       help='Class to test (default: all classes)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='Dataset split to use (default: val)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images per class to test (default: all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample predictions')
    parser.add_argument('--compare', action='store_true',
                       help='Compare train vs validation performance (default: True when testing all classes)')
    parser.add_argument('--no-compare', action='store_true', dest='no_compare',
                       help='Disable train/val comparison (test only on specified split)')
    
    args = parser.parse_args()
    
    # Load model
    # NOTE: Make sure model_name matches what you used in train_model.py (MODEL_NAME)
    model, device = load_model(model_name=args.model_name, model_path=args.model)
    
    # Determine if we should compare train vs val
    # Default: compare when testing all classes, unless --no-compare is set
    # Only compare if testing all classes (not a specific class)
    is_testing_all_classes = (args.class_name is None or args.class_name == 'all')
    should_compare = (args.compare or (is_testing_all_classes and not args.no_compare))
    
    # Test
    if should_compare and is_testing_all_classes:
        # Compare train vs validation
        compare_train_val(model, device=device, max_images_per_class=args.max_images)
    elif is_testing_all_classes:
        # Test on single split only (when --no-compare is set)
        test_all_classes(model, split=args.split, device=device, 
                        max_images_per_class=args.max_images)
    else:
        # Test specific class
        results = test_class(model, args.class_name, split=args.split, device=device, 
                           max_images=args.max_images)
        
        if args.visualize and results is not None:
            visualize_predictions(model, args.class_name, split=args.split, device=device)

if __name__ == "__main__":
    # If run without arguments, test all classes
    import sys
    
    if len(sys.argv) == 1:
        print("Testing model and comparing train vs validation performance...")
        print(f"\nUsing default model: {DEFAULT_MODEL_NAME}")
        print(f"To test a different model, use: python test_model.py --model-name <model_name>")
        print(f"\nAvailable models: {', '.join(AVAILABLE_MODELS)}")
        print("\nUsage examples:")
        print(f"  python test_model.py --model-name resnet50  # Test and compare with resnet50")
        print(f"  python test_model.py --model-name resnet50 --no-compare  # Test only validation")
        print(f"  python test_model.py --model-name resnet50 --split train  # Test only training")
        print(f"  python test_model.py --model-name efficientnet_b2 --class-name 'Non Demented'")
        print(f"  python test_model.py --model-name resnet50 --class-name 'Mild Dementia' --visualize")
        print()
        
        # Default: compare train vs validation
        model, device = load_model(model_name=DEFAULT_MODEL_NAME)
        compare_train_val(model, device=device)
    else:
        main()

