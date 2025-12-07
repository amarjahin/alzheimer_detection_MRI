import os
import re
from collections import defaultdict
import random
import shutil
from pathlib import Path

# Path to the dataset
base_path = "/Users/ammarjahin/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1"
data_path = os.path.join(base_path, "Data")

# Class folders
class_folders = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3
}

def extract_patient_number(filename):
    """Extract patient number from filename like OAS1_0001_MR1_mpr-1_100.jpg"""
    match = re.search(r'OAS1_(\d{4})_', filename)
    if match:
        return match.group(1)
    return None

def collect_patient_data():
    """Collect all images grouped by patient number and class"""
    patient_data = defaultdict(lambda: defaultdict(list))
    
    for class_name, class_label in class_folders.items():
        class_folder = os.path.join(data_path, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder {class_folder} does not exist")
            continue
            
        image_files = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]
        
        for img_file in image_files:
            patient_num = extract_patient_number(img_file)
            if patient_num:
                patient_data[patient_num][class_name].append(img_file)
    
    return patient_data

def create_train_val_split(patient_data, val_ratio=0.2, random_state=42, split_per_class=False):
    """Split patients into train and validation sets
    
    Args:
        patient_data: Dictionary of patient data
        val_ratio: Ratio of validation patients
        random_state: Random seed for reproducibility
        split_per_class: If True, split patients per class (ensures balanced class distribution)
                        If False, split all patients globally (current method)
    
    Returns:
        train_patients, val_patients sets
    """
    random.seed(random_state)
    
    if split_per_class:
        # Split per class to ensure balanced class distribution
        train_patients = set()
        val_patients = set()
        
        print("  Splitting patients per class...")
        
        for class_name in class_folders.keys():
            # Get patients that have images in this class
            class_patients = [p for p in patient_data.keys() 
                            if class_name in patient_data[p] and len(patient_data[p][class_name]) > 0]
            
            if len(class_patients) == 0:
                continue
            
            # Shuffle patients for this class
            shuffled = class_patients.copy()
            random.shuffle(shuffled)
            
            # Split for this class
            split_idx = round(len(shuffled) * (1 - val_ratio))
            split_idx = max(1, min(split_idx, len(shuffled) - 1))
            
            class_train = set(shuffled[:split_idx])
            class_val = set(shuffled[split_idx:])
            
            # Resolve conflicts: if a patient is already assigned, keep the first assignment
            # This prevents data leakage (patient in both train and val)
            new_train = class_train - val_patients  # Only add if not already in val
            new_val = class_val - train_patients     # Only add if not already in train
            
            # For conflicts, assign to the majority vote or keep existing assignment
            conflicts_train = class_train & val_patients  # Should be in train but already in val
            conflicts_val = class_val & train_patients    # Should be in val but already in train
            
            if conflicts_train or conflicts_val:
                print(f"    {class_name}: {len(conflicts_train)} conflicts (train->val), {len(conflicts_val)} conflicts (val->train)")
                # Keep existing assignments to prevent data leakage
                # These patients stay in their original split
            
            train_patients.update(new_train)
            val_patients.update(new_val)
            
            print(f"    {class_name:25s}: {len(new_train):3d} train, {len(new_val):3d} val (total: {len(class_patients)})")
        
        # Convert to lists for compatibility
        train_patients = list(train_patients)
        val_patients = list(val_patients)
        
    else:
        # Original global split
        patient_numbers = list(patient_data.keys())
        
        # Shuffle patients
        shuffled_patients = patient_numbers.copy()
        random.shuffle(shuffled_patients)
        
        # Split patients (not images) into train and validation
        split_idx = round(len(shuffled_patients) * (1 - val_ratio))
        split_idx = max(1, min(split_idx, len(shuffled_patients) - 1))
        
        train_patients = shuffled_patients[:split_idx]
        val_patients = shuffled_patients[split_idx:]
    
    return train_patients, val_patients

def organize_data_for_training(patient_data, train_patients, val_patients, output_dir="dataset_split"):
    """Organize data into train/val folders with class subfolders"""
    output_path = Path(output_dir)
    
    # Remove existing directory if it exists to ensure clean split
    if output_path.exists():
        print(f"Removing existing '{output_dir}' directory to create fresh split...")
        shutil.rmtree(output_path)
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in class_folders.keys():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy files to appropriate folders
    train_count = defaultdict(int)
    val_count = defaultdict(int)
    
    for patient_num, classes in patient_data.items():
        split = 'train' if patient_num in train_patients else 'val'
        
        for class_name, image_files in classes.items():
            for img_file in image_files:
                src = os.path.join(data_path, class_name, img_file)
                dst = output_path / split / class_name / img_file
                
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    if split == 'train':
                        train_count[class_name] += 1
                    else:
                        val_count[class_name] += 1
    
    return train_count, val_count

def print_statistics(patient_data, train_patients, val_patients, train_count, val_count):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal unique patients: {len(patient_data)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    
    print("\n" + "-"*60)
    print("TRAIN SET:")
    print("-"*60)
    total_train = 0
    for class_name in class_folders.keys():
        count = train_count[class_name]
        total_train += count
        print(f"  {class_name:25s}: {count:4d} images")
    print(f"  {'Total':25s}: {total_train:4d} images")
    
    print("\n" + "-"*60)
    print("VALIDATION SET:")
    print("-"*60)
    total_val = 0
    for class_name in class_folders.keys():
        count = val_count[class_name]
        total_val += count
        print(f"  {class_name:25s}: {count:4d} images")
    print(f"  {'Total':25s}: {total_val:4d} images")
    
    print("\n" + "-"*60)
    print("SAMPLE PATIENT DISTRIBUTION:")
    print("-"*60)
    print("\nTrain patients (first 10):")
    for patient in sorted(train_patients)[:10]:
        classes = list(patient_data[patient].keys())
        print(f"  Patient {patient}: {', '.join(classes)}")
    
    print("\nValidation patients (first 10):")
    for patient in sorted(val_patients)[:10]:
        classes = list(patient_data[patient].keys())
        print(f"  Patient {patient}: {', '.join(classes)}")

def main():
    print("Collecting patient data...")
    patient_data = collect_patient_data()
    
    print(f"Found {len(patient_data)} unique patients")
    
    print("\nCreating train/validation split based on patient numbers...")
    # ============================================
    # TRAIN/VAL SPLIT CONFIGURATION
    # ============================================
    # Change val_ratio to adjust the split:
    #   val_ratio=0.2  -> 80% train, 20% validation (default)
    #   val_ratio=0.15 -> 85% train, 15% validation
    #   val_ratio=0.25 -> 75% train, 25% validation
    #   val_ratio=0.1  -> 90% train, 10% validation
    #   val_ratio=0.5  -> 50% train, 50% validation
    #
    # SPLIT_PER_CLASS: If True, splits patients per class separately
    #   - Ensures balanced class distribution in train/val
    #   - Better for imbalanced datasets
    #   - Prevents one class from dominating the split
    #   - Note: If a patient has images in multiple classes, they will be
    #     assigned to one split only (to prevent data leakage)
    # ============================================
    VAL_RATIO = 0.5  # Change this to adjust train/val split
    SPLIT_PER_CLASS = True  # Set to True for per-class splitting, False for global split
    train_patients, val_patients = create_train_val_split(
        patient_data, val_ratio=VAL_RATIO, split_per_class=SPLIT_PER_CLASS
    )
    
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    
    print("\nOrganizing data into train/val folders...")
    train_count, val_count = organize_data_for_training(
        patient_data, train_patients, val_patients
    )
    
    print_statistics(patient_data, train_patients, val_patients, train_count, val_count)
    
    print("\n" + "="*60)
    print(f"Dataset organized in 'dataset_split' folder")
    print("="*60)
    print("\nStructure:")
    print("  dataset_split/")
    print("    train/")
    print("      Non Demented/")
    print("      Very mild Dementia/")
    print("      Mild Dementia/")
    print("      Moderate Dementia/")
    print("    val/")
    print("      Non Demented/")
    print("      Very mild Dementia/")
    print("      Mild Dementia/")
    print("      Moderate Dementia/")

if __name__ == "__main__":
    main()

