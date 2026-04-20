import os
import json
import shutil
import random
from collections import Counter
from pathlib import Path

def balance_roboflow_dataset(dataset_path, output_path):
    """
    Balance a Roboflow dataset export by:
    - Reducing highest class by 1000 annotations
    - Reducing second highest class by 500 annotations  
    - Duplicating lowest 3 classes by 5x each
    
    Args:
        dataset_path (str): Path to the original dataset folder
        output_path (str): Path where balanced dataset will be saved
    """
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Find all annotation files (assuming YOLO format with .txt files)
    annotation_files = []
    image_files = []
    
    # Look in train/valid/test folders
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            # Get annotation files
            labels_path = os.path.join(split_path, 'labels')
            if os.path.exists(labels_path):
                for file in os.listdir(labels_path):
                    if file.endswith('.txt'):
                        annotation_files.append(os.path.join(labels_path, file))
            
            # Get image files
            images_path = os.path.join(split_path, 'images')
            if os.path.exists(images_path):
                for file in os.listdir(images_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_files.append(os.path.join(images_path, file))
    
    # Count class distribution
    class_counts = Counter()
    file_to_classes = {}
    
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            classes_in_file = []
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
                    classes_in_file.append(class_id)
            file_to_classes[ann_file] = classes_in_file
    
    print("Original class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} annotations")
    
    # Sort classes by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_classes) < 5:
        print(f"Warning: Only {len(sorted_classes)} classes found. Adjusting strategy.")
    
    # Identify which files belong to which classes
    highest_class = sorted_classes[0][0] if len(sorted_classes) > 0 else None
    second_highest_class = sorted_classes[1][0] if len(sorted_classes) > 1 else None
    lowest_3_classes = [cls[0] for cls in sorted_classes[-3:]] if len(sorted_classes) >= 3 else [cls[0] for cls in sorted_classes]
    
    print(f"\nBalancing strategy:")
    print(f"Highest class ({highest_class}): Reduce by 1000")
    print(f"Second highest class ({second_highest_class}): Reduce by 500") 
    print(f"Lowest 3 classes {lowest_3_classes}: Duplicate 5x each")
    
    # Group files by their primary class (class with most annotations in that file)
    files_by_class = {cls: [] for cls in class_counts.keys()}
    
    for ann_file, classes_in_file in file_to_classes.items():
        if classes_in_file:
            # Find the most common class in this file
            primary_class = max(set(classes_in_file), key=classes_in_file.count)
            files_by_class[primary_class].append(ann_file)
    
    # Create balanced dataset
    files_to_copy = []
    files_to_duplicate = []
    
    for class_id, files in files_by_class.items():
        if class_id == highest_class:
            # Reduce by 1000 (randomly select files to keep)
            target_count = max(0, len(files) - 500)
            selected_files = random.sample(files, min(target_count, len(files)))
            files_to_copy.extend(selected_files)
            print(f"Class {class_id}: Keeping {len(selected_files)}/{len(files)} files")
            
        elif class_id == second_highest_class:
            # Reduce by 500
            target_count = max(0, len(files) - 250)
            selected_files = random.sample(files, min(target_count, len(files)))
            files_to_copy.extend(selected_files)
            print(f"Class {class_id}: Keeping {len(selected_files)}/{len(files)} files")
            
        elif class_id in lowest_3_classes:
            # Duplicate 5x
            files_to_copy.extend(files)
            for _ in range(4):  # 4 more times to make 5x total
                files_to_duplicate.extend(files)
            print(f"Class {class_id}: Duplicating {len(files)} files 5x")
            
        else:
            # Keep all files for middle classes
            files_to_copy.extend(files)
            print(f"Class {class_id}: Keeping all {len(files)} files")
    
    # Copy the dataset structure
    for split in ['train', 'valid', 'test']:
        split_src = os.path.join(dataset_path, split)
        split_dst = os.path.join(output_path, split)
        if os.path.exists(split_src):
            os.makedirs(os.path.join(split_dst, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dst, 'labels'), exist_ok=True)
    
    # Copy selected files
    def copy_file_pair(ann_file, suffix=""):
        # Copy annotation file
        rel_path = os.path.relpath(ann_file, dataset_path)
        dst_ann = os.path.join(output_path, rel_path)
        
        if suffix:
            # Add suffix before extension
            name_parts = os.path.splitext(dst_ann)
            dst_ann = name_parts[0] + suffix + name_parts[1]
        
        os.makedirs(os.path.dirname(dst_ann), exist_ok=True)
        shutil.copy2(ann_file, dst_ann)
        
        # Find and copy corresponding image file
        ann_name = os.path.splitext(os.path.basename(ann_file))[0]
        ann_dir = os.path.dirname(ann_file)
        img_dir = ann_dir.replace('labels', 'images')
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_file = os.path.join(img_dir, ann_name + ext)
            if os.path.exists(img_file):
                rel_img_path = os.path.relpath(img_file, dataset_path)
                dst_img = os.path.join(output_path, rel_img_path)
                
                if suffix:
                    name_parts = os.path.splitext(dst_img)
                    dst_img = name_parts[0] + suffix + name_parts[1]
                
                os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                shutil.copy2(img_file, dst_img)
                break
    
    # Copy regular files
    print(f"\nCopying {len(files_to_copy)} files...")
    for ann_file in files_to_copy:
        copy_file_pair(ann_file)
    
    # Copy duplicated files with suffixes
    print(f"Duplicating {len(files_to_duplicate)} files...")
    duplicate_counter = {}
    for ann_file in files_to_duplicate:
        if ann_file not in duplicate_counter:
            duplicate_counter[ann_file] = 0
        duplicate_counter[ann_file] += 1
        suffix = f"_dup{duplicate_counter[ann_file]}"
        copy_file_pair(ann_file, suffix)
    
    # Copy other files (data.yaml, etc.)
    for item in os.listdir(dataset_path):
        src_item = os.path.join(dataset_path, item)
        dst_item = os.path.join(output_path, item)
        
        if os.path.isfile(src_item) and not os.path.exists(dst_item):
            shutil.copy2(src_item, dst_item)
    
    print(f"\nBalanced dataset saved to: {output_path}")
    
    # Count final distribution
    final_counts = Counter()
    for split in ['train', 'valid', 'test']:
        labels_path = os.path.join(output_path, split, 'labels')
        if os.path.exists(labels_path):
            for file in os.listdir(labels_path):
                if file.endswith('.txt'):
                    with open(os.path.join(labels_path, file), 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.strip().split()[0])
                                final_counts[class_id] += 1
    
    print("\nFinal class distribution:")
    for class_id, count in sorted(final_counts.items()):
        original_count = class_counts.get(class_id, 0)
        change = count - original_count
        print(f"Class {class_id}: {count} annotations (change: {change:+d})")

# Usage example
if __name__ == "__main__":
    # Set your paths
    dataset_path = r"C:\Users\Test\Documents\Thesis Files\datasetv11"  # Replace with your dataset path
    output_path = r"C:\Users\Test\Documents\Thesis Files\datasetv11Rebalanced"        # Replace with desired output path
    
    # Run the balancing
    balance_roboflow_dataset(dataset_path, output_path)
