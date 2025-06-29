# Dataset Splitting Script for Multi-Frame LiDAR Object Detection
# This script splits multi-frame LiDAR datasets into train/validation/test sets
# It groups files by main scenario to ensure related frames stay together
# Supports both simple_fused and aligned_fused data formats with optional test mode

import os
import glob
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

def get_label_files(fusion_type, scan_time, use_cleaned):
    """
    Get X and Y label files from the specified folders.
    Locates the image (X) and label (Y) folders based on fusion type and scan time.
    
    Args:
        fusion_type (str): Type of fusion ('simple_fused' or 'aligned_fused')
        scan_time (str): Scan timestamp (e.g., '2025-06-05_17-13-28')
        use_cleaned (bool): Whether to use cleaned data (adds 'cleaned_' prefix)
    
    Returns:
        tuple: (x_folder_path, y_folder_path, output_dir)
    
    Raises:
        FileNotFoundError: If X or Y folders are not found
    """
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(base_dir, 'output', 'multi_frame', 'multi_scenarios', fusion_type)
    
    # Set frame count based on fusion type
    frame_count = "5_frame" if fusion_type == "aligned_fused" else "3_frame"
    
    # Set prefix based on whether to use cleaned data
    prefix = "cleaned_" if use_cleaned else ""
    
    # Find X and Y folders matching the scan time
    x_folder = f'{prefix}X_{frame_count}_{fusion_type}_yolo1d_scan_{scan_time}'
    y_folder = f'{prefix}Y_{frame_count}_{fusion_type}_yolo1d_scan_{scan_time}'
    
    x_folder_path = os.path.join(output_dir, x_folder)
    y_folder_path = os.path.join(output_dir, y_folder)
    
    if not os.path.exists(x_folder_path) or not os.path.exists(y_folder_path):
        raise FileNotFoundError(f"X or Y folders not found in {output_dir} for scan time {scan_time}")
    
    return x_folder_path, y_folder_path, output_dir

def create_split_directories(output_dir, fusion_type, scan_time, test_mode=False):
    """
    Create directories for train/val/test splits with fusion_type and scan_time in the name.
    Creates the directory structure needed for YOLO training.
    
    Args:
        output_dir (str): Base output directory
        fusion_type (str): Type of fusion ('simple_fused' or 'aligned_fused')
        scan_time (str): Scan timestamp
        test_mode (bool): If True, adds 'test_' prefix to directory name
    
    Returns:
        str: Path to the created splits directory
    """
    prefix = "test_" if test_mode else ""
    splits_dir = os.path.join(output_dir, f'{prefix}splits_{fusion_type}_{scan_time}')
    
    # Create main splits directory
    os.makedirs(splits_dir, exist_ok=True)
    
    # Create subdirectories for images and labels
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'labels']:
            path = os.path.join(splits_dir, data_type, split)
            os.makedirs(path, exist_ok=True)
    
    return splits_dir

def group_files_by_main_scenario(x_folder, y_folder, test_mode=False, max_scenarios=10):
    """
    Group files by their main scenario number (y in frame_w_x_y_z).
    This ensures that related frames from the same scenario stay together in the same split.
    
    Args:
        x_folder (str): Path to folder containing image files (.png)
        y_folder (str): Path to folder containing label files (.txt)
        test_mode (bool): If True, limit to first max_scenarios scenarios
        max_scenarios (int): Maximum number of scenarios to include in test mode
    
    Returns:
        defaultdict: Dictionary mapping scenario numbers to lists of (image_file, label_file) tuples
    
    Raises:
        FileNotFoundError: If no PNG or TXT files are found
    """
    x_files = glob.glob(os.path.join(x_folder, '*.png'))
    y_files = glob.glob(os.path.join(y_folder, '*.txt'))
    
    if not x_files or not y_files:
        raise FileNotFoundError(f"No PNG or TXT files found in X or Y folders")
    
    # Group files by main scenario number
    scenario_groups = defaultdict(list)
    
    print("\nGrouping files by main scenario...")
    for x_file in tqdm(x_files, desc="Processing files"):
        try:
            filename = os.path.basename(x_file)
            parts = filename.split('_')
            if len(parts) >= 4:
                main_scenario = int(parts[3])  # y in w_x_y_z
                y_file = os.path.join(y_folder, filename.replace('.png', '.txt'))
                if os.path.exists(y_file):
                    scenario_groups[main_scenario].append((x_file, y_file))
        except (IndexError, ValueError):
            continue
    
    # In test mode, limit to first max_scenarios scenarios
    if test_mode:
        limited_groups = defaultdict(list)
        for i, (scenario, files) in enumerate(sorted(scenario_groups.items())):
            if i >= max_scenarios:
                break
            limited_groups[scenario] = files
        scenario_groups = limited_groups
    
    return scenario_groups

def split_and_copy_files(scenario_groups, splits_dir, val_percent, test_percent):
    """
    Split scenarios into train/val/test and copy files.
    Ensures that all files from the same scenario go to the same split.
    
    Args:
        scenario_groups (defaultdict): Dictionary mapping scenarios to file lists
        splits_dir (str): Directory where splits will be created
        val_percent (float): Percentage of scenarios for validation
        test_percent (float): Percentage of scenarios for testing
    
    Returns:
        dict: Dictionary with counts of files in each split
    """
    # Get all main scenario numbers
    main_scenarios = list(scenario_groups.keys())
    random.shuffle(main_scenarios)  # Randomize scenario order for fair splitting
    
    # Calculate split indices
    n_scenarios = len(main_scenarios)
    n_val = int(n_scenarios * val_percent / 100)
    n_test = int(n_scenarios * test_percent / 100)
    
    # Split scenarios into train/val/test sets
    val_scenarios = main_scenarios[:n_val]
    test_scenarios = main_scenarios[n_val:n_val + n_test]
    train_scenarios = main_scenarios[n_val + n_test:]
    
    # Initialize counters for tracking file counts
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Copy files for each split
    for split, scenarios in [('train', train_scenarios), ('val', val_scenarios), ('test', test_scenarios)]:
        print(f"\nCopying files for {split} split...")
        for scenario in tqdm(scenarios, desc=f"Processing {split} scenarios"):
            for x_file, y_file in scenario_groups[scenario]:
                # Copy image file to appropriate split directory
                x_dest = os.path.join(splits_dir, 'images', split, os.path.basename(x_file))
                shutil.copy2(x_file, x_dest)
                
                # Copy label file to appropriate split directory
                y_dest = os.path.join(splits_dir, 'labels', split, os.path.basename(y_file))
                shutil.copy2(y_file, y_dest)
                
                counts[split] += 1
    
    return counts

def main():
    """
    Main function that orchestrates the dataset splitting process.
    Handles user input for configuration and executes the splitting pipeline.
    """
    # Get fusion type from user
    print("\nSelect fusion type:")
    print("1. simple_fused")
    print("2. aligned_fused")
    fusion_choice = input("Enter choice (1 or 2): ").strip()
    
    if fusion_choice == "1":
        fusion_type = "simple_fused"
    elif fusion_choice == "2":
        fusion_type = "aligned_fused"
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return
    
    # Get scan time from user
    print("\nEnter scan time (e.g., 2025-06-05_17-13-28):")
    scan_time = input().strip()
    
    # Ask if user wants to use cleaned data
    print("\nUse cleaned data? (y/n, default: n):")
    use_cleaned_choice = input().strip().lower()
    use_cleaned = use_cleaned_choice == 'y'
    
    # Ask if user wants to run in test mode (limited scenarios for quick testing)
    print("\nRun in test mode with first 10 scenarios? (y/n, default: y):")
    test_mode_choice = input().strip().lower()
    test_mode = test_mode_choice != 'n'
    
    # Get validation percentage from user
    print("\nEnter validation percentage (e.g., 10):")
    try:
        val_percent = float(input().strip())
    except ValueError:
        print("Invalid input. Using default value of 10%.")
        val_percent = 10.0
    
    # Get test percentage from user
    print("\nEnter test percentage (e.g., 5):")
    try:
        test_percent = float(input().strip())
    except ValueError:
        print("Invalid input. Using default value of 5%.")
        test_percent = 5.0
    
    # Get the label files and verify they exist
    try:
        x_folder, y_folder, output_dir = get_label_files(fusion_type, scan_time, use_cleaned)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create split directories with appropriate structure
    splits_dir = create_split_directories(output_dir, fusion_type, scan_time, test_mode)
    
    # Group files by main scenario to ensure related frames stay together
    try:
        scenario_groups = group_files_by_main_scenario(x_folder, y_folder, test_mode)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Split and copy files to appropriate directories
    counts = split_and_copy_files(scenario_groups, splits_dir, val_percent, test_percent)
    
    # Print summary of the splitting operation
    print("\nDataset split complete!")
    print(f"Total scenarios: {len(scenario_groups)}")
    print(f"Train: {counts['train']} files")
    print(f"Validation: {counts['val']} files")
    print(f"Test: {counts['test']} files")
    print(f"\nSplit data saved to: {splits_dir}")

if __name__ == "__main__":
    main()  # Execute main function when script is run directly 