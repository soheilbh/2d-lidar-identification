import os
import json
import glob
import shutil
from datetime import datetime
from tqdm import tqdm

def get_label_files(fusion_type, scan_time):
    """Get X and Y label files from the specified folders."""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(base_dir, 'output', 'multi_frame', 'multi_scenarios', fusion_type)
    
    # Set frame count based on fusion type
    frame_count = "5_frame" if fusion_type == "aligned_fused" else "3_frame"
    
    # Find X and Y folders matching the scan time
    x_folder = f'X_{frame_count}_{fusion_type}_yolo1d_scan_{scan_time}'
    y_folder = f'Y_{frame_count}_{fusion_type}_yolo1d_scan_{scan_time}'
    
    x_folder_path = os.path.join(output_dir, x_folder)
    y_folder_path = os.path.join(output_dir, y_folder)
    
    if not os.path.exists(x_folder_path) or not os.path.exists(y_folder_path):
        raise FileNotFoundError(f"X or Y folders not found in {output_dir} for scan time {scan_time}")
    
    return x_folder_path, y_folder_path, output_dir, x_folder, y_folder

def create_new_label_files(x_folder, y_folder, scenarios_to_remove, output_dir, x_folder_name, y_folder_name):
    """Create new X and Y label files without the specified scenarios."""
    # Create new directories with 'cleaned' prefix
    new_x_folder = os.path.join(output_dir, f'cleaned_{x_folder_name}')
    new_y_folder = os.path.join(output_dir, f'cleaned_{y_folder_name}')
    os.makedirs(new_x_folder, exist_ok=True)
    os.makedirs(new_y_folder, exist_ok=True)
    
    # Get all files
    x_files = glob.glob(os.path.join(x_folder, '*.png'))
    y_files = glob.glob(os.path.join(y_folder, '*.txt'))
    
    if not x_files or not y_files:
        raise FileNotFoundError(f"No PNG or TXT files found in X or Y folders")
    
    # Process X files (PNG)
    kept_files = []
    removed_files = []
    print("\nProcessing X files...")
    for x_file in tqdm(x_files, desc="Processing PNG files"):
        # Extract scenario number from filename (frame_w_x_y_z.png)
        try:
            filename = os.path.basename(x_file)
            parts = filename.split('_')
            if len(parts) >= 4:
                sub_scenario = int(parts[2])  # x in w_x_y_z
                if sub_scenario not in scenarios_to_remove:
                    kept_files.append(filename)
                    shutil.copy2(x_file, os.path.join(new_x_folder, filename))
                else:
                    removed_files.append(filename)
        except (IndexError, ValueError):
            # If filename doesn't match expected format, keep it
            kept_files.append(filename)
            shutil.copy2(x_file, os.path.join(new_x_folder, filename))
    
    # Process Y files (TXT)
    print("\nProcessing Y files...")
    for y_file in tqdm(y_files, desc="Processing TXT files"):
        # Extract scenario number from filename (frame_w_x_y_z.txt)
        try:
            filename = os.path.basename(y_file)
            parts = filename.split('_')
            if len(parts) >= 4:
                sub_scenario = int(parts[2])  # x in w_x_y_z
                if sub_scenario not in scenarios_to_remove:
                    shutil.copy2(y_file, os.path.join(new_y_folder, filename))
        except (IndexError, ValueError):
            # If filename doesn't match expected format, keep it
            shutil.copy2(y_file, os.path.join(new_y_folder, filename))
    
    print(f"\nProcessed files:")
    print(f"Original X folder: {x_folder}")
    print(f"Original Y folder: {y_folder}")
    print(f"New X folder: {new_x_folder}")
    print(f"New Y folder: {new_y_folder}")
    print(f"\nKept {len(kept_files)} files")
    print(f"Removed {len(removed_files)} files with sub-scenario numbers: {sorted(scenarios_to_remove)}")
    
    return new_x_folder, new_y_folder

def main():
    # Get fusion type (simple_fused/aligned_fused)
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
    
    # Get scan time
    print("\nEnter scan time (e.g., 2025-06-05_17-13-28):")
    scan_time = input().strip()
    
    # Get the label files
    try:
        x_folder, y_folder, output_dir, x_folder_name, y_folder_name = get_label_files(fusion_type, scan_time)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Get scenarios to remove from user
    print("\nEnter the sub-scenario numbers to remove (comma-separated, e.g., '1,2,3'):")
    print("This will remove all files where the second number in w_x_y_z matches your input")
    scenarios_input = input().strip()
    
    try:
        scenarios_to_remove = [int(x.strip()) for x in scenarios_input.split(',') if x.strip()]
    except ValueError:
        print("Error: Please enter valid numbers separated by commas")
        return
    
    if not scenarios_to_remove:
        print("No scenarios specified for removal")
        return
    
    # Create new files without the specified scenarios
    create_new_label_files(x_folder, y_folder, scenarios_to_remove, output_dir, x_folder_name, y_folder_name)

if __name__ == "__main__":
    main() 