"""
Visualization script for simple fused LiDAR data with bounding box labels.

This script processes simple fused range images and their corresponding YOLO format labels
to create visualizations with bounding boxes drawn on the images. It can also create
a video from the visualizations for easier review of the dataset.

The script handles both normalized and non-normalized label formats and supports
visualization of both normal and cleaned datasets. Simple fused data combines multiple
frames without pose-based alignment, using basic concatenation or averaging.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Configuration flag: Set to True to visualize cleaned data, False for normal data
USE_CLEANED_DATA = False    

def draw_bounding_boxes(image_path, labels_path, output_path):
    """
    Draw bounding boxes on range image based on YOLO format labels.
    
    This function reads a range image (360x64 pixels) and its corresponding label file,
    then draws colored bounding boxes for each detected object. The function handles
    both normalized (0-1) and non-normalized label formats.
    
    Args:
        image_path (str): Path to the range image file (PNG format, 360x64 pixels)
        labels_path (str): Path to the YOLO format labels file (.txt)
        output_path (str): Path where the visualization will be saved
    
    Raises:
        FileNotFoundError: If the image file cannot be read
    """
    # Read the range image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Read labels from the text file
    with open(labels_path, 'r') as f:
        labels = f.readlines()
    
    # Define colors for different object classes in BGR format (OpenCV default)
    colors = {
        0: (0, 255, 0),    # Green for chair
        1: (0, 0, 255),    # Red for box
        2: (0, 255, 255),  # Yellow for desk
        3: (255, 255, 0)   # Cyan for door frame
    }
    
    # Process each label line
    for label in labels:
        # Parse the normalized label format: class_id center_x center_y width height
        parts = label.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            center_x_norm = float(parts[1])  # Normalized center x (0-1)
            center_y_norm = float(parts[2])  # Normalized center y (0-1)
            width_norm = float(parts[3])     # Normalized width (0-1)
            height_norm = float(parts[4])    # Normalized height (0-1)
            
            # De-normalize coordinates to pixel values
            center_angle = int(center_x_norm * 359)  # Convert to 0-359 range
            center_row = int(center_y_norm * 63)     # Convert to 0-63 range
            angle_width = int(width_norm * 359)      # Width in angle units
            row_height = int(height_norm * 63)       # Height in row units
        else:
            # Fallback for old format where all values are integers
            class_id, center_angle, center_row, angle_width, row_height = map(int, parts)
        
        # Calculate bounding box corner coordinates
        # Note: In range image, x represents angle (0-359) and y represents row (0-63)
        x1 = center_angle - angle_width // 2  # Left edge
        x2 = center_angle + angle_width // 2  # Right edge
        y1 = center_row - row_height // 2     # Top edge
        y2 = center_row + row_height // 2     # Bottom edge
        
        # Ensure coordinates are within image bounds to prevent drawing errors
        x1 = max(0, min(x1, 359))
        x2 = max(0, min(x2, 359))
        y1 = max(0, min(y1, 63))
        y2 = max(0, min(y2, 63))
        
        # Draw rectangle with class-specific color
        color = colors.get(class_id, (255, 255, 255))  # White for unknown classes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        
        # Add class label text above the bounding box
        class_names = {0: "Chair", 1: "Box", 2: "Desk", 3: "Door"}
        label = class_names.get(class_id, f"Class {class_id}")
        cv2.putText(image, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw center point as a small white circle
        cv2.circle(image, (center_angle, center_row), 2, (255, 255, 255), -1)
    
    # Save the annotated image
    cv2.imwrite(output_path, image)
    return image

def create_video(image_paths, output_path, fps=30):
    """
    Create a video from a list of image paths with filename overlay.
    
    This function creates an MP4 video from a sequence of visualization images.
    Each frame includes the original image with a padded area at the top showing
    the filename for easy identification during playback.
    
    Args:
        image_paths (list): List of paths to image files
        output_path (str): Path where the video will be saved
        fps (int): Frames per second for the video (default: 30)
    
    Raises:
        ValueError: If the first image cannot be read
    """
    if not image_paths:
        return
    
    # Read first image to get dimensions for video setup
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    height, width = first_image.shape[:2]
    
    # Add padding at the top for filename display
    padding_height = 30
    total_height = height + padding_height
    
    # Create video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, total_height))
    
    # Process each image and add to video
    for image_path in tqdm(image_paths, desc="Creating video"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not read frame {image_path}")
            continue
        
        # Handle frames with different dimensions by resizing
        if frame.shape[:2] != (height, width):
            print(f"Warning: Frame {image_path} has different dimensions: {frame.shape[:2]} vs {(height, width)}")
            frame = cv2.resize(frame, (width, height))
        
        # Create padded frame with black background
        padded_frame = np.zeros((total_height, width, 3), dtype=np.uint8)
        padded_frame[padding_height:, :] = frame  # Place original image below padding
        
        # Extract and format filename for display
        filename = os.path.basename(image_path)
        # Remove 'visualize_' prefix and '.png' extension for cleaner display
        filename = filename.replace('visualize_', '').replace('.png', '')
        
        # Add filename text to the padding area
        cv2.putText(padded_frame, filename, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Write frame to video
        video.write(padded_frame)
    
    # Release video writer to finalize the video file
    video.release()

if __name__ == "__main__":
    # Find the latest log file to determine the raw log name for directory setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Setup input and output directories for simple fused data
    # Use cleaned data if specified, otherwise use normal data
    prefix = "cleaned_" if USE_CLEANED_DATA else ""
    data_type = "cleaned" if USE_CLEANED_DATA else "normal"
    
    # Define paths for input data (range images and labels)
    # Note: Simple fused uses 3 frames instead of 5 frames like aligned fused
    range_images_dir = Path("output/multi_frame/multi_scenarios/simple_fused") / f"{prefix}X_3_frame_simple_fused_{raw_log_name}"
    labels_dir = Path("output/multi_frame/multi_scenarios/simple_fused") / f"{prefix}Y_3_frame_simple_fused_{raw_log_name}"
    
    # Define output directory for visualizations
    output_dir = Path("output/multi_frame/multi_scenarios/simple_fused_visualize") / f"visualize_{data_type}_3_frame_simple_fused_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files and sort them numerically by global counter
    # Filename format: frame_w_x_y_z.png where z is the global counter
    frame_files = sorted([f for f in range_images_dir.glob("frame_*.png")], 
                        key=lambda x: int(x.stem.split('_')[4]))  # Sort by global counter (index 4)
    
    # Process each frame and collect output paths for video creation
    output_paths = []
    for frame_file in tqdm(frame_files, desc="Visualizing labels"):
        # Parse filename components: frame_w_x_y_z.png
        parts = frame_file.stem.split('_')
        frame_num = parts[1]      # w: frame number within scenario
        scenario_num = parts[2]   # x: scenario number
        main_scenario_num = parts[3]  # y: main scenario number
        global_counter = parts[4]     # z: global counter across all scenarios
        
        # Construct paths for input and output files
        image_path = str(frame_file)
        labels_path = str(labels_dir / f"frame_{frame_num}_{scenario_num}_{main_scenario_num}_{global_counter}.txt")
        output_path = output_dir / f"visualize_frame_{frame_num}_{scenario_num}_{main_scenario_num}_{global_counter}.png"
        
        # Draw bounding boxes and save visualization
        try:
            draw_bounding_boxes(image_path, labels_path, str(output_path))
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing frame {frame_num} scenario {scenario_num}: {e}")
            continue
    
    # Create video from all processed frames if any were successful
    if output_paths:
        video_path = output_dir / f"visualize_{data_type}_3_frame_simple_fused_{raw_log_name}.mp4"
        create_video(output_paths, video_path)
        print(f"All visualizations saved to {output_dir}")
        print(f"Video saved to {video_path}")
    else:
        print("No frames were processed successfully") 