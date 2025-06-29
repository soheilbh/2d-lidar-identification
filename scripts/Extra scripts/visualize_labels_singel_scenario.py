import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def draw_bounding_boxes(image_path, labels_path, output_path):
    """
    Draw bounding boxes on range image based on labels.
    
    Args:
        image_path: Path to range image (360x64)
        labels_path: Path to labels file
        output_path: Path to save the visualization
    """
    # Read the range image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Read labels
    with open(labels_path, 'r') as f:
        labels = f.readlines()
    
    # Define colors for different classes (BGR format)
    colors = {
        0: (0, 255, 0),    # Green for chair
        1: (0, 0, 255),    # Red for box
        2: (0, 255, 255),    # Yellow for desk
        3: (255, 255, 0)   # Cyan for door frame
    }
    
    # Process each label
    for label in labels:
        # Parse normalized label
        parts = label.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            center_x_norm = float(parts[1])
            center_y_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            # De-normalize
            center_angle = int(center_x_norm * 359)
            center_row = int(center_y_norm * 63)
            angle_width = int(width_norm * 359)
            row_height = int(height_norm * 63)
        else:
            # Fallback for old format (all ints)
            class_id, center_angle, center_row, angle_width, row_height = map(int, parts)
        
        # Calculate bounding box corners
        # Note: In range image, x is angle (0-359) and y is row (0-63)
        x1 = center_angle - angle_width // 2
        x2 = center_angle + angle_width // 2
        y1 = center_row - row_height // 2
        y2 = center_row + row_height // 2
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, 359))
        x2 = max(0, min(x2, 359))
        y1 = max(0, min(y1, 63))
        y2 = max(0, min(y2, 63))
        
        # Draw rectangle
        color = colors.get(class_id, (255, 255, 255))  # White for unknown classes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        
        # Add class label
        class_names = {0: "Chair", 1: "Box", 2: "Desk", 3: "Door"}
        label = class_names.get(class_id, f"Class {class_id}")
        cv2.putText(image, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw center point
        cv2.circle(image, (center_angle, center_row), 2, (255, 255, 255), -1)
    
    # Save the result
    cv2.imwrite(output_path, image)
    return image

def create_video(image_paths, output_path, fps=30):
    """
    Create a video from a list of image paths.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save the video
        fps: Frames per second
    """
    if not image_paths:
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_paths[0]}")
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Add frames to video
    for image_path in tqdm(image_paths, desc="Creating video"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not read frame {image_path}")
            continue
        if frame.shape[:2] != (height, width):
            print(f"Warning: Frame {image_path} has different dimensions: {frame.shape[:2]} vs {(height, width)}")
            frame = cv2.resize(frame, (width, height))
        video.write(frame)
    
    # Release video writer
    video.release()

if __name__ == "__main__":
    # Find latest log file to get the raw log name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Setup input and output directories
    range_images_dir = Path("output/multi_frame/range_images") / f"X_{raw_log_name}"
    labels_dir = Path("output/multi_frame/labels") / f"Y_{raw_log_name}"
    output_dir = Path("output/multiframe_visualize_label") / raw_log_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files and sort them numerically
    frame_files = sorted([f for f in range_images_dir.glob("frame_*.png")], 
                        key=lambda x: int(x.stem.split('_')[1]))
    
    # Process each frame and collect output paths
    output_paths = []
    for frame_file in tqdm(frame_files, desc="Visualizing labels"):
        frame_num = frame_file.stem.split('_')[1]  # Get frame number from filename
        image_path = str(frame_file)
        labels_path = str(labels_dir / f"frame_{frame_num}.txt")
        output_path = output_dir / f"visualize_frame_label_{frame_num}.png"
        
        # Draw bounding boxes
        try:
            draw_bounding_boxes(image_path, labels_path, str(output_path))
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            continue
    
    # Create video from all frames
    if output_paths:
        video_path = output_dir / f"{raw_log_name}_visualization.mp4"
        create_video(output_paths, video_path)
        print(f"All visualizations saved to {output_dir}")
        print(f"Video saved to {video_path}")
    else:
        print("No frames were processed successfully") 