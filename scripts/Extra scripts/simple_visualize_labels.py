import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from pathlib import Path

def draw_boxes(image_path, label_path):
    """
    Draw bounding boxes on a single image using its corresponding label file.
    
    Args:
        image_path: Path to the RGB image
        label_path: Path to the YOLO format label file
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Define colors for different classes (BGR format)
    colors = {
        0: (0, 255, 0),    # Green for chair
        1: (0, 0, 255),    # Red for box
        2: (0, 255, 255),  # Yellow for desk
        3: (255, 255, 0)   # Cyan for door frame
    }
    
    # Read and process labels
    with open(label_path, 'r') as f:
        for line in f:
            # Parse normalized label
            parts = line.strip().split()
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
    
    # Display the image
    cv2.imshow('Labeled Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage - modify these paths as needed
    image_path = "output/multi_frame/multi_scenarios/simple_fused/X_3_frame_simple_fused_yolo1d_scan_2025-06-01_19-14-36/frame_0000_001_000762.png"
    label_path = "output/multi_frame/multi_scenarios/simple_fused/Y_3_frame_simple_fused_yolo1d_scan_2025-06-01_19-14-36/frame_0000_001_000762.txt"
    
    # Draw boxes on the image
    draw_boxes(image_path, label_path) 