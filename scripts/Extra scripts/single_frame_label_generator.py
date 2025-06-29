import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_process import process_lidar_data_new
from bot3.utils.lidar_utils import LidarScan, KNOWN_OBJECTS
import json
import numpy as np
import math
from pathlib import Path

# Variables to extend bounding box width and height (in pixels)
WIDTH_EXTENSION = 22  # e.g., set to 10 to add 10 pixels to width
HEIGHT_EXTENSION = 12  # e.g., set to 4 to add 4 pixels to height

def apply_fallback_labeling(corners, center):
    """Apply fallback labeling for desk and chair objects when center is too close to robot"""
    # Get corner distances (rows)
    corner_rows = [corner['pixel_row'] for corner in corners]
    center_row = center['pixel_row']
    
    # Check if center's row is between min and max corner rows
    min_corner_row = min(corner_rows)
    max_corner_row = max(corner_rows)
    
    # If center is not within range, adjust it
    if not (min_corner_row <= center_row <= max_corner_row):
        # Calculate new center row (average of corner rows)
        new_center_row = int(np.mean(corner_rows))
        # Keep the same angle, just update the row
        return {'angle': center['angle'], 'pixel_row': new_center_row}, True
    
    return center, False

def get_class_id(obj_name):
    """Get class ID based on object name"""
    class_mapping = {
        "chair": 0,
        "box": 1,
        "desk": 2,
        "door_frame": 3
    }
    return class_mapping.get(obj_name, -1)

def clamp_yolo_box(center, width, min_angle=0, max_angle=359):
    half_w = width / 2
    left_edge = center - half_w
    right_edge = center + half_w
    # Clamp right edge
    if right_edge > max_angle:
        delta = right_edge - max_angle
        center = center - (delta/2)
        width = width -  delta
        left_edge = center - width / 2
        right_edge = center + width / 2
    # Clamp left edge
    if left_edge < min_angle:
        delta = min_angle - left_edge
        center = center + delta/2
        width = width - delta
        left_edge = center - width / 2
        right_edge = center + width / 2
    return center, width, left_edge, right_edge

def minimal_arc_width(angles):
    angles = sorted(a % 360 for a in angles)
    gaps = [(angles[(i+1)%4] - angles[i]) % 360 for i in range(4)]
    max_gap = max(gaps)
    return 360 - max_gap

if __name__ == "__main__":
    # Specify the exact log file and frame number
    target_log_file = "yolo1d_scan_2025-06-01_19-14-36.jsonl"
    target_frame = 765
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    log_file = os.path.join(log_dir, target_log_file)
    
    # Read and process the specific frame
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if target_frame < len(lines):
            data = json.loads(lines[target_frame])
            
            # Skip if it's a summary/logging frame or invalid data
            if "status" in data or "raw_scan" not in data:
                print(f"Frame {target_frame} is not a valid data frame")
                sys.exit(1)
            
            # Process the frame
            scan = LidarScan(data["raw_scan"], data["pose"], 
                           frame_id=data.get("frame_id", target_frame))
            angles = scan.angles
            distances = scan.raw_scan
            pose = scan.pose
            robot_theta = pose[2]
            object_details = data.get('object_details', None)
            detected_objects = process_lidar_data_new(angles, distances, pose, object_details)
            
            print(f"\nProcessing Line {target_frame} from file")
            print(f"Actual Frame ID: {data.get('frame_id', 'Not specified')}")
            print("=" * 50)
            print(f"Robot Position: x={pose[0]:.2f}, y={pose[1]:.2f}, theta={pose[2]:.2f} rad")
            print(f"Number of detected objects: {len(detected_objects)}")
            print("\nDetected Objects:")
            print("-" * 50)
            
            # Print first object's structure to understand available fields
            if detected_objects:
                print("\nFirst object's data structure:")
                print(json.dumps(detected_objects[0], indent=2))
                print("-" * 50)
            
            for obj in detected_objects:
                class_id = get_class_id(obj['class'])
                # Use pixel rows directly from the new format
                angles_c = [corner['angle'] for corner in obj['corners']]
                pixel_rows = [corner['pixel_row'] for corner in obj['corners']]
                
                # Apply fallback labeling for desk and chair if needed
                if obj['class'] in ['desk', 'chair']:
                    original_center = obj['center']
                    obj['center'], was_adjusted = apply_fallback_labeling(obj['corners'], obj['center'])
                    print(f"\n{obj['class'].capitalize()} Center Debug:")
                    print(f"  Original Center: angle={original_center['angle']:.2f}°, row={original_center['pixel_row']}")
                    print(f"  Corner Rows: {pixel_rows}")
                    if was_adjusted:
                        print(f"  Adjusted Center: angle={obj['center']['angle']:.2f}°, row={obj['center']['pixel_row']}")
                    else:
                        print("  No adjustment needed - center already within corner row range")
                
                # Use minimal arc for width
                width = minimal_arc_width(angles_c) + WIDTH_EXTENSION
                # Clamp box to [0, 359]
                center_angle_clamped, width_clamped, _, _ = clamp_yolo_box(obj['center']['angle'], width, 0, 359)
                min_row = min(pixel_rows)
                max_row = max(pixel_rows)
                row_diff = abs(max_row - min_row) + HEIGHT_EXTENSION
                # Normalize for YOLO format
                center_x_norm = round(center_angle_clamped / 359, 6)
                center_y_norm = round(obj['center']['pixel_row'] / 63, 6)
                width_norm = round(width_clamped / 359, 6)
                height_norm = round(row_diff / 63, 6)
                
                print(f"\nObject: {obj['class']} (Class ID: {class_id})")
                print(f"Center in polar: angle={obj['center']['angle']:.2f}°, row={obj['center']['pixel_row']}")
                print(f"Corners: {[(c['angle'], c['pixel_row']) for c in obj['corners']]}")
                print(f"YOLO Format: {class_id} {center_x_norm} {center_y_norm} {width_norm} {height_norm}")
                print("-" * 30)
        else:
            print(f"Frame {target_frame} not found in log file") 