# Multi-frame scenarios aligned fused label generator
# This script processes LiDAR scan data to generate YOLO format labels for aligned fused RGB images
# It uses a buffer system to process multiple frames and generates labels for the last frame
# Labels are created for detected objects like chairs, boxes, desks, and door frames
# This version corresponds to the aligned fused RGB generator and uses the last frame for label generation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_process import process_lidar_data_new
from bot3.utils.lidar_utils import LidarScan, KNOWN_OBJECTS
import json
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
from collections import deque

# Variables to extend bounding box width and height (in pixels)
# These extensions help ensure the bounding box fully encompasses the detected objects
WIDTH_EXTENSION = 25  # e.g., set to 10 to add 10 pixels to width
HEIGHT_EXTENSION = 15  # e.g., set to 4 to add 4 pixels to height

def get_class_id(obj_name):
    """
    Get class ID based on object name for YOLO format.
    
    Args:
        obj_name (str): Name of the object class
    
    Returns:
        int: Class ID (0-3) or -1 if not found
    """
    class_mapping = {
        "chair": 0,
        "box": 1,
        "desk": 2,
        "door_frame": 3
    }
    return class_mapping.get(obj_name, -1)

def clamp_yolo_box(center, width, min_angle=0, max_angle=359):
    """
    Clamp YOLO bounding box to ensure it stays within valid angle range.
    Adjusts center and width if the box extends beyond the valid range.
    
    Args:
        center (float): Center angle of the bounding box
        width (float): Width of the bounding box in angles
        min_angle (int): Minimum valid angle (default: 0)
        max_angle (int): Maximum valid angle (default: 359)
    
    Returns:
        tuple: (adjusted_center, adjusted_width, left_edge, right_edge)
    """
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
    """
    Calculate the minimal arc width that encompasses all given angles.
    This is used to determine the optimal bounding box width.
    
    Args:
        angles (list): List of angle values
    
    Returns:
        float: Minimal arc width in degrees
    """
    angles = sorted(a % 360 for a in angles)  # Sort and normalize angles to 0-359
    gaps = [(angles[(i+1)%4] - angles[i]) % 360 for i in range(4)]  # Calculate gaps between consecutive angles
    max_gap = max(gaps)  # Find the largest gap
    return 360 - max_gap  # Return the complement of the largest gap

def adjust_center_for_close_objects(corners, center):
    """
    Adjust center position for desk and chair objects when they are too close to robot.
    This function ensures the center point is within the bounds of the object corners.
    
    Args:
        corners: List of corner points with pixel_row information
        center: Center point with angle and pixel_row information
    
    Returns:
        dict: Adjusted center point with corrected pixel_row if needed
    """
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
        return {'angle': center['angle'], 'pixel_row': new_center_row}
    
    return center

class LabelBuffer:
    """
    A buffer class that maintains a fixed-size queue of frame data.
    Used to process multiple frames and extract labels from the last frame.
    This version corresponds to the aligned fused RGB generator.
    """
    def __init__(self, buffer_size=5):
        """
        Initialize the label buffer with a specified buffer size.
        
        Args:
            buffer_size (int): Number of frames to keep in buffer (default: 5 for aligned fusion)
        """
        self.buffer_size = buffer_size
        self.frames = deque(maxlen=buffer_size)  # FIFO queue with max length
    
    def add_frame(self, frame_data):
        """
        Add a new frame to buffer (FIFO).
        
        Args:
            frame_data: Frame data to add to the buffer
        """
        self.frames.append(frame_data)
    
    def get_last_frame(self):
        """
        Get the last frame from buffer when full.
        This is used for label generation as it corresponds to the most recent frame.
        
        Returns:
            dict: Last frame data or None if buffer not full
        """
        if len(self.frames) < self.buffer_size:
            return None
        return self.frames[-1]  # Return last frame (index 4)

if __name__ == "__main__":
    # Find latest log file in the raw_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)  # Get most recent log file
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]  # Extract filename without extension
    
    # Create output directory for labels - matching RGB generator structure
    # Uses 'Y_' prefix to distinguish from RGB files ('X_' prefix) and '5_frame' to match aligned fusion
    output_dir = Path("output/multi_frame/multi_scenarios/aligned_fused") / f"Y_5_frame_aligned_fused_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count total lines in log file for progress bar
    with open(latest_log, 'r') as f:
        total_frames = sum(1 for _ in f)

    total_scenarios = 1  # Initialize with default value

    # Get total scenarios from the last frame (look for scenarios_completed field)
    with open(latest_log, 'r') as f:
        for line in f:
            data = json.loads(line)
            if "scenarios_completed" in data:
                total_scenarios = data["scenarios_completed"]
                break

    # Process frames from log file
    frame_count = 0  # Total frames processed
    current_scenario = None  # Current scenario being processed
    scenario_frame_count = 0  # Frames within current scenario
    global_frame_counter = 0  # Global counter for all frames across all scenarios
    buffer = LabelBuffer(buffer_size=5)  # Initialize buffer for storing frames

    with open(latest_log, 'r') as f:
        pbar = tqdm(f, total=total_frames, desc="Processing frames", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {postfix} [{elapsed}<{remaining}, {rate_fmt}]')
        for line in pbar:
            data = json.loads(line)
            
            # Skip summary/logging frames and invalid data (frames without raw_scan)
            if "status" in data or "raw_scan" not in data:
                continue

            # Check if we're in a new scenario
            if current_scenario != data.get("scenario_number"):
                buffer = LabelBuffer(buffer_size=5)  # Reset buffer for new scenario to avoid mixing frames
                current_scenario = data.get("scenario_number")
                scenario_frame_count = 0
                pbar.set_postfix_str(f"scenario={current_scenario}/{total_scenarios-1} , frames={frame_count}/{total_frames}")

            # Add frame to buffer
            buffer.add_frame(data)
            
            # Process and save labels after buffer is full (after 5th frame)
            if scenario_frame_count >= 4:  # buffer_size - 1 (need 5 frames to start generating)
                last_frame = buffer.get_last_frame()
                if last_frame is not None:
                    # Create LidarScan object from last frame data
                    scan = LidarScan(last_frame["raw_scan"], last_frame["pose"], 
                                   frame_id=last_frame.get("frame_id", frame_count))
                    angles = scan.angles
                    distances = scan.raw_scan
                    pose = scan.pose
                    robot_theta = pose[2]  # Robot orientation
                    object_details = last_frame.get('object_details', None)  # Ground truth object information
                    detected_objects = process_lidar_data_new(angles, distances, pose, object_details)  # Process LiDAR data

                    # Prepare label file name with scenario and global counter
                    frame_num = scenario_frame_count - 4  # buffer_size - 1 (adjust for buffer offset)
                    label_file = f"frame_{frame_num}_{current_scenario}_{data.get('main_scenario_id', current_scenario)}_{global_frame_counter}.txt"
                    output_file = os.path.join(output_dir, label_file)

                    with open(output_file, 'w') as f_out:
                        for obj in detected_objects:
                            # Skip objects that are too far (pixel_row >= 56) to avoid distant false positives
                            if obj['center']['pixel_row'] >= 56:
                                continue
                                
                            class_id = get_class_id(obj['class'])  # Get YOLO class ID
                            # Use pixel rows directly from the new format
                            angles_c = [corner['angle'] for corner in obj['corners']]  # Extract corner angles
                            pixel_rows = [corner['pixel_row'] for corner in obj['corners']]  # Extract corner pixel rows
                            
                            # Apply center adjustment for desk and chair if needed (handles close objects)
                            if obj['class'] in ['desk', 'chair']:
                                obj['center'] = adjust_center_for_close_objects(obj['corners'], obj['center'])
                            
                            # Use minimal arc for width calculation
                            width = minimal_arc_width(angles_c) + WIDTH_EXTENSION
                            # Clamp box to [0, 359] angle range to ensure valid bounds
                            center_angle_clamped, width_clamped, _, _ = clamp_yolo_box(obj['center']['angle'], width, 0, 359)
                            min_row = min(pixel_rows)  # Minimum pixel row
                            max_row = max(pixel_rows)  # Maximum pixel row
                            row_diff = abs(max_row - min_row) + HEIGHT_EXTENSION  # Calculate height with extension
                            # Normalize coordinates for YOLO format (0-1 range)
                            center_x_norm = round(center_angle_clamped / 359, 6)  # Normalize x center (angle)
                            center_y_norm = round(obj['center']['pixel_row'] / 63, 6)  # Normalize y center (row)
                            width_norm = round(width_clamped / 359, 6)  # Normalize width (angle)
                            height_norm = round(row_diff / 63, 6)  # Normalize height (row)
                            # Write YOLO format line: class_id center_x center_y width height
                            f_out.write(f"{class_id} {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n")
                    
                    global_frame_counter += 1

            frame_count += 1
            scenario_frame_count += 1

    print(f"Labels saved to {output_dir}")  # Final confirmation message 