# Post-Processing Script for LiDAR Data
# This script processes raw LiDAR scan data to extract object detection information
# It supports both old and new data formats, calculating relative positions, angles, and pixel coordinates
# The script can process all frames or a single selected frame for analysis

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import Object2D, KNOWN_OBJECTS, LidarScan

# --- Configuration Parameters ---
PROCESS_ALL_FRAMES = True  # Set to False to process only one frame
SELECTED_FRAME_INDEX = 100   # Index of the frame to process if not all

# Ensure door_frame is included in processing for this script
# This enables door frame detection in the LiDAR processing pipeline
for obj in KNOWN_OBJECTS:
    if obj.name == "door_frame":
        obj.should_scan = True

def find_latest_log_file(log_dir):
    """
    Find the latest log file in the specified directory.
    Searches for files matching the pattern "yolo1d_scan_*.jsonl" and returns the most recent one.
    
    Args:
        log_dir (str): Directory path to search for log files
    
    Returns:
        Path: Path to the most recent log file
    
    Raises:
        FileNotFoundError: If no log files are found in the directory
    """
    log_files = list(Path(log_dir).glob("yolo1d_scan_*.jsonl"))
    if not log_files:
        raise FileNotFoundError("No log files found in the specified directory")
    return max(log_files, key=lambda x: x.stat().st_mtime)

def scale_distance_to_pixel(distance, max_range=4.0):
    """
    Scale distance (0-4m) to pixel row (0-63).
    Converts real-world distance measurements to pixel coordinates for visualization.
    
    Args:
        distance (float): Distance in meters (0 to max_range)
        max_range (float): Maximum range in meters (default: 4.0)
    
    Returns:
        int: Pixel row index (0-63)
    """
    return int((distance / max_range) * 63)

def process_lidar_data(angles, distances, robot_pose):
    """
    Process LiDAR data using KNOWN_OBJECTS list.
    This function is designed for the old raw data format.
    Calculates relative positions, angles, and distances for all known objects.
    
    Args:
        angles (array): LiDAR angle measurements
        distances (array): LiDAR distance measurements
        robot_pose (tuple): Robot pose (x, y, theta)
    
    Returns:
        list: List of detected objects with their properties
    """
    robot_x, robot_y, robot_theta = robot_pose
    detected_objects = []
    
    for obj in KNOWN_OBJECTS:
        if not obj.should_scan:
            continue
        
        # Get object dimensions
        width, depth = obj.get_dimensions()
        
        # Calculate object position relative to robot
        global_rel_x = obj.x - robot_x
        global_rel_y = obj.y - robot_y
        
        # Apply rotation matrix to convert to robot's local coordinate frame
        # Rotation Matrix for theta (CW - clockwise)
        rel_x = np.cos(robot_theta) * global_rel_x + np.sin(robot_theta) * global_rel_y
        rel_y = -np.sin(robot_theta) * global_rel_x + np.cos(robot_theta) * global_rel_y
        
        # Calculate distance and angle from robot to object
        distance = np.sqrt(rel_x**2 + rel_y**2)
        angle = np.degrees(np.arctan2(rel_y, rel_x))
        
        # Convert to clockwise (CW) coordinate system
        if angle < 0:
            angle += 360
        angle = (360 - angle) % 360
        
        # Create detected object dictionary with all calculated properties
        detected_object = {
            "class": obj.name,
            "x_center": float(rel_x),
            "y_center": float(rel_y),
            "distance": float(distance),
            "center_angle": float(angle),
            "width": float(width),
            "depth": float(depth)
        }
        detected_objects.append(detected_object)
    
    return detected_objects

def process_lidar_data_new(angles, distances, robot_pose, object_details=None):
    """
    Process LiDAR data using only objects from object_details in raw data.
    This function is designed for the new raw data format.
    Excludes v_box from detection and calculates detailed corner information.
    Calculates relative positions, angles, distances, and pixel rows for both center and corners.
    
    Args:
        angles (array): LiDAR angle measurements
        distances (array): LiDAR distance measurements
        robot_pose (tuple): Robot pose (x, y, theta)
        object_details (dict): Dictionary containing object information from raw data
    
    Returns:
        list: List of detected objects with center and corner information
    """
    robot_x, robot_y, robot_theta = robot_pose
    detected_objects = []
    
    if object_details is None:
        return detected_objects
    
    # Process all objects from object_details except v_box
    for obj_name, obj in object_details.items():
        if obj_name == "v_box":  # Skip v_box objects
            continue
            
        # Get object dimensions from the object details
        width = obj["width"]
        depth = obj["height"]  # height in object_details is equivalent to depth
        
        # Get center coordinates of the object
        x_center = obj["center"][0]
        y_center = obj["center"][1]
        
        # Calculate center relative position and angle
        global_rel_x = x_center - robot_x
        global_rel_y = y_center - robot_y
        
        # Apply rotation matrix to convert to robot's local coordinate frame
        # Rotation Matrix for theta (CW - clockwise)
        rel_x = np.cos(robot_theta) * global_rel_x + np.sin(robot_theta) * global_rel_y
        rel_y = -np.sin(robot_theta) * global_rel_x + np.cos(robot_theta) * global_rel_y
        
        # Calculate center distance and angle from robot to object center
        center_distance = np.sqrt(rel_x**2 + rel_y**2)
        center_angle = np.degrees(np.arctan2(rel_y, rel_x))
        
        # Convert to clockwise (CW) coordinate system
        if center_angle < 0:
            center_angle += 360
        center_angle = (360 - center_angle) % 360
        
        # Calculate center pixel row for visualization
        center_pixel_row = scale_distance_to_pixel(center_distance)
        
        # Calculate corner positions and properties
        corners = []
        for corner in obj["corners"]:
            # Get corner coordinates
            x_corner = corner[0]
            y_corner = corner[1]
            
            # Calculate corner relative position
            global_rel_x_corner = x_corner - robot_x
            global_rel_y_corner = y_corner - robot_y
            
            # Apply rotation matrix to convert corner to robot's local coordinate frame
            # Rotation Matrix for theta (CW - clockwise)
            rel_x_corner = np.cos(robot_theta) * global_rel_x_corner + np.sin(robot_theta) * global_rel_y_corner
            rel_y_corner = -np.sin(robot_theta) * global_rel_x_corner + np.cos(robot_theta) * global_rel_y_corner
            
            # Calculate corner distance and angle from robot to corner
            corner_distance = np.sqrt(rel_x_corner**2 + rel_y_corner**2)
            corner_angle = np.degrees(np.arctan2(rel_y_corner, rel_x_corner))
            
            # Convert to clockwise (CW) coordinate system
            if corner_angle < 0:
                corner_angle += 360
            corner_angle = (360 - corner_angle) % 360
            
            # Calculate corner pixel row for visualization
            corner_pixel_row = scale_distance_to_pixel(corner_distance)
            
            # Store corner information
            corners.append({
                "x": float(rel_x_corner),
                "y": float(rel_y_corner),
                "distance": float(corner_distance),
                "angle": float(corner_angle),
                "pixel_row": corner_pixel_row
            })
        
        # Create detected object dictionary with center and corner information
        detected_object = {
            "class": obj_name,
            "center": {
                "x": float(rel_x),
                "y": float(rel_y),
                "distance": float(center_distance),
                "angle": float(center_angle),
                "pixel_row": center_pixel_row
            },
            "corners": corners,
            "width": float(width),
            "depth": float(depth)
        }
        detected_objects.append(detected_object)
    
    return detected_objects

def main():
    """
    Main function that orchestrates the LiDAR data processing pipeline.
    Reads the latest log file, processes frames according to configuration,
    and saves the results to output files.
    """
    # Set up file paths and directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = find_latest_log_file(log_dir)
    print(f"Processing log file: {latest_log}")
    output_dir = Path("output/post_process_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_name = Path(latest_log).stem.replace('.jsonl', '')

    # Read all lines from the log file
    with open(latest_log, 'r') as f:
        lines = f.readlines()

    if PROCESS_ALL_FRAMES:
        # Process all frames in the log file
        all_frames = []
        total_frames = len(lines)
        for idx, line in enumerate(lines):
            # Show progress every 2% or on the last frame
            if idx % max(1, total_frames // 50) == 0 or idx == total_frames - 1:
                percent = (idx + 1) / total_frames * 100
                print(f"Processing frame {idx + 1}/{total_frames} ({percent:.1f}%)", end='\r')
            
            # Parse JSON data and create LidarScan object
            data = json.loads(line)
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            angles = scan.angles
            distances = scan.raw_scan
            pose = scan.pose
            
            # Process LiDAR data to detect objects
            detected_objects = process_lidar_data(angles, distances, pose)
            
            # Create frame output with detected objects
            frame_output = {
                "frame_id": scan.frame_id,
                "objects": detected_objects
            }
            all_frames.append(frame_output)
        
        print()  # Newline after progress bar
        
        # Save all processed frames to a single JSON file
        output_file = output_dir / f"processed_{raw_log_name}_all_frames.json"
        with open(output_file, 'w') as out_f:
            json.dump(all_frames, out_f, indent=4)
        print(f"Results saved to {output_file}")
    else:
        # Process only the selected frame
        data = json.loads(lines[SELECTED_FRAME_INDEX])
        scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
        angles = scan.angles
        distances = scan.raw_scan
        pose = scan.pose
        
        # Process LiDAR data to detect objects
        detected_objects = process_lidar_data(angles, distances, pose)
        
        # Create frame output with detected objects
        frame_output = {
            "frame_id": scan.frame_id,
            "objects": detected_objects
        }
        
        # Save single frame results to JSON file
        output_file = output_dir / f"processed_{raw_log_name}_frame{scan.frame_id}.json"
        with open(output_file, 'w') as out_f:
            json.dump(frame_output, out_f, indent=4)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()  # Execute main function when script is run directly
