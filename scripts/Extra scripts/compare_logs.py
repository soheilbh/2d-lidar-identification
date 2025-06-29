import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_process import process_lidar_data, process_lidar_data_new
from bot3.utils.lidar_utils import LidarScan
import json
import numpy as np
from pathlib import Path

def load_and_process_frame(log_path):
    """Load first frame from log and process it with both functions"""
    with open(log_path, 'r') as f:
        first_line = f.readline().strip()
        frame_data = json.loads(first_line)
        
    # Extract data
    frame_id = frame_data['frame_id']
    pose = frame_data['pose']
    raw_scan = frame_data['raw_scan']
    object_details = frame_data.get('object_details', None)  # Get object_details if available
    
    # Create LidarScan object
    lidar_scan = LidarScan(raw_scan, pose, frame_id=frame_id)
    
    # Process with both functions
    old_detected = process_lidar_data(lidar_scan.angles, lidar_scan.raw_scan, pose)
    new_detected = process_lidar_data_new(lidar_scan.angles, lidar_scan.raw_scan, pose, object_details)
    
    return {
        'frame_id': frame_id,
        'pose': pose,
        'old_detected': old_detected,
        'new_detected': new_detected
    }

def compare_outputs(old_log, new_log):
    """Compare the outputs of process_lidar_data for both logs"""
    print(f"\nProcessing old log: {old_log}")
    old_output = load_and_process_frame(old_log)
    
    print(f"\nProcessing new log: {new_log}")
    new_output = load_and_process_frame(new_log)
    
    print("\nComparison Results:")
    print("-" * 50)
    
    # Show detected objects from process_lidar_data (old raw data)
    print("\nDetected Objects (process_lidar_data - old raw data):")
    print(f"Number of objects: {len(old_output['old_detected'])}")
    for obj in old_output['old_detected']:
        print(f"\nObject: {obj['class']}")
        print(f"Center: x={obj['x_center']:.3f}, y={obj['y_center']:.3f}, "
              f"dist={obj['distance']:.3f}, angle={obj['center_angle']:.3f}")
    
    # Show detected objects from process_lidar_data_new (new raw data)
    print("\nDetected Objects (process_lidar_data_new - new raw data):")
    print(f"Number of objects: {len(new_output['new_detected'])}")
    for obj in new_output['new_detected']:
        print(f"\nObject: {obj['class']}")
        print(f"Center: x={obj['center']['x']:.3f}, y={obj['center']['y']:.3f}, "
              f"dist={obj['center']['distance']:.3f}, angle={obj['center']['angle']:.3f}, "
              f"pixel_row={obj['center']['pixel_row']}")
        print("Corners:")
        for i, corner in enumerate(obj['corners']):
            print(f"  Corner {i+1}: x={corner['x']:.3f}, y={corner['y']:.3f}, "
                  f"dist={corner['distance']:.3f}, angle={corner['angle']:.3f}, "
                  f"pixel_row={corner['pixel_row']}")

def main():
    # Define log paths
    old_log = "raw_logs/yolo1d_scan_2025-05-19_15-12-56.jsonl"
    new_log = "raw_logs/yolo1d_scan_2025-06-01_19-14-36.jsonl"
    
    # Compare the outputs
    compare_outputs(old_log, new_log)

if __name__ == "__main__":
    main() 