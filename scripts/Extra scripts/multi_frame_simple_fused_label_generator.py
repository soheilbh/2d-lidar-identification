import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from post_process import process_lidar_data
from bot3.utils.lidar_utils import LidarScan, KNOWN_OBJECTS
import json
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path

# Variables to extend bounding box width and height (in pixels)
WIDTH_EXTENSION = 22  # e.g., set to 10 to add 10 pixels to width
HEIGHT_EXTENSION = 12  # e.g., set to 4 to add 4 pixels to height

def get_class_id(obj_name):
    """Get class ID based on object name"""
    class_mapping = {
        "chair": 0,
        "box": 1,
        "desk": 2,
        "door_frame": 3
    }
    return class_mapping.get(obj_name, -1)

def scale_distance_to_pixel(distance, max_range=4.0):
    """Scale distance (0-4m) to pixel row (0-63)"""
    return int((distance / max_range) * 63)

def calculate_corner_distance_angle(x, y):
    """
    Calculate distance and angle for a point (x,y) relative to robot (0,0)
    using the same logic as process_lidar_data.
    """
    # Calculate distance
    distance = np.sqrt(x**2 + y**2)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(y, x))
    
    # Convert to clockwise (CW)
    if angle < 0:
        angle += 360
    angle = (360 - angle) % 360
    
    # Round angle to nearest integer
    angle = round(angle)
    
    return distance, angle

def calculate_object_corners(x_center, y_center, width, depth, robot_theta=0):
    """
    Calculate the four corners of an object rectangle.
    
    Args:
        x_center: x-coordinate of object center
        y_center: y-coordinate of object center
        width: width of object
        depth: depth of object
        robot_theta: robot's orientation in radians (default 0)
    
    Returns:
        List of dictionaries containing corner coordinates, distances, angles, and pixel rows
    """
    # Calculate unrotated corners
    corners = [
        (x_center - width/2, y_center - depth/2),  # bottom-left
        (x_center + width/2, y_center - depth/2),  # bottom-right
        (x_center + width/2, y_center + depth/2),  # top-right
        (x_center - width/2, y_center + depth/2)   # top-left
    ]
    
    # Rotate corners if robot_theta is not 0
    if robot_theta != 0:
        corners = [rotate_point(x, y, -robot_theta, x_center, y_center) for x, y in corners]
    
    # Calculate distance and angle for each corner
    corner_info = []
    for i, (x, y) in enumerate(corners):
        distance, angle = calculate_corner_distance_angle(x, y)
        pixel_row = scale_distance_to_pixel(distance)  # Scale distance to pixel row (0-63)
        corner_info.append({
            'corner': i,
            'x': x,
            'y': y,
            'distance': distance,
            'angle': angle,
            'pixel_row': pixel_row
        })
    
    return corner_info

def rotate_point(x, y, theta, cx=0, cy=0):
    """
    Rotate point (x, y) by theta radians around center (cx, cy).
    Positive theta is CCW.
    """
    x_shifted = x - cx
    y_shifted = y - cy
    xr = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted + cx
    yr = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted + cy
    return xr, yr

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

def adjust_center_for_close_objects(corners, center):
    """Adjust center position for desk and chair objects when they are too close to robot"""
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

if __name__ == "__main__":
    # Find latest log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Create output directory for labels
    output_dir = Path("output/multi_frame/simple_labels") / f"Y_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all frames
    with open(latest_log, 'r') as f:
        lines = f.readlines()

    # Start from frame 1 (index 1) since that's our reference frame in the 3-frame buffer
    # and continue until the last frame that can be used in a complete RGB fusion
    # We need to stop 2 frames before the end since each RGB fusion needs 3 frames
    for idx in tqdm(range(1, len(lines) - 1), desc="Generating labels"):
        data = json.loads(lines[idx])
        scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id", idx))
        angles = scan.angles
        distances = scan.raw_scan
        pose = scan.pose
        robot_theta = pose[2]
        detected_objects = process_lidar_data(angles, distances, pose)

        # Prepare label file name (starting from 0 to match RGB images)
        # Since we start from frame 1, we subtract 1 from the index
        label_file = f"frame_{idx-1}.txt"
        output_file = os.path.join(output_dir, label_file)

        with open(output_file, 'w') as f_out:
            for obj in detected_objects:
                class_id = get_class_id(obj['class'])
                pixel_row = scale_distance_to_pixel(obj['distance'])
                corners = calculate_object_corners(obj['x_center'], obj['y_center'], obj['width'], obj['depth'], robot_theta)
                
                # Apply center adjustment for desk and chair if needed
                if obj['class'] in ['desk', 'chair']:
                    center = {'angle': obj['center_angle'], 'pixel_row': pixel_row}
                    adjusted_center = adjust_center_for_close_objects(corners, center)
                    pixel_row = adjusted_center['pixel_row']
                
                angles_c = [corner['angle'] for corner in corners]
                pixel_rows = [corner['pixel_row'] for corner in corners]
                # Use minimal arc for width
                width = minimal_arc_width(angles_c) + WIDTH_EXTENSION
                # Clamp box to [0, 359]
                center_angle_clamped, width_clamped, _, _ = clamp_yolo_box(obj['center_angle'], width, 0, 359)
                min_row = min(pixel_rows)
                max_row = max(pixel_rows)
                row_diff = abs(max_row - min_row) + HEIGHT_EXTENSION
                # Normalize for YOLO format
                center_x_norm = round(center_angle_clamped / 359, 6)
                center_y_norm = round(pixel_row / 63, 6)
                width_norm = round(width_clamped / 359, 6)
                height_norm = round(row_diff / 63, 6)
                f_out.write(f"{class_id} {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n")

    print(f"Labels saved to {output_dir}") 