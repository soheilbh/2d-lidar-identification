import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan
import cv2
import matplotlib.pyplot as plt
from collections import deque
from post_process import process_lidar_data_new

# Variables to extend bounding box width and height (in pixels)
WIDTH_EXTENSION = 25
HEIGHT_EXTENSION = 15

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

class SingleFusedRGBWithLabels:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)
        self.frame_data = deque(maxlen=buffer_size)
    
    def add_frame(self, raw_scan, angles, frame_data):
        """Add a new frame to buffer (FIFO)"""
        # Convert raw data to binary array
        binary_array = np.zeros((64, 360), dtype=np.uint8)
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)
        binary_array[distance_indices, angle_indices] = 255
        
        self.binary_arrays.append(binary_array)
        self.frame_data.append(frame_data)
    
    def get_rgb_image(self):
        """Get RGB image from current buffer state"""
        if len(self.binary_arrays) < self.buffer_size:
            return None
        
        # Create RGB image with padding
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)
        rgb_image[:, :360, 0] = self.binary_arrays[0]  # R channel (t-2)
        rgb_image[:, :360, 1] = self.binary_arrays[1]  # G channel (t-1)
        rgb_image[:, :360, 2] = self.binary_arrays[2]  # B channel (t)
        
        return rgb_image
    
    def generate_labels(self):
        """Generate labels for the middle frame using the same logic as multi_frame_scenarios_simple_fused_label_generator.py"""
        if len(self.frame_data) < self.buffer_size:
            return []
        
        # Use middle frame (index 1) for labeling
        middle_frame = self.frame_data[1]
        scan = LidarScan(middle_frame["raw_scan"], middle_frame["pose"], 
                       frame_id=middle_frame.get("frame_id"))
        angles = scan.angles
        distances = scan.raw_scan
        pose = scan.pose
        robot_theta = pose[2]
        object_details = middle_frame.get('object_details', None)
        detected_objects = process_lidar_data_new(angles, distances, pose, object_details)
        
        labels = []
        for obj in detected_objects:
            # Skip objects that are too far (pixel_row >= 56)
            if obj['center']['pixel_row'] >= 56:
                continue
                
            class_id = get_class_id(obj['class'])
            # Use pixel rows directly from the new format
            angles_c = [corner['angle'] for corner in obj['corners']]
            pixel_rows = [corner['pixel_row'] for corner in obj['corners']]
            
            # Apply center adjustment for desk and chair if needed
            if obj['class'] in ['desk', 'chair']:
                obj['center'] = adjust_center_for_close_objects(obj['corners'], obj['center'])
            
            # Use minimal arc for width
            width = minimal_arc_width(angles_c) + WIDTH_EXTENSION
            # Clamp box to [0, 359]
            center_angle_clamped, width_clamped, _, _ = clamp_yolo_box(obj['center']['angle'], width, 0, 359)
            min_row = min(pixel_rows)
            max_row = max(pixel_rows)
            row_diff = abs(max_row - min_row) + HEIGHT_EXTENSION
            
            # Store label information
            labels.append({
                'class_id': class_id,
                'center_angle': center_angle_clamped,
                'center_row': obj['center']['pixel_row'],
                'width': width_clamped,
                'height': row_diff,
                'class_name': obj['class']
            })
        
        return labels
    
    def draw_labels_on_image(self, rgb_image, labels):
        """Draw bounding boxes and labels on the RGB image, with label at the bottom of the box"""
        # Create a copy of the image for drawing
        image_with_labels = rgb_image.copy()
        
        # Define colors for different classes (BGR format for OpenCV)
        colors = {
            0: (0, 255, 0),    # Green for chair
            1: (0, 0, 255),    # Red for box
            2: (0, 255, 255),  # Yellow for desk
            3: (255, 255, 0)   # Cyan for door frame
        }
        
        class_names = {0: "Chair", 1: "Box", 2: "Desk", 3: "Door"}
        
        for label in labels:
            class_id = label['class_id']
            center_angle = int(label['center_angle'])
            center_row = int(label['center_row'])
            width = int(label['width'])
            height = int(label['height'])
            
            # Calculate bounding box corners
            x1 = center_angle - width // 2
            x2 = center_angle + width // 2
            y1 = center_row - height // 2
            y2 = center_row + height // 2
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, 359))
            x2 = max(0, min(x2, 359))
            y1 = max(0, min(y1, 63))
            y2 = max(0, min(y2, 63))
            
            # Draw rectangle
            color = colors.get(class_id, (255, 255, 255))  # White for unknown classes
            cv2.rectangle(image_with_labels, (x1, y1), (x2, y2), color, 1)
            
            # Add class label at the bottom of the box
            label_text = class_names.get(class_id, f"Class {class_id}")
            text_y = y2 + 12 if y2 + 12 < 64 else 63  # Place below box, but stay in bounds
            cv2.putText(image_with_labels, label_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw center point
            cv2.circle(image_with_labels, (center_angle, center_row), 3, (255, 255, 255), -1)
        
        return image_with_labels
    
    def create_visualization(self, output_path):
        """Create a single matplotlib plot of the fused RGB image with labels and axes"""
        if len(self.binary_arrays) < self.buffer_size:
            print("Buffer not full yet. Need 3 frames.")
            return
        
        # Get the RGB image
        rgb_image = self.get_rgb_image()
        
        # Generate labels
        labels = self.generate_labels()
        
        # Draw labels on image
        image_with_labels = self.draw_labels_on_image(rgb_image, labels)
        
        # Plot with matplotlib (with axes and title)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(image_with_labels, aspect='equal', interpolation='nearest')
        # ax.set_title('2D LiDAR Simple Fused RGB with Labels', fontsize=16, fontweight='bold')
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Distance (bins)', fontsize=12)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 359, 383])
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_xlim(0, 383)
        ax.set_ylim(63, 0)
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Labeled RGB matplotlib plot saved to: {output_path}")
        print(f"Number of detected objects: {len(labels)}")

def main():
    # Find latest log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Create output directory
    output_dir = Path("output/visualizations") / f"single_fused_rgb_with_labels_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize buffer
    buffer = SingleFusedRGBWithLabels(buffer_size=3)
    
    print("Finding first 3 frames for RGB fusion...")
    
    # Get the first 3 frames
    frame_count = 0
    with open(latest_log, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Skip summary/logging frames and invalid data
            if "status" in data or "raw_scan" not in data:
                continue
                
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            buffer.add_frame(scan.raw_scan, scan.angles, data)
            frame_count += 1
            
            if frame_count >= 3:
                break
    
    # Create visualization
    output_path = output_dir / f"single_fused_rgb_with_labels.png"
    buffer.create_visualization(str(output_path))
    
    print(f"\nSingle fused RGB with labels completed!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 