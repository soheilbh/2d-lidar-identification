import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan
import cv2
from tqdm import tqdm
from collections import deque

class SimpleRGBBuffer:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)
    
    def add_frame(self, raw_scan, angles):
        """Add a new frame to buffer (FIFO)"""
        # Convert raw data to binary array
        binary_array = np.zeros((64, 360), dtype=np.uint8)
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)
        binary_array[distance_indices, angle_indices] = 255  # Use 255 for better visibility
        
        self.binary_arrays.append(binary_array)
    
    def get_rgb_image(self):
        """Get RGB image from current buffer state"""
        if len(self.binary_arrays) < self.buffer_size:
            return None
        
        # Create RGB image
        rgb_image = np.zeros((64, 360, 3), dtype=np.uint8)
        rgb_image[:, :, 0] = self.binary_arrays[0]  # R channel
        rgb_image[:, :, 1] = self.binary_arrays[1]  # G channel
        rgb_image[:, :, 2] = self.binary_arrays[2]  # B channel
        
        return rgb_image
    
    def save_rgb_image(self, rgb_image, output_path):
        """Save RGB image to file"""
        if output_path and rgb_image is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, rgb_image)

def create_simple_rgb_image(frame1, frame2, frame3):
    """
    Create RGB image by putting each frame in a different channel.
    frame1 -> R channel
    frame2 -> G channel
    frame3 -> B channel
    """
    # Create empty RGB image
    rgb_image = np.zeros((64, 360, 3), dtype=np.uint8)
    
    # Put each frame in its respective channel
    rgb_image[:, :, 0] = frame1  # R channel
    rgb_image[:, :, 1] = frame2  # G channel
    rgb_image[:, :, 2] = frame3  # B channel
    
    return rgb_image

def create_visualization(frame1, frame2, frame3, fused_image, output_path):
    """
    Create a visualization showing individual frames and the fused result.
    """
    # Create a larger image to hold all visualizations
    vis_height = 64 * 2  # 2 rows
    vis_width = 360 * 2  # 2 columns
    visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # Convert single channel images to RGB for visualization
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_GRAY2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_GRAY2RGB)
    frame3_rgb = cv2.cvtColor(frame3, cv2.COLOR_GRAY2RGB)
    
    # Place images in the visualization
    # Top row: individual frames
    visualization[0:64, 0:360] = frame1_rgb
    visualization[0:64, 360:720] = frame2_rgb
    # Bottom row: frame3 and fused result
    visualization[64:128, 0:360] = frame3_rgb
    visualization[64:128, 360:720] = fused_image
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Frame 1 (R)", (10, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(visualization, "Frame 2 (G)", (370, 20), font, 0.5, (255, 255, 255), 1)
    cv2.putText(visualization, "Frame 3 (B)", (10, 84), font, 0.5, (255, 255, 255), 1)
    cv2.putText(visualization, "Fused RGB", (370, 84), font, 0.5, (255, 255, 255), 1)
    
    # Save visualization
    cv2.imwrite(output_path, visualization)
    
    # Display visualization
    cv2.imshow('RGB Fusion Visualization', visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return visualization

def main():
    # Find latest log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Create output directory
    output_dir = Path("output/multi_frame/simple_rgb") / f"X_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress bar
    with open(latest_log, 'r') as f:
        total_frames = sum(1 for _ in f)
    
    # Initialize buffer
    buffer = SimpleRGBBuffer(buffer_size=3)
    
    # Process frames
    frame_count = 0
    with open(latest_log, 'r') as f:
        for line in tqdm(f, total=total_frames, desc="Processing frames"):
            data = json.loads(line)
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            buffer.add_frame(scan.raw_scan, scan.angles)
            
            # Process and save images after buffer is full
            if frame_count >= 2:  # buffer_size - 1
                rgb_image = buffer.get_rgb_image()
                if rgb_image is not None:
                    # Save with frame number in filename (frame 0-2 -> frame_0.png, frame 1-3 -> frame_1.png, etc.)
                    frame_num = frame_count - 2  # buffer_size - 1
                    output_path = output_dir / f"frame_{frame_num}.png"
                    buffer.save_rgb_image(rgb_image, str(output_path))
            
            frame_count += 1
    
    print(f"All frames processed and saved to {output_dir}")

if __name__ == "__main__":
    main() 