import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan
from scipy import ndimage
from PIL import Image
from collections import deque
import time
from tqdm import tqdm

class LidarBuffer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)
        self.poses = deque(maxlen=buffer_size)
        # Set threshold to be more than half of buffer size
        self.threshold = int(np.ceil(buffer_size / 2))
    
    def add_frame(self, raw_scan, angles, pose):
        """Add a new frame to buffer (FIFO)"""
        # Convert raw data to binary array
        binary_array = np.zeros((64, 360), dtype=np.uint8)
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)
        binary_array[distance_indices, angle_indices] = 1
        
        self.binary_arrays.append(binary_array)
        self.poses.append(pose)
    
    def get_rgb_image(self):
        """Get RGB image from current buffer state for model inference"""
        if len(self.binary_arrays) < self.buffer_size:
            return None
        
        # Calculate shifts and create density array
        angles = [pose[2] for pose in self.poses]
        reference_angle = angles[-1]
        # Simple shift calculation with modulo for wrapping
        shifts = ((np.degrees(reference_angle - np.array(angles))) % 360).astype(np.uint16)
        # Density calculation (np.roll already handles wrapping)
        density_array = np.sum([np.roll(arr, shift, axis=1) for arr, shift in zip(self.binary_arrays, shifts)], axis=0).astype(np.uint8)
        
        # Create mask with dilation
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        mask = ndimage.binary_dilation(density_array >= self.threshold, structure=structure).astype(np.uint8)
        
        # Edge detection using Sobel
        binary = (density_array >= self.threshold).astype(np.uint8)
        sobel_x = ndimage.sobel(binary, axis=1).astype(np.uint8)
        sobel_y = ndimage.sobel(binary, axis=0).astype(np.uint8)
        edges = np.clip(np.abs(sobel_x) + np.abs(sobel_y), 0, 255).astype(np.uint8)
        edges = (edges > 0).astype(np.uint8) * 255
        
        # Create padded RGB image directly
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)  # Create padded array directly
        rgb_image[:, :360, 0] = mask * 255  # R channel
        rgb_image[:, :360, 1] = edges  # G channel (edges)
        rgb_image[:, :360, 2] = mask * (density_array / self.buffer_size * 255).astype(np.uint8)  # B channel
        
        return rgb_image
    
    def save_rgb_image(self, rgb_image, output_path):
        """Save RGB image to file (for testing purposes)"""
        if output_path and rgb_image is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb_image).save(output_path)

def main(buffer_size=5):
    # Initialize buffer
    buffer = LidarBuffer(buffer_size=buffer_size)
    
    # Find latest log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    
    # Create output directory for multi-frame images
    output_dir = Path("output/multi_frame/range_images") / f"X_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count total lines for progress bar
    with open(latest_log, 'r') as f:
        total_frames = sum(1 for _ in f)
    
    # Load and process frames
    frame_count = 0
    with open(latest_log, 'r') as f:
        for line in tqdm(f, total=total_frames, desc="Processing frames"):
            data = json.loads(line)
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            buffer.add_frame(scan.raw_scan, scan.angles, scan.pose)
            
            # Process and save images after buffer is full
            if frame_count >= buffer_size - 1:
                rgb_image = buffer.get_rgb_image()
                if rgb_image is not None:
                    # Save with frame number in filename (frame 0-4 -> frame_0.png, frame 1-5 -> frame_1.png, etc.)
                    frame_num = frame_count - (buffer_size - 1)
                    output_path = output_dir / f"frame_{frame_num}.png"
                    buffer.save_rgb_image(rgb_image, str(output_path))
            
            frame_count += 1

if __name__ == "__main__":
    main(buffer_size=5)  # Can be changed for different use cases 