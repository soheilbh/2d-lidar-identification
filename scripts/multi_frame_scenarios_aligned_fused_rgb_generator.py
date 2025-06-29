# Multi-frame scenarios aligned fused RGB generator
# This script processes LiDAR scan data to create advanced RGB images by aligning and fusing multiple frames
# Uses pose information to align frames before fusion, creating more accurate temporal representations
# Implements density-based fusion with edge detection and morphological operations

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
    """
    A buffer class that maintains a fixed-size queue of LiDAR scans with pose information.
    Used to create aligned and fused RGB images by considering robot movement between frames.
    Implements advanced fusion techniques including density calculation, edge detection, and morphological operations.
    """
    def __init__(self, buffer_size=5):
        """
        Initialize the LiDAR buffer with a specified buffer size.
        
        Args:
            buffer_size (int): Number of frames to keep in buffer (default: 5 for aligned fusion)
        """
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)  # FIFO queue for binary arrays
        self.poses = deque(maxlen=buffer_size)  # FIFO queue for robot poses
        # Set threshold to be more than half of buffer size for density-based fusion
        self.threshold = int(np.ceil(buffer_size / 2))
    
    def add_frame(self, raw_scan, angles, pose):
        """
        Add a new frame to buffer (FIFO).
        Converts raw LiDAR scan data to a binary array representation.
        
        Args:
            raw_scan: Raw distance measurements from LiDAR
            angles: Angular measurements corresponding to each distance reading
            pose: Robot pose information (x, y, theta) for alignment
        """
        # Convert raw data to binary array
        binary_array = np.zeros((64, 360), dtype=np.uint8)  # 64 distance bins, 360 angle bins
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)  # Convert to degrees and wrap to 0-359
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)  # Scale distances to 0-63 range
        binary_array[distance_indices, angle_indices] = 1  # Set detected points to 1
        
        self.binary_arrays.append(binary_array)  # Add to FIFO queue
        self.poses.append(pose)  # Add pose to FIFO queue
    
    def get_rgb_image(self):
        """
        Get RGB image from current buffer state for model inference.
        Creates an aligned and fused RGB image using advanced computer vision techniques.
        
        Returns:
            numpy.ndarray: RGB image with shape (64, 384, 3) or None if buffer not full
        """
        if len(self.binary_arrays) < self.buffer_size:
            return None  # Buffer not full yet
        
        # Calculate shifts based on robot orientation changes for frame alignment
        angles = [pose[2] for pose in self.poses]  # Extract robot orientation angles
        reference_angle = angles[-1]  # Use the most recent frame as reference
        # Simple shift calculation with modulo for wrapping around 360 degrees
        shifts = ((np.degrees(reference_angle - np.array(angles))) % 360).astype(np.uint16)
        # Density calculation by rolling arrays and summing (np.roll handles wrapping automatically)
        density_array = np.sum([np.roll(arr, shift, axis=1) for arr, shift in zip(self.binary_arrays, shifts)], axis=0).astype(np.uint8)
        
        # Create mask with morphological dilation to expand detected regions
        structure = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)  # Cross-shaped dilation kernel
        mask = ndimage.binary_dilation(density_array >= self.threshold, structure=structure).astype(np.uint8)
        
        # Edge detection using Sobel operators for feature enhancement
        binary = (density_array >= self.threshold).astype(np.uint8)  # Thresholded density array
        sobel_x = ndimage.sobel(binary, axis=1).astype(np.uint8)  # Horizontal edge detection
        sobel_y = ndimage.sobel(binary, axis=0).astype(np.uint8)  # Vertical edge detection
        edges = np.clip(np.abs(sobel_x) + np.abs(sobel_y), 0, 255).astype(np.uint8)  # Combine edge responses
        edges = (edges > 0).astype(np.uint8) * 255  # Threshold edges to binary
        
        # Create padded RGB image directly with three distinct channels
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)  # Create padded array directly
        rgb_image[:, :360, 0] = mask * 255  # R channel: dilated mask (expanded regions)
        rgb_image[:, :360, 1] = edges  # G channel: edge features
        rgb_image[:, :360, 2] = mask * (density_array / self.buffer_size * 255).astype(np.uint8)  # B channel: density-weighted mask
        
        return rgb_image
    
    def save_rgb_image(self, rgb_image, output_path):
        """
        Save RGB image to file (for testing purposes).
        
        Args:
            rgb_image: The RGB image to save
            output_path: Path where the image should be saved
        """
        if output_path and rgb_image is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            Image.fromarray(rgb_image).save(output_path)  # Use PIL for image saving

def main(buffer_size=5):
    """
    Main function that processes LiDAR log files and generates aligned fused RGB images.
    Reads the latest log file, processes frames in scenarios, and saves advanced RGB images.
    
    Args:
        buffer_size (int): Number of frames to use for fusion (default: 5)
    """
    # Initialize buffer for storing frames and poses
    buffer = LidarBuffer(buffer_size=buffer_size)
    
    # Find latest log file in the raw_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)  # Get most recent log file
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]  # Extract filename without extension
    
    # Create output directory for multi-frame aligned fused images
    output_dir = Path("output/multi_frame/multi_scenarios/aligned_fused") / f"X_5_frame_aligned_fused_{raw_log_name}"
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
    
    
    with open(latest_log, 'r') as f:
        pbar = tqdm(f, total=total_frames, desc="Processing frames", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {postfix} [{elapsed}<{remaining}, {rate_fmt}]')
        for line in pbar:
            data = json.loads(line)
            
            # Skip summary/logging frames and invalid data (frames without raw_scan)
            if "status" in data or "raw_scan" not in data:
                continue
                
            # Create LidarScan object from frame data
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            
            # Check if we're in a new scenario
            if current_scenario != data.get("scenario_number"):
                # Reset buffer for new scenario to avoid mixing frames from different scenarios
                buffer = LidarBuffer(buffer_size=5)
                current_scenario = data.get("scenario_number")
                scenario_frame_count = 0
                pbar.set_postfix_str(f"scenario={current_scenario}/{total_scenarios-1} , frames={frame_count}/{total_frames}")
            
            # Add current frame to buffer (includes pose information for alignment)
            buffer.add_frame(scan.raw_scan, scan.angles, scan.pose)
            
            # Process and save images after buffer is full (after 5th frame)
            if scenario_frame_count >= 4:  # buffer_size - 1 (need 5 frames to start generating)
                rgb_image = buffer.get_rgb_image()
                if rgb_image is not None:
                    # Save with frame number, scenario, and global counter in filename
                    frame_num = scenario_frame_count - 4  # buffer_size - 1 (adjust for buffer offset)
                    output_path = output_dir / f"frame_{frame_num}_{current_scenario}_{data.get('main_scenario_id', current_scenario)}_{global_frame_counter}.png"
                    buffer.save_rgb_image(rgb_image, str(output_path))
                    global_frame_counter += 1
            
            frame_count += 1
            scenario_frame_count += 1

if __name__ == "__main__":
    main(buffer_size=5)  # Can be changed for different use cases - execute main function when script is run directly 