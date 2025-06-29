# Multi-frame scenarios simple fused RGB generator
# This script processes LiDAR scan data to create RGB images by combining multiple frames
# Each frame is assigned to a different color channel (R, G, B) to create temporal information

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
    """
    A buffer class that maintains a fixed-size queue of binary arrays from LiDAR scans.
    Used to create RGB images by assigning different frames to different color channels.
    """
    def __init__(self, buffer_size=3):
        """
        Initialize the RGB buffer with a specified buffer size.
        
        Args:
            buffer_size (int): Number of frames to keep in buffer (default: 3 for RGB)
        """
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)  # FIFO queue with max length
    
    def add_frame(self, raw_scan, angles):
        """
        Add a new frame to buffer (FIFO).
        Converts raw LiDAR scan data to a binary array representation.
        
        Args:
            raw_scan: Raw distance measurements from LiDAR
            angles: Angular measurements corresponding to each distance reading
        """
        # Convert raw data to binary array
        binary_array = np.zeros((64, 360), dtype=np.uint8)  # 64 distance bins, 360 angle bins
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)  # Convert to degrees and wrap to 0-359
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)  # Scale distances to 0-63 range
        binary_array[distance_indices, angle_indices] = 255  # Use 255 for better visibility
        
        self.binary_arrays.append(binary_array)  # Add to FIFO queue
    
    def get_rgb_image(self):
        """
        Get RGB image from current buffer state.
        Creates a 3-channel image where each channel represents a different frame.
        
        Returns:
            numpy.ndarray: RGB image with shape (64, 384, 3) or None if buffer not full
        """
        if len(self.binary_arrays) < self.buffer_size:
            return None  # Buffer not full yet
        
        # Create RGB image with padding (384 width to accommodate 360 + padding)
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)  # Create padded array directly
        rgb_image[:, :360, 0] = self.binary_arrays[0]  # R channel - oldest frame
        rgb_image[:, :360, 1] = self.binary_arrays[1]  # G channel - middle frame
        rgb_image[:, :360, 2] = self.binary_arrays[2]  # B channel - newest frame
        
        return rgb_image
    
    def save_rgb_image(self, rgb_image, output_path):
        """
        Save RGB image to file.
        
        Args:
            rgb_image: The RGB image to save
            output_path: Path where the image should be saved
        """
        if output_path and rgb_image is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            cv2.imwrite(output_path, rgb_image)

def create_simple_rgb_image(frame1, frame2, frame3):
    """
    Create RGB image by putting each frame in a different channel.
    This function is an alternative way to create RGB images from three frames.
    
    Args:
        frame1: First frame (assigned to R channel)
        frame2: Second frame (assigned to G channel)
        frame3: Third frame (assigned to B channel)
    
    Returns:
        numpy.ndarray: RGB image with shape (64, 384, 3)
    """
    # Create empty RGB image with padding
    rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)
    
    # Put each frame in its respective channel
    rgb_image[:, :360, 0] = frame1  # R channel
    rgb_image[:, :360, 1] = frame2  # G channel
    rgb_image[:, :360, 2] = frame3  # B channel
    
    return rgb_image

def main():
    """
    Main function that processes LiDAR log files and generates RGB images.
    Reads the latest log file, processes frames in scenarios, and saves RGB images.
    """
    # Find latest log file in the raw_logs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)  # Get most recent log file
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]  # Extract filename without extension
    
    # Create output directory for generated RGB images
    output_dir = Path("output/multi_frame/multi_scenarios/simple_fused") / f"X_3_frame_simple_fused_{raw_log_name}"
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
    
    # Initialize buffer for storing frames
    buffer = SimpleRGBBuffer(buffer_size=3)
    
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
                buffer = SimpleRGBBuffer(buffer_size=3)
                current_scenario = data.get("scenario_number")
                scenario_frame_count = 0
                pbar.set_postfix_str(f"scenario={current_scenario}/{total_scenarios-1} , frames={frame_count}/{total_frames}")
            
            # Add current frame to buffer
            buffer.add_frame(scan.raw_scan, scan.angles)
            
            # Process and save images after buffer is full (after 3rd frame)
            if scenario_frame_count >= 2:  # buffer_size - 1 (need 3 frames to start generating)
                rgb_image = buffer.get_rgb_image()
                if rgb_image is not None:
                    # Save with frame number, scenario, and global counter in filename
                    frame_num = scenario_frame_count - 2  # buffer_size - 1 (adjust for buffer offset)
                    output_path = output_dir / f"frame_{frame_num}_{current_scenario}_{data.get('main_scenario_id', current_scenario)}_{global_frame_counter}.png"
                    buffer.save_rgb_image(rgb_image, str(output_path))
                    global_frame_counter += 1
            
            frame_count += 1
            scenario_frame_count += 1
    
    print(f"All frames processed and saved to {output_dir}")

if __name__ == "__main__":
    main()  # Execute main function when script is run directly 