import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan

def plot_frame_discontinuities(frame_id, raw_data_file, discontinuities_file):
    # Read raw data
    with open(raw_data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data.get("frame_id") == frame_id:
                scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
                break
    
    # Read discontinuities
    with open(discontinuities_file, 'r') as f:
        for line in f:
            frame_data = json.loads(line)
            if frame_data["frame_id"] == frame_id:
                discontinuities = frame_data["discontinuities"]
                break
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Raw distances
    angles = np.degrees(scan.angles)
    distances = scan.raw_scan
    ax1.plot(angles, distances, 'b-', label='Raw distances')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title(f'Frame {frame_id}: Raw LiDAR Data')
    ax1.grid(True)
    
    # Plot 2: Discontinuities as vertical lines
    angles_short = angles[:-1]  # One less point due to diff
    # Find indices where discontinuities occur
    disc_indices = np.where(np.array(discontinuities) == 1)[0]
    disc_angles = angles_short[disc_indices]
    
    # Plot vertical lines at discontinuity points
    for angle in disc_angles:
        ax2.vlines(angle, 0, 1, colors='r', linewidth=2)
    
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Discontinuity')
    ax2.set_title(f'Frame {frame_id}: Detected Discontinuities')
    ax2.grid(True)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"discontinuities_frame_{frame_id}.png")
    print(f"Plot saved to: {output_dir}/discontinuities_frame_{frame_id}.png")

if __name__ == "__main__":
    # Get the latest files
    raw_logs_dir = Path("raw_logs")
    raw_logs = list(raw_logs_dir.glob("yolo1d_scan_*.jsonl"))
    latest_raw = max(raw_logs, key=lambda x: x.stat().st_mtime)
    
    post_process_dir = Path("output/post_process_data")
    discontinuity_files = list(post_process_dir.glob("discontinuities_*.jsonl"))
    latest_discontinuities = max(discontinuity_files, key=lambda x: x.stat().st_mtime)
    
    # Plot frame 0
    plot_frame_discontinuities(0, latest_raw, latest_discontinuities) 