import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import os
import sys
from pathlib import Path
import argparse
import imageio
import concurrent.futures

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from bot3.utils.lidar_utils import LidarScan

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print('tqdm is not installed. Progress bars will not be shown. Install with: pip install tqdm')

def find_latest_log_file(log_dir):
    """Find the latest log file in the specified directory."""
    log_files = list(Path(log_dir).glob("yolo1d_scan_*.jsonl"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")
    return max(log_files, key=lambda x: x.stat().st_mtime)

def find_processed_data_file(log_name):
    """Find the corresponding processed data file."""
    # Try to find the all_frames version first
    all_frames_path = os.path.join('output', 'post_process_data', f"processed_{log_name}_all_frames.json")
    if os.path.exists(all_frames_path):
        return all_frames_path
    
    # If not found, try the single frame version
    single_frame_path = os.path.join('output', 'post_process_data', f"processed_{log_name}.json")
    if os.path.exists(single_frame_path):
        return single_frame_path
    
    raise FileNotFoundError(f"No processed data found for {log_name}")

def plot_all_views(raw_data, processed_data, output_path, figsize=(20, 15)):
    """
    Plot all views of the data.
    
    Args:
        raw_data (dict): Dictionary containing raw LiDAR data
        processed_data (dict): Dictionary containing processed data
        output_path (str): Path to save the output plot
        figsize (tuple): Figure size for the plot
    """
    # Create figure with GridSpec for better subplot arrangement
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
    
    # Add frame number as super-title
    frame_id = raw_data.get('frame_id', None)
    if frame_id is not None:
        fig.suptitle(f'Frame: {frame_id}', fontsize=18, fontweight='bold')

    # Create LidarScan object
    lidar_scan = LidarScan(raw_data['raw_scan'], raw_data['pose'])
    
    # For both ax2 and ax3
    dot_color = '#003566'  # high-contrast color
    dot_alpha = 0.7     # higher opacity
    dot_size = 12       # slightly larger

    # 1. Angle vs Distance (from raw data) - Top row, full width
    ax1 = fig.add_subplot(gs[0, :])
    angles_deg = np.degrees(lidar_scan.angles)  # Get angles from LidarScan
    markerline, stemlines, baseline = ax1.stem(
        angles_deg, raw_data['raw_scan'], linefmt='gray', markerfmt='o', basefmt=' ')
    plt.setp(markerline, markersize=3, color=dot_color, alpha=dot_alpha)
    plt.setp(stemlines, linewidth=1, color='gray', alpha=0.5)
    ax1.grid(True, which='major', linestyle='-', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.minorticks_on()
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Angle vs Distance\n(LiDAR beam measurements)", pad=10)
    ax1.set_xlim(-5, 365)
    # Replace Infinity values with NaN before calculating max
    raw_scan_clean = np.array(raw_data['raw_scan'])
    raw_scan_clean[raw_scan_clean == float('inf')] = np.nan
    ymax = np.nanmax(raw_scan_clean) * 1.1
    ax1.set_ylim(0, ymax)
    ax1.set_xticks(np.arange(0, 361, 45))
    ax1.set_xticks(np.arange(0, 361, 15), minor=True)
    # Highlight points where value is exactly 4 (plot after stem, with high zorder)
    idx_4 = np.isclose(raw_scan_clean, 4.0)
    ax1.scatter(angles_deg[idx_4], raw_scan_clean[idx_4], color='red', s=20, label='No return (4m)', zorder=10)

    # 2. Top View - Bottom row, left
    ax2 = fig.add_subplot(gs[1, 0])
    # Convert polar to cartesian using LidarScan (convert CW to CCW)
    angles_ccw = (2 * np.pi - lidar_scan.angles) % (2 * np.pi)
    x = lidar_scan.raw_scan * np.cos(angles_ccw)
    y = lidar_scan.raw_scan * np.sin(angles_ccw)
    ax2.scatter(x, y, s=dot_size, c=dot_color, alpha=dot_alpha, label='LiDAR points')
    # Highlight points where value is exactly 4
    ax2.scatter(x[idx_4], y[idx_4], s=dot_size+6, c='red', alpha=0.9, label='No return (4m)')
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (LiDAR points)', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right')

    # Define a nice color palette for different objects
    color_map = {
        'chair': '#219EBC',  # Coral Red
        'box': '#FFB703',    # Turquoise
        'desk': '#FB8500',   # Orange
        'door_frame': '#6a994e'  # Sage Green
    }

    robot_theta = raw_data['pose'][2] if 'pose' in raw_data else 0
    for obj in processed_data:
        center_x = obj.get("x_center", 0)
        center_y = obj.get("y_center", 0)
        width = obj.get("width", 0)
        depth = obj.get("depth", 0)
        name = obj.get("class", "Unknown")
        distance = obj.get("distance", 0)
        rotation = obj.get("rotation", 0) if "rotation" in obj else 0

        # Get color for this object type
        color = color_map.get(name, '#808080')  # Default to gray if unknown type

        # Draw filled rectangle (with low alpha) for the object, rotated by -robot_theta
        # Rectangle corners (unrotated)
        corners = [
            (center_x - width/2, center_y - depth/2),
            (center_x + width/2, center_y - depth/2),
            (center_x + width/2, center_y + depth/2),
            (center_x - width/2, center_y + depth/2)
        ]
        corners_rot = [rotate_point(x, y, -robot_theta, center_x, center_y) for x, y in corners]
        rect_patch = plt.Polygon(corners_rot, closed=True, color=color, alpha=0.08, zorder=2)
        ax2.add_patch(rect_patch)

        # Draw cross (width and depth lines), rotated by -robot_theta
        # Width line
        x1, y1 = rotate_point(center_x - width/2, center_y, -robot_theta, center_x, center_y)
        x2, y2 = rotate_point(center_x + width/2, center_y, -robot_theta, center_x, center_y)
        ax2.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.8, zorder=3)
        # Depth line
        x3, y3 = rotate_point(center_x, center_y - depth/2, -robot_theta, center_x, center_y)
        x4, y4 = rotate_point(center_x, center_y + depth/2, -robot_theta, center_x, center_y)
        ax2.plot([x3, x4], [y3, y4], color=color, linewidth=3, alpha=0.8, zorder=3)

        # Plot center dot
        ax2.scatter(center_x, center_y, s=80, c=color, marker='o', label=f'{name} center', edgecolor='black', linewidth=2)

        # Write name and coordinates next to center
        ax2.text(center_x + 0.1, center_y + 0.1, 
                f'{name}\n({center_x:.2f}, {center_y:.2f})', 
                fontsize=10, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

        # Draw line from robot to center and write distance
        ax2.plot([0, center_x], [0, center_y], color=color, linestyle='--', linewidth=2, alpha=0.6)
        mid_x, mid_y = center_x/2, center_y/2
        ax2.text(mid_x, mid_y, f'{distance:.2f}m', 
                fontsize=10, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3),
                color=color)

    # Add robot position marker
    ax2.scatter(0, 0, s=150, c='red', marker='o', label='Robot', zorder=5)

    # 3. Polar View - Bottom row, right
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')
    ax3.scatter(lidar_scan.angles, raw_scan_clean, s=dot_size, c=dot_color, alpha=dot_alpha)
    # Highlight points where value is exactly 4
    ax3.scatter(lidar_scan.angles[idx_4], raw_scan_clean[idx_4], s=dot_size+6, c='red', alpha=0.9, label='No return (4m)')
    
    # Add robot position marker in polar view
    ax3.scatter(0, 0, s=150, c='red', marker='o', label='Robot', zorder=5)
    
    # Customize the polar plot
    ax3.set_theta_zero_location('E')  # Set 0 degrees to North (12 o'clock)
    ax3.set_theta_direction(-1)  # Set clockwise direction
    ax3.set_title('Polar View\n(LiDAR measurements)', pad=10)
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # Add distance labels
    ax3.set_rlabel_position(45)  # Move radial labels to 45 degrees
    
    # Plot objects in polar view
    for obj in processed_data:
        center_x = obj.get("x_center", 0)
        center_y = obj.get("y_center", 0)
        name = obj.get("class", "Unknown")
        distance = obj.get("distance", 0)
        center_angle = obj.get("center_angle", 0)
        
        # Get color for this object type
        color = color_map.get(name, '#808080')
        
        # Plot object center in polar coordinates
        ax3.scatter(center_angle * np.pi/180, distance, s=50, c=color, marker='o', 
                   edgecolor='black', linewidth=2, label=f'{name} center')
        
        # Add object name and distance
        ax3.text(center_angle * np.pi/180, distance + 0.1, 
                f'{name}\n{distance:.2f}m', 
                fontsize=10, ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_frame_for_batch(raw_log_path, frame, batch_dir):
    frame_id = frame['frame_id']
    processed_data = frame['objects']
    # Find raw data for this frame
    raw_data = None
    with open(raw_log_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get('frame_id') == frame_id:
                raw_data = data
                break
    if raw_data is None:
        print(f"Warning: Frame ID {frame_id} not found in raw log, skipping.")
        return None
    output_path = os.path.join(batch_dir, f"frame_{frame_id}.png")
    plot_all_views(raw_data, processed_data, output_path, figsize=(20, 15))
    return output_path

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

def main():
    parser = argparse.ArgumentParser(description='Visualize LiDAR data frames (single or batch).')
    args = parser.parse_args()

    # Find the latest log file
    log_dir = 'raw_logs'
    try:
        raw_log = find_latest_log_file(log_dir)
        print(f"Using latest log file: {raw_log}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have log files in the 'raw_logs' directory")
        return

    # Create descriptive output filename
    raw_log_name = os.path.splitext(os.path.basename(raw_log))[0]
    output_dir = 'output/post_process_plot'
    single_frame_dir = os.path.join(output_dir, 'single_frames')
    batch_dir = os.path.join(output_dir, f"batch_{raw_log_name}")
    video_dir = 'output/video'

    # Load processed data
    try:
        processed_data_path = find_processed_data_file(raw_log_name)
        print(f"Using processed data file: {processed_data_path}")
        with open(processed_data_path, 'r') as f:
            processed_data_all = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run post-processing on the log file first")
        return

    # Ask user if they want to visualize all frames as a batch
    all_frames_input = input('Visualize all frames as a batch? (Y/N): ').strip().lower()
    if all_frames_input == 'y':
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        image_paths = []
        # Ask user how many frames to skip
        skip = int(input('How many frames do you want to skip between each processed frame? (e.g., 0 = use every frame, 9 = use every 10th frame): ').strip())
        print(f"Generating visualizations for all {len(processed_data_all)} frames (parallel), skipping {skip} frame(s) between each...")
        # Only process every (skip+1)th frame
        selected_frames = [frame for idx, frame in enumerate(processed_data_all) if idx % (skip+1) == 0]
        frame_iter = tqdm(selected_frames, desc='Frames') if tqdm else selected_frames
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(plot_frame_for_batch, raw_log, frame, batch_dir) for frame in frame_iter]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Saving', disable=(tqdm is None)):
                result = f.result()
                if result:
                    image_paths.append(result)
        # Create MP4 in video_dir
        print('Creating MP4...')
        mp4_path = os.path.join(video_dir, f'{raw_log_name}.mp4')
        image_iter = tqdm(image_paths, desc='MP4 frames') if tqdm else image_paths
        with imageio.get_writer(mp4_path, fps=32, format='ffmpeg') as writer:
            for p in sorted(image_iter, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])):
                image = imageio.imread(p)
                writer.append_data(image)
        print(f"Batch visualizations saved in {batch_dir}")
        print(f"MP4 animation saved as {mp4_path}")
        return
    else:
        # Single frame mode
        os.makedirs(single_frame_dir, exist_ok=True)
        frame_id = input('Enter frame ID to visualize: ').strip()
        frame_id = int(frame_id)
        # Find the frame with the requested frame_id
        processed_data = None
        for frame in processed_data_all:
            if frame['frame_id'] == frame_id:
                processed_data = frame['objects']
                break
        if processed_data is None:
            print(f"Error: Frame ID {frame_id} not found in processed data")
            return
        # Load raw log and find the frame with the requested frame_id
        try:
            with open(raw_log, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('frame_id') == frame_id:
                        raw_data = data
                        break
                else:
                    print(f"Error: Frame ID {frame_id} not found in raw log")
                    return
        except FileNotFoundError:
            print(f"Error: Could not read raw log file {raw_log}")
            return
        output_path = os.path.join(single_frame_dir, f"{raw_log_name}_frame_{frame_id}.png")
        plot_all_views(raw_data, processed_data, output_path, figsize=(20, 15))
        print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
