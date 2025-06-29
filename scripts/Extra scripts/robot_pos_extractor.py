import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_latest_log_file(log_dir):
    """Find the latest log file in the specified directory."""
    log_files = list(Path(log_dir).glob("yolo1d_scan_*.jsonl"))
    if not log_files:
        raise FileNotFoundError("No log files found in the specified directory")
    return max(log_files, key=lambda x: x.stat().st_mtime)

def extract_robot_positions(log_file):
    """Extract robot positions from the log file."""
    positions = []
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            frame_id = data.get("frame_id", 0)
            pose = data["pose"]  # [x, y, theta]
            positions.append({
                "frame_id": frame_id,
                "x": pose[0],
                "y": pose[1],
                "theta": pose[2]
            })
    return positions

def plot_robot_trajectory(positions, log_filename):
    """Plot the robot's trajectory."""
    x_coords = [pos["x"] for pos in positions]
    y_coords = [pos["y"] for pos in positions]
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'b-', label='Robot trajectory')
    plt.scatter(x_coords, y_coords, c='r', s=20, label='Robot positions')
    
    # Mark start point
    plt.scatter(x_coords[0], y_coords[0], c='g', s=200, marker='^', label='Start')
    plt.text(x_coords[0], y_coords[0], 'Start', fontsize=12, ha='right', va='bottom')
    
    # Mark end point
    plt.scatter(x_coords[-1], y_coords[-1], c='r', s=200, marker='v', label='End')
    plt.text(x_coords[-1], y_coords[-1], 'End', fontsize=12, ha='left', va='top')
    
    # Plot arrows to show orientation
    for pos in positions[::10]:  # Plot every 10th position to avoid overcrowding
        dx = 0.2 * np.cos(pos["theta"])
        dy = 0.2 * np.sin(pos["theta"])
        plt.arrow(pos["x"], pos["y"], dx, dy, 
                 head_width=0.05, head_length=0.1, fc='g', ec='g')
    
    plt.title('Robot Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Save the plot with the log filename
    output_dir = Path("output/robot_trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"robot_trajectory_{Path(log_filename).stem}.png"
    plt.savefig(output_dir / output_filename)
    plt.close()
    return output_filename

def save_robot_positions(positions, log_filename):
    """Save robot positions data to a JSON file."""
    output_dir = Path("output/robot_trajectory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"robot_positions_{Path(log_filename).stem}.json"
    
    # Convert positions to a format suitable for JSON serialization
    positions_data = {
        "metadata": {
            "total_frames": len(positions),
            "x_range": [min(pos["x"] for pos in positions), max(pos["x"] for pos in positions)],
            "y_range": [min(pos["y"] for pos in positions), max(pos["y"] for pos in positions)],
            "total_distance": sum(np.sqrt(np.diff([pos["x"] for pos in positions])**2 + 
                                        np.diff([pos["y"] for pos in positions])**2))
        },
        "positions": positions
    }
    
    with open(output_dir / output_filename, 'w') as f:
        json.dump(positions_data, f, indent=2)
    
    return output_filename

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = find_latest_log_file(log_dir)
    print(f"Processing log file: {latest_log}")
    
    # Extract positions
    positions = extract_robot_positions(latest_log)
    print(f"Extracted {len(positions)} robot positions")
    
    # Plot trajectory
    plot_filename = plot_robot_trajectory(positions, latest_log)
    print(f"Trajectory plot saved to output/robot_trajectory/{plot_filename}")
    
    # Save positions data
    data_filename = save_robot_positions(positions, latest_log)
    print(f"Robot positions data saved to output/robot_trajectory/{data_filename}")
    
    # Print some statistics
    x_coords = [pos["x"] for pos in positions]
    y_coords = [pos["y"] for pos in positions]
    print(f"\nPosition Statistics:")
    print(f"X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
    print(f"Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
    print(f"Total distance traveled: {sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)):.2f} units")

if __name__ == "__main__":
    main() 