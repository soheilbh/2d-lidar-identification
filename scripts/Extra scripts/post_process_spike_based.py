import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms
from pathlib import Path
import glob
from bot3.utils.lidar_utils import Object2D, KNOWN_OBJECTS, LidarScan

# Get the project root (parent of the scripts folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOT_OUTPUT_DIR = PROJECT_ROOT / "output/plots/"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spike detection parameter
EPSILON = 0.25  # Threshold for detecting significant distance changes

def get_object_angle_range(obj, robot_x, robot_y, robot_theta):
    """Calculate the angle of an object's center relative to robot's forward direction (now +y axis)"""
    # Global center
    global_center = (obj.x, obj.y)
    
    # Transform global position to robot-relative coordinates (no 90-degree rotation)
    rel_x = obj.x - robot_x
    rel_y = obj.y - robot_y
    
    # Calculate distance from robot to object center
    distance = np.sqrt(rel_x**2 + rel_y**2)
    
    # Calculate angle relative to robot's forward direction (now +y axis)
    angle = np.degrees(np.arctan2(rel_x, rel_y))  # rel_y is forward, rel_x is left
    if angle < 0:
        angle += 360  # Convert negative angles to positive
    
    print(f"\n{obj.name}:")
    print(f"Global center: ({global_center[0]:.2f}, {global_center[1]:.2f})")
    print(f"Relative center: ({rel_x:.2f}, {rel_y:.2f})")
    print(f"Distance from robot: {distance:.2f}m")
    print(f"Center angle: {angle:.2f}°")
    
    # Return a range of ±45 degrees around the center angle
    return angle - 45, angle + 45

def match_spikes_to_objects(spike_angles, spike_values, objects, robot_pose):
    """Match spike ranges to objects based on robot pose"""
    robot_x, robot_y, robot_theta = robot_pose
    matches = []
    
    for obj in objects:
        if not obj.should_scan:
            continue
            
        obj_start, obj_end = get_object_angle_range(obj, robot_x, robot_y, robot_theta)
        
        # Find spikes that fall within this object's angle range
        if obj_end < obj_start:  # Handle wrapping around 360
            mask = (spike_angles >= obj_start) | (spike_angles <= obj_end)
        else:
            mask = (spike_angles >= obj_start) & (spike_angles <= obj_end)
            
        if np.any(mask):
            matches.append({
                'object': obj.name,
                'start_angle': obj_start,
                'end_angle': obj_end,
                'spike_angles': spike_angles[mask],
                'spike_values': spike_values[mask]
            })
    
    return matches

# --- Visualization ---
def plot_all_views(angles, distances, pose, output_path):
    # Door frame detection BEFORE any filtering
    angles_full = angles.copy()
    distances_full = distances.copy()
    angles_deg_full = np.degrees(angles_full)
    zero_mask = (np.isinf(distances_full)) | (np.isnan(distances_full)) | (distances_full < 0.05)
    if np.any(zero_mask):
        zero_angles = angles_deg_full[zero_mask]
        door_start = np.min(zero_angles)
        door_end = np.max(zero_angles)
    else:
        door_start = door_end = None

    # Now apply valid_mask for the rest of the processing
    valid_mask = np.isfinite(distances)
    angles = angles[valid_mask]
    distances = distances[valid_mask]

    # Calculate spikes
    jumps = np.abs(np.diff(distances))
    spike_mask = jumps > EPSILON
    spike_angles = np.degrees(angles[:-1])[spike_mask]
    spike_values = jumps[spike_mask]
    
    # Match spikes to objects
    spike_matches = match_spikes_to_objects(spike_angles, spike_values, KNOWN_OBJECTS, pose)
    
    # Print spike matches
    print("\n--- Spike Ranges for Each Object ---")
    for match in spike_matches:
        print(f"\n{match['object']}:")
        print(f"Expected angle range: {match['start_angle']:.2f}° to {match['end_angle']:.2f}°")
        print("Spikes in this range:")
        for angle, value in zip(match['spike_angles'], match['spike_values']):
            print(f"  Angle: {angle:.2f}°, Jump: {value:.3f}m")

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    
    # 1. Angle vs Distance
    ax1 = fig.add_subplot(gs[0, 0])
    angles_deg = np.degrees(angles)  # Use raw angles directly
    markerline, stemlines, baseline = ax1.stem(
        angles_deg, distances, linefmt='gray', markerfmt='o', basefmt=' ')
    plt.setp(markerline, markersize=3, color='darkblue', alpha=0.7)
    plt.setp(stemlines, linewidth=1, color='gray', alpha=0.5)
    ax1.fill_between(angles_deg, 0, distances, alpha=0.1, color='blue')
    ax1.grid(True, which='major', linestyle='-', alpha=0.3)
    ax1.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax1.minorticks_on()
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Angle vs Distance\n(LiDAR beam measurements)", pad=10)
    ax1.set_xlim(-5, 365)
    ymax = np.nanmax(distances) * 1.1
    ax1.set_ylim(0, ymax)
    ax1.set_xticks(np.arange(0, 361, 45))
    ax1.set_xticks(np.arange(0, 361, 15), minor=True)

    # 2. Distance Jumps
    ax2 = fig.add_subplot(gs[0, 1])
    jumps = np.abs(np.diff(distances))
    ax2.plot(angles_deg[:-1], jumps, '.-', color='gray')
    ax2.axhline(y=EPSILON, color='red', linestyle='--', label=f'EPSILON={EPSILON}')
    ax2.set_xlabel("Angle (degrees)")
    ax2.set_ylabel("Distance Jump (m)")
    ax2.set_title("Distance Difference Between Samples")
    ax2.legend()
    ax2.grid(True)

    # 3. Polar Contour
    ax3 = fig.add_subplot(gs[1, 0])
    # Use raw LiDAR data directly with correct coordinate transformation
    x = distances * np.sin(angles)
    y = distances * np.cos(angles)
    ax3.plot(x, y, '.', color='lightgray', alpha=0.5)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Polar Contour from Distance Measurements")
    ax3.grid(True)
    ax3.axis('equal')

    # 4. Top-down View
    ax4 = fig.add_subplot(gs[1, 1])
    # Use raw LiDAR data directly with correct coordinate transformation
    x_raw = distances * np.sin(angles)
    y_raw = distances * np.cos(angles)
    ax4.scatter(x_raw, y_raw, color='gray', s=5, alpha=0.3, label="Raw LiDAR")
    
    # Plot known objects as rotated rectangles
    for obj in KNOWN_OBJECTS:
        if not obj.should_scan:
            continue
        w, d = obj.get_dimensions()
        rect = Rectangle((-w/2, -d/2), w, d, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        t = transforms.Affine2D().translate(obj.x, obj.y) + ax4.transData
        rect.set_transform(t)
        ax4.add_patch(rect)
        ax4.text(obj.x, obj.y, obj.name, fontsize=10, color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax4.plot(0, 0, 'ro', markersize=10, label="Robot")  # Robot is at origin
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_title("Top-down View with Objects")
    ax4.grid(True)
    ax4.axis('equal')
    ax4.legend(loc='upper right', fontsize=8)

    # Replace the last subplot with a filtered spike plot
    ax5 = fig.add_subplot(gs[2, :])
    # Plot only spikes above EPSILON
    jumps = np.abs(np.diff(distances))
    spike_mask = jumps > EPSILON
    spike_angles = angles_deg[:-1][spike_mask]
    spike_values = jumps[spike_mask]
    
    # Sort spikes by angle
    sorted_indices = np.argsort(spike_angles)
    spike_angles = spike_angles[sorted_indices]
    spike_values = spike_values[sorted_indices]
    
    # Match spikes to objects
    spike_matches = match_spikes_to_objects(spike_angles, spike_values, KNOWN_OBJECTS, pose)
    
    # Plot spikes for each object with different colors
    colors = ['red', 'blue', 'green']
    for match, color in zip(spike_matches, colors):
        obj_name = match['object']
        obj_spikes = match['spike_angles']
        obj_values = match['spike_values']
        
        # Plot all spikes for this object
        ax5.vlines(obj_spikes, 0, obj_values, color=color, linewidth=2, label=f'{obj_name} spikes')
        
        # Mark start and end spikes
        if len(obj_spikes) > 0:
            start_angle = obj_spikes[0]
            end_angle = obj_spikes[-1]
            start_value = obj_values[0]
            end_value = obj_values[-1]
            
            # Mark start spike
            ax5.scatter(start_angle, start_value, color=color, s=100, marker='^', label=f'{obj_name} start')
            ax5.text(start_angle, start_value + 0.1, f'Start: {start_angle:.0f}°', 
                    color=color, fontsize=8, ha='center', va='bottom')
            
            # Mark end spike
            ax5.scatter(end_angle, end_value, color=color, s=100, marker='v', label=f'{obj_name} end')
            ax5.text(end_angle, end_value + 0.1, f'End: {end_angle:.0f}°', 
                    color=color, fontsize=8, ha='center', va='bottom')
            
            # Add orientation arrow
            center_angle = (start_angle + end_angle) / 2
            center_value = max(obj_values) / 2
            # Draw arrow from center to end
            ax5.annotate('', xy=(end_angle, center_value), xytext=(center_angle, center_value),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
            # Add orientation text
            ax5.text(center_angle, center_value + 0.2, f'Orientation: {end_angle-start_angle:.0f}°', 
                    color=color, fontsize=8, ha='center', va='bottom')

    # Plot all distance==0 as special markers (door frame or no data)
    ax5.scatter(angles_deg_full[zero_mask], np.zeros(np.sum(zero_mask)), color='black', s=30, label='distance = 0 (door frame)')

    ax5.set_xlabel("Angle (degrees)")
    ax5.set_ylabel("Distance Jump (m)")
    ax5.set_title("Spikes for Each Object (with Start/End and Orientation)")
    ax5.grid(True)
    ax5.legend()
    ax5.set_xlim(-5, 365)
    ax5.set_xticks(np.arange(0, 361, 45))
    ax5.set_xticks(np.arange(0, 361, 15), minor=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Main ---
def find_latest_log_file(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "yolo1d_scan_*.jsonl"))
    if not log_files:
        raise FileNotFoundError("No log files found in the specified directory")
    return max(log_files, key=os.path.getctime)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = find_latest_log_file(log_dir)
    print(f"Processing log file: {latest_log}")
    with open(latest_log, 'r') as f_in:
        first_line = f_in.readline()
        frame_data = json.loads(first_line)
        scan = LidarScan(frame_data["raw_scan"], frame_data["pose"], frame_id=frame_data.get("frame_id"))
        raw_scan = scan.raw_scan
        angles = scan.angles
        pose = scan.pose
        valid_mask = np.isfinite(raw_scan)
        filtered_scan = raw_scan[valid_mask]
        filtered_angles = angles[valid_mask]
        output_path = PLOT_OUTPUT_DIR / "static_frame_plot_spike_based_scan000_postproc.png"
        plot_all_views(filtered_angles, filtered_scan, pose, output_path)
        print(f"Saved multi-panel plot to {output_path}")

if __name__ == "__main__":
    main() 