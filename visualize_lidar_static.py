import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import os
import math
import matplotlib as mpl

# Configuration
SCAN_NUMBER = 0  # Select which scan to plot (0 for first scan, 1 for second, etc.)

# Global configuration
movement_mode = 0

def create_output_dir():
    """Create output directory if it doesn't exist."""
    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_scan_data(log_file, scan_index):
    """Load scan data from log file."""
    print(f"\nLoading scan data from: {log_file}")
    print(f"Requested scan index: {scan_index}")
    
    with open(log_file, "r") as f:
        for i, line in enumerate(f):
            if i == scan_index:
                data = json.loads(line)
                print("\nScan data loaded successfully:")
                print(f"Number of clusters: {len(data['clusters'])}")
                print(f"Raw scan length: {len(data['raw_scan'])}")
                print(f"Angles length: {len(data['angles'])}")
                
                # Print first cluster details
                if data['clusters']:
                    first_cluster = data['clusters'][0]
                    print("\nFirst cluster details:")
                    print(f"  ID: {first_cluster['cluster_id']}")
                    print(f"  Start index: {first_cluster['start_index']}")
                    print(f"  End index: {first_cluster['end_index']}")
                    print(f"  Size: {first_cluster['end_index'] - first_cluster['start_index'] + 1}")
                    if 'xy_points' in first_cluster:
                        print(f"  Number of xy_points: {len(first_cluster['xy_points'])}")
                return data
    return None

def plot_angle_vs_distance(ax, angles, distances):
    """Plot angle vs distance with improved readability."""
    # Convert angles to degrees for plotting
    angles_deg = np.degrees(angles)
    
    # Create stem plot with better styling
    markerline, stemlines, baseline = ax.stem(
        angles_deg, 
        distances,
        linefmt='gray',
        markerfmt='o',
        basefmt=' '
    )
    
    # Style the markers and stems
    plt.setp(markerline, markersize=3, color='darkblue', alpha=0.7)
    plt.setp(stemlines, linewidth=1, color='gray', alpha=0.5)
    
    # Add a light fill
    ax.fill_between(angles_deg, 0, distances, alpha=0.1, color='blue')
    
    # Customize grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.minorticks_on()
    
    # Set labels and title
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Angle vs Distance\n(LiDAR beam measurements)", pad=10)
    
    # Set axis limits with some padding
    ax.set_xlim(-5, 365)
    ymax = max(distances) * 1.1
    ax.set_ylim(0, ymax)
    
    # Add x-axis ticks at meaningful angles
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xticks(np.arange(0, 361, 15), minor=True)

def plot_distance_jumps(ax, angles, distances, epsilon):
    """Plot angle vs distance jumps."""
    jumps = np.abs(np.diff(distances))
    ax.plot(np.degrees(angles[:-1]), jumps, '.-', color='gray')
    ax.axhline(y=epsilon, color='red', linestyle='--', label=f'EPSILON={epsilon}')
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Distance Jump (m)")
    ax.set_title("Distance Difference Between Samples")
    ax.legend()
    ax.grid(True)

def plot_polar_contour(ax, angles, distances):
    """Plot polar contour view."""
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    ax.plot(x, y, '.', color='lightgray', alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Polar Contour from Distance Measurements")
    ax.grid(True)
    ax.axis('equal')

def plot_top_down_view(ax, angles, distances, clusters):
    """Plot top-down view with clusters and features."""
    # Plot raw points in gray
    x_raw = distances * np.cos(angles)
    y_raw = distances * np.sin(angles)
    ax.scatter(x_raw, y_raw, color='gray', s=5, alpha=0.3, label="Raw LiDAR")
    
    # Plot clusters with their features
    cmap = plt.cm.tab20
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        color = cmap(cluster_id % 20)
        
        # Plot cluster points
        xy = np.array(cluster.get("xy_points", []))
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=20, alpha=0.7, color=color, label=f"Cluster {cluster_id}")
        
        # Plot centroid
        centroid = cluster.get("centroid", [float('nan'), float('nan')])
        if not any(math.isnan(c) for c in centroid):
            ax.plot(centroid[0], centroid[1], 'kx', markersize=8)
        
        # Plot start/end markers
        line_start = cluster.get("line_start", [float('nan'), float('nan')])
        line_end = cluster.get("line_end", [float('nan'), float('nan')])
        if not any(math.isnan(p) for p in line_start):
            ax.plot(line_start[0], line_start[1], '^', color=color, markersize=8)
        if not any(math.isnan(p) for p in line_end):
            ax.plot(line_end[0], line_end[1], 'v', color=color, markersize=8)
        
        # Plot corners
        corners = cluster.get("corners", [])
        corner_angles = cluster.get("corner_angles", [])
        for p, a in zip(corners, corner_angles):
            # Determine color based on angle
            color = 'red' if a < 60.0 else 'black'
            weight = 'bold' if a < 60.0 else 'normal'
            
            # Plot corner point
            ax.plot(p[0], p[1], '*', color=color, markersize=10)
            
            # Add angle label
            ax.text(p[0] + 0.05, p[1] + 0.05, f"{a:.1f}°", color=color, weight=weight)
        
        # Add warning for short clusters
        if cluster.get("size", 0) < 5:
            ax.text(centroid[0], centroid[1] - 0.2, "SHORT", color='red', weight='bold')
    
    # Plot robot position
    ax.plot(0, 0, 'ro', markersize=10, label="Robot")
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-down View with Clusters")
    ax.grid(True)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=8)

def plot_polar_beam_view(ax, angles, distances, clusters):
    """Plot LiDAR data in polar coordinates matching paper style."""
    # Convert angles to degrees
    angles_deg = np.degrees(angles)
    
    # Get colormap
    cmap = plt.cm.tab20
    
    # Debug: Print total number of clusters
    print(f"\nTotal clusters: {len(clusters)}")
    
    # Plot each cluster separately with its own color
    for i, cluster in enumerate(clusters):
        start_idx = cluster['start_index']
        end_idx = cluster['end_index']
        cluster_id = cluster['cluster_id']
        color = cmap(cluster_id % 20)  # Same color as in plot (d)
        
        # Debug print cluster details
        print(f"\nCluster {i}:")
        print(f"  ID: {cluster_id}")
        print(f"  Indices: {start_idx} to {end_idx}")
        print(f"  Size: {end_idx-start_idx+1}")
        
        # Validate indices
        if start_idx > end_idx:
            print(f"  WARNING: Invalid index range (start > end) for cluster {i}")
            continue
            
        if start_idx < 0 or end_idx >= len(angles):
            print(f"  WARNING: Index out of bounds for cluster {i}")
            continue
        
        # Get angles and distances for this cluster
        cluster_angles = angles_deg[start_idx:end_idx+1]
        cluster_distances = distances[start_idx:end_idx+1]
        
        # Debug print data points
        print(f"  Number of points: {len(cluster_angles)}")
        if len(cluster_angles) > 0:
            print(f"  First point: angle={cluster_angles[0]:.2f}, dist={cluster_distances[0]:.2f}")
            print(f"  Last point: angle={cluster_angles[-1]:.2f}, dist={cluster_distances[-1]:.2f}")
        
        # Sort points within this cluster
        sort_idx = np.argsort(cluster_angles)
        angles_sorted = cluster_angles[sort_idx]
        distances_sorted = cluster_distances[sort_idx]
        
        # Plot just this cluster's line with matching color
        if len(angles_sorted) > 0:  # Only plot if we have points
            ax.plot(angles_sorted, distances_sorted, '-', color=color, linewidth=1, 
                   label=f'Cluster {cluster_id}')
        else:
            print(f"  WARNING: No points to plot for cluster {i}")
    
    # Set x-axis from 0 to 360 degrees
    ax.set_xlim(0, 360)
    
    # Set title and labels
    ax.set_title("Angle vs. Distance Measurements")
    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("D (m)")
    
    # Configure grid
    ax.grid(True, linestyle='-', alpha=0.2)
    
    # Set x-ticks every 45 degrees to match first plot
    xticks = np.arange(0, 361, 45)
    ax.set_xticks(xticks)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend to match plot (d)
    ax.legend(loc='upper right', fontsize=8)

def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Load scan data using SCAN_NUMBER from configuration
    log_dir = Path("bot3/controllers/turtlebot3_controller/logs")
    log_file = sorted(log_dir.glob("scan_log_*.jsonl"))[-1]
    scan_data = load_scan_data(log_file, SCAN_NUMBER)  # Use SCAN_NUMBER instead of args.scan
    
    if scan_data is None:
        print(f"Error: Scan index {SCAN_NUMBER} not found in log file")
        return
    
    # Extract data
    raw_scan = np.array(scan_data["raw_scan"])
    angles = np.array(scan_data["angles"])
    clusters = scan_data["clusters"]
    
    # Create figure with 3x2 subplots (5 plots total)
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    
    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Plot all views
    plot_angle_vs_distance(ax1, angles, raw_scan)
    plot_distance_jumps(ax2, angles, raw_scan, 0.25)
    plot_polar_contour(ax3, angles, raw_scan)
    plot_top_down_view(ax4, angles, raw_scan, clusters)
    plot_polar_beam_view(ax5, angles, raw_scan, clusters)
    
    # Add timestamp to the empty space in bottom right
    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.65, 0.25, f"Scan Index: {SCAN_NUMBER}\nTimestamp: {scan_time}\nClusters: {len(clusters)}", 
             fontsize=10, ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"static_frame_plot_scan{SCAN_NUMBER:03d}_{timestamp}.png"
    output_path = output_dir / filename
    
    # Save with extra padding to prevent cutoff
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main() 