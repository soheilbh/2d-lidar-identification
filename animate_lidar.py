import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
import matplotlib as mpl
from datetime import datetime
import argparse

# Configuration
MAX_FRAMES = None  # Maximum number of frames to animate
SAVE_GIF = 0      # Save GIF animation (0: no, 1: yes)
SAVE_MP4 = 1      # Save MP4 animation (0: no, 1: yes)

# Set up matplotlib for better quality
plt.style.use('default')  # Use default style instead of seaborn
mpl.rcParams['figure.dpi'] = 150  # Increased DPI for better quality
mpl.rcParams['font.size'] = 10
mpl.rcParams['lines.antialiased'] = True
mpl.rcParams['text.antialiased'] = True
mpl.rcParams['font.family'] = 'sans-serif'  # Use system sans-serif font
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.facecolor'] = '#f0f0f0'

def read_log_file(log_file, max_frames=None):
    """Read the JSONL log file and return a list of frames."""
    frames = []
    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(json.loads(line))
    return frames

def create_animation(frames, output_dir='output', save_gif=True, save_mp4=True, output_name=None):
    """Create animation from LiDAR data frames."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('LiDAR Clusters Visualization', fontsize=14, pad=20)
    
    # Set initial plot limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # Get colormap for consistent cluster colors
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(20)]
    
    # Initialize empty lists for artists
    line_segments = []
    centroids = []
    bboxes = []
    labels = []
    info_text = None
    
    def update(frame):
        """Update function for animation."""
        nonlocal info_text
        
        # Clear previous frame
        for artist in line_segments + centroids + bboxes + labels:
            artist.remove()
        line_segments.clear()
        centroids.clear()
        bboxes.clear()
        labels.clear()
        
        if info_text:
            info_text.remove()
        
        # Get current frame data
        clusters = frame['clusters']
        pose = frame['pose']
        timestamp = frame['timestamp']
        
        # Plot robot position (always at origin since we're in robot frame)
        robot_marker = ax.plot(0, 0, 'o', color='red', markersize=10, label='Robot')[0]
        line_segments.append(robot_marker)
        
        # Plot each cluster
        for cluster in clusters:
            try:
                cluster_id = cluster['cluster_id']
                color = colors[cluster_id % 20]
                
                # Plot individual LiDAR points
                if 'xy_points' in cluster and cluster['xy_points']:
                    points = np.array(cluster['xy_points'])
                    if len(points) > 0:
                        # Plot points with small size and transparency
                        points_plot = ax.scatter(
                            points[:, 0], points[:, 1],
                            color=color,
                            s=20,  # Size of points
                            alpha=0.6,  # Transparency
                            edgecolors='none',  # No edge color
                            zorder=2  # Draw points above bounding boxes
                        )
                        line_segments.append(points_plot)
                
                # Check if we have valid centroid
                if all(isinstance(x, (int, float)) and np.isfinite(x) 
                      for x in cluster['centroid']):
                    # Plot centroid with larger marker
                    centroid = ax.plot(
                        cluster['centroid'][0], cluster['centroid'][1],
                        'x', color='black', markersize=10, markeredgewidth=2,
                        zorder=3  # Draw centroids above points
                    )[0]
                    centroids.append(centroid)
                    
                    # Add cluster ID label with better positioning
                    label = ax.text(
                        cluster['centroid'][0] + 0.15, cluster['centroid'][1] + 0.15,
                        f'ID: {cluster_id}', 
                        fontsize=8, 
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
                        zorder=4  # Draw labels above everything
                    )
                    labels.append(label)
                
                # Check if we have valid bounding box
                if all(isinstance(x, (int, float)) and np.isfinite(x) 
                      for x in cluster['bounding_box']):
                    # Draw bounding box with better visibility
                    min_x, max_x, min_y, max_y = cluster['bounding_box']
                    bbox = Rectangle(
                        (min_x, min_y), max_x - min_x, max_y - min_y,
                        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15,
                        zorder=1  # Draw bounding boxes below points
                    )
                    ax.add_patch(bbox)
                    bboxes.append(bbox)
            except (KeyError, TypeError, ValueError) as e:
                print(f"Error processing cluster {cluster.get('cluster_id', 'unknown')}: {e}")
                continue
        
        # Add info text with better formatting
        try:
            info_text = ax.text(
                0.98, 0.98,
                f'Time: {timestamp:.2f}s\n'
                f'Pose: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})\n'
                f'Clusters: {len(clusters)}',
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                fontsize=10,
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='gray',
                    boxstyle='round,pad=0.5'
                ),
                zorder=5  # Draw info text above everything
            )
        except (TypeError, ValueError) as e:
            print(f"Error displaying info text: {e}")
        
        return line_segments + centroids + bboxes + labels + [info_text]
    
    # Create animation with smooth transitions
    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=50,  # 20 fps
        blit=True,
        repeat=True
    )
    
    # Use provided output name or default
    if output_name is None:
        output_name = 'lidar_animation'
    
    # Save as high-quality GIF if requested
    gif_path = None
    if save_gif:
        print("\nSaving GIF animation...")
        gif_path = output_dir / f'{output_name}.gif'
        anim.save(
            gif_path,
            writer='pillow',
            fps=20,
            dpi=150,
            savefig_kwargs={'facecolor': 'white'}
        )
    
    # Save as MP4 if requested
    mp4_path = None
    if save_mp4:
        print("Saving MP4 animation...")
        mp4_path = output_dir / f'{output_name}.mp4'
        anim.save(
            mp4_path,
            writer='ffmpeg',
            fps=20,
            dpi=150,
            bitrate=5000,
            savefig_kwargs={'facecolor': 'white'}
        )
    
    # Show preview
    plt.tight_layout()
    plt.show()
    
    return gif_path, mp4_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create animation from LiDAR log file')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    parser.add_argument('--max-frames', type=int, default=MAX_FRAMES, help='Maximum number of frames to animate')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--gif', type=int, default=SAVE_GIF, choices=[0, 1], help='Save GIF animation (0: no, 1: yes)')
    parser.add_argument('--mp4', type=int, default=SAVE_MP4, choices=[0, 1], help='Save MP4 animation (0: no, 1: yes)')
    args = parser.parse_args()
    
    # Find the most recent log file if not specified
    if args.log_file is None:
        log_dir = Path('bot3/controllers/turtlebot3_controller/logs')
        log_files = sorted(log_dir.glob('scan_log_*.jsonl'), reverse=True)
        if not log_files:
            print("No log files found!")
            return
        log_file = log_files[0]
    else:
        log_file = Path(args.log_file)
    
    print(f"Processing log file: {log_file}")
    
    # Read and process the log file
    frames = read_log_file(log_file, max_frames=args.max_frames)
    print(f"Processing {len(frames)} frames out of {args.max_frames} requested")
    
    # Create animation
    output_name = log_file.stem  # Use log filename without extension
    gif_path, mp4_path = create_animation(
        frames, 
        output_dir=args.output_dir,
        save_gif=bool(args.gif),
        save_mp4=bool(args.mp4),
        output_name=output_name
    )
    
    print("\nAnimation saved to:")
    if gif_path:
        print(f"GIF: {gif_path}")
    if mp4_path:
        print(f"MP4: {mp4_path}")

if __name__ == "__main__":
    main() 