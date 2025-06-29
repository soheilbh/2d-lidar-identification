import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path

def load_robot_positions(data_file):
    """Load robot positions from the saved JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data["positions"]

def analyze_movement_space(positions):
    """Analyze the robot's movement space and coverage."""
    x_coords = [pos["x"] for pos in positions]
    y_coords = [pos["y"] for pos in positions]
    thetas = [pos["theta"] for pos in positions]
    
    # Calculate movement space statistics
    print("\nMovement Space Analysis:")
    print(f"X range: [{min(x_coords):.3f}, {max(x_coords):.3f}] (span: {max(x_coords)-min(x_coords):.3f})")
    print(f"Y range: [{min(y_coords):.3f}, {max(y_coords):.3f}] (span: {max(y_coords)-min(y_coords):.3f})")
    print(f"Theta range: [{min(thetas):.3f}, {max(thetas):.3f}] (span: {max(thetas)-min(thetas):.3f})")
    
    # Calculate coverage
    x_span = max(x_coords) - min(x_coords)
    y_span = max(y_coords) - min(y_coords)
    total_area = x_span * y_span
    print(f"\nTotal covered area: {total_area:.3f} square units")
    
    # Analyze movement density
    grid_size = 0.1  # 10cm grid
    x_bins = int(x_span / grid_size) + 1
    y_bins = int(y_span / grid_size) + 1
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
    
    # Calculate coverage statistics
    covered_cells = np.sum(hist > 0)
    total_cells = x_bins * y_bins
    coverage_percentage = (covered_cells / total_cells) * 100
    
    print(f"\nCoverage Analysis (using {grid_size} unit grid):")
    print(f"Covered cells: {covered_cells}")
    print(f"Total cells: {total_cells}")
    print(f"Coverage percentage: {coverage_percentage:.1f}%")
    
    # Analyze movement patterns
    print("\nMovement Pattern Analysis:")
    print(f"Average movement per frame: {np.mean([np.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2) for i in range(len(x_coords)-1)]):.3f}")
    print(f"Maximum movement between frames: {max([np.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2) for i in range(len(x_coords)-1)]):.3f}")
    
    # Analyze orientation changes
    theta_changes = [abs(thetas[i+1] - thetas[i]) for i in range(len(thetas)-1)]
    print(f"\nOrientation Analysis:")
    print(f"Average theta change: {np.mean(theta_changes):.3f}")
    print(f"Maximum theta change: {max(theta_changes):.3f}")
    print(f"Number of significant turns (>0.5 rad): {sum(1 for t in theta_changes if t > 0.5)}")
    
    return {
        'x_range': (min(x_coords), max(x_coords)),
        'y_range': (min(y_coords), max(y_coords)),
        'theta_range': (min(thetas), max(thetas)),
        'coverage_percentage': coverage_percentage,
        'total_area': total_area
    }

def find_sequence_boundaries(positions, movement_threshold=0.01, theta_threshold=0.1):
    """Find sequence boundaries based on movement and theta changes."""
    sequences = []
    current_sequence = [0]  # Start with first frame
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        # Calculate movement
        dx = curr_pos["x"] - prev_pos["x"]
        dy = curr_pos["y"] - prev_pos["y"]
        movement = np.sqrt(dx**2 + dy**2)
        
        # Calculate theta change
        theta_change = abs(curr_pos["theta"] - prev_pos["theta"])
        
        # If significant change in either movement or theta, start new sequence
        if movement > movement_threshold or theta_change > theta_threshold:
            if len(current_sequence) >= 5:  # Only keep sequences with at least 5 frames
                sequences.append(current_sequence)
            current_sequence = [i]
        else:
            current_sequence.append(i)
    
    # Add the last sequence if it has enough frames
    if len(current_sequence) >= 5:
        sequences.append(current_sequence)
    
    return sequences

def analyze_sequences(positions, sequences, window_size=5):
    """Analyze the sequences for data splitting."""
    print("\nSequence Analysis:")
    print(f"Total number of sequences: {len(sequences)}")
    
    # Calculate sequence lengths
    lengths = [len(seq) for seq in sequences]
    print(f"Average sequence length: {np.mean(lengths):.1f} frames")
    print(f"Min sequence length: {min(lengths)} frames")
    print(f"Max sequence length: {max(lengths)} frames")
    
    # Calculate total frames in sequences
    total_frames = sum(lengths)
    print(f"Total frames in sequences: {total_frames}")
    print(f"Percentage of frames in sequences: {(total_frames/len(positions)*100):.1f}%")
    
    # Calculate number of 5-frame windows
    windows = []
    for seq in sequences:
        for i in range(len(seq) - window_size + 1):
            windows.append(seq[i:i+window_size])
    
    print(f"\nNumber of 5-frame windows: {len(windows)}")
    
    # Propose data split
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    n_windows = len(windows)
    n_train = int(n_windows * train_ratio)
    n_val = int(n_windows * val_ratio)
    n_test = n_windows - n_train - n_val
    
    print("\nProposed Data Split:")
    print(f"Training windows: {n_train} ({train_ratio*100:.1f}%)")
    print(f"Validation windows: {n_val} ({val_ratio*100:.1f}%)")
    print(f"Testing windows: {n_test} ({test_ratio*100:.1f}%)")
    
    # Analyze sequence distribution
    print("\nSequence Distribution Analysis:")
    sequence_lengths = [len(seq) for seq in sequences]
    print(f"Sequences with length < 10 frames: {sum(1 for x in sequence_lengths if x < 10)}")
    print(f"Sequences with length 10-20 frames: {sum(1 for x in sequence_lengths if 10 <= x < 20)}")
    print(f"Sequences with length 20-50 frames: {sum(1 for x in sequence_lengths if 20 <= x < 50)}")
    print(f"Sequences with length > 50 frames: {sum(1 for x in sequence_lengths if x >= 50)}")
    
    # Analyze sequence positions
    print("\nSequence Position Analysis:")
    for i, seq in enumerate(sequences):
        start_frame = seq[0]
        end_frame = seq[-1]
        length = len(seq)
        print(f"Sequence {i+1}: Frames {start_frame}-{end_frame} (Length: {length})")
    
    # Verify no overlap between sequences
    all_frames = set()
    for seq in sequences:
        seq_frames = set(seq)
        if all_frames.intersection(seq_frames):
            print("\nWARNING: Overlap detected between sequences!")
        all_frames.update(seq_frames)
    
    print(f"\nTotal unique frames in sequences: {len(all_frames)}")
    print(f"Frames not in any sequence: {len(positions) - len(all_frames)}")

def analyze_frame_similarity(positions, window_size=5):
    """Analyze similarity between frames in the buffer window."""
    print("\nFrame Similarity Analysis:")
    
    # Calculate average movement between consecutive frames
    movements = []
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        dx = curr_pos["x"] - prev_pos["x"]
        dy = curr_pos["y"] - prev_pos["y"]
        movement = np.sqrt(dx**2 + dy**2)
        movements.append(movement)
    
    print(f"Average movement between consecutive frames: {np.mean(movements):.6f}")
    print(f"Max movement between consecutive frames: {max(movements):.6f}")
    print(f"Min movement between consecutive frames: {min(movements):.6f}")
    
    # Analyze similarity within buffer windows
    window_similarities = []
    window_changes = []  # Track changes within each window
    for i in range(len(positions) - window_size + 1):
        window = positions[i:i+window_size]
        # Calculate changes within window
        window_movements = []
        window_thetas = []
        for j in range(1, len(window)):
            prev_pos = window[j-1]
            curr_pos = window[j]
            # Position changes
            dx = curr_pos["x"] - prev_pos["x"]
            dy = curr_pos["y"] - prev_pos["y"]
            movement = np.sqrt(dx**2 + dy**2)
            window_movements.append(movement)
            # Theta changes
            theta_change = abs(curr_pos["theta"] - prev_pos["theta"])
            window_thetas.append(theta_change)
        
        window_similarities.append(np.mean(window_movements))
        window_changes.append({
            'movement': window_movements,
            'theta': window_thetas,
            'total_movement': sum(window_movements),
            'total_theta_change': sum(window_thetas)
        })
    
    print(f"\nBuffer Window Analysis:")
    print(f"Average movement within {window_size}-frame windows: {np.mean(window_similarities):.6f}")
    print(f"Max movement within windows: {max(window_similarities):.6f}")
    print(f"Min movement within windows: {min(window_similarities):.6f}")
    
    # Analyze changes within windows
    total_movements = [w['total_movement'] for w in window_changes]
    total_thetas = [w['total_theta_change'] for w in window_changes]
    
    print(f"\nChanges Within Windows:")
    print(f"Average total movement in window: {np.mean(total_movements):.6f}")
    print(f"Max total movement in window: {max(total_movements):.6f}")
    print(f"Min total movement in window: {min(total_movements):.6f}")
    print(f"Average total theta change in window: {np.mean(total_thetas):.6f}")
    print(f"Max total theta change in window: {max(total_thetas):.6f}")
    print(f"Min total theta change in window: {min(total_thetas):.6f}")
    
    # Count windows with significant changes
    movement_threshold = 0.01  # 1cm
    theta_threshold = 0.1  # ~5.7 degrees
    
    significant_movement_windows = sum(1 for m in total_movements if m > movement_threshold)
    significant_theta_windows = sum(1 for t in total_thetas if t > theta_threshold)
    
    print(f"\nWindows with Significant Changes:")
    print(f"Windows with movement > {movement_threshold}: {significant_movement_windows} ({(significant_movement_windows/len(window_changes)*100):.1f}%)")
    print(f"Windows with theta change > {theta_threshold}: {significant_theta_windows} ({(significant_theta_windows/len(window_changes)*100):.1f}%)")
    
    # Propose new splitting strategy
    print("\nProposed New Splitting Strategy:")
    print("1. Use similar frames for validation and test")
    print("2. Select frames with similar movement patterns")
    print("3. Ensure each split has representative samples")
    
    return {
        'window_similarities': window_similarities,
        'window_changes': window_changes,
        'significant_movement_windows': significant_movement_windows,
        'significant_theta_windows': significant_theta_windows
    }

def normalize_angle(angle):
    """Normalize angle to be between -pi and pi."""
    return np.arctan2(np.sin(angle), np.cos(angle))

def analyze_batch_changes(positions, batch_sizes=[3, 4, 5, 6, 10]):
    """Analyze changes in x, y, and angle for different batch sizes."""
    print("\nBatch Size Analysis:")
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        
        # Calculate changes for each batch
        x_changes = []
        y_changes = []
        angle_changes = []
        
        for i in range(0, len(positions) - batch_size + 1, batch_size):
            batch = positions[i:i+batch_size]
            if len(batch) == batch_size:  # Only process complete batches
                # Calculate total change in x, y, and angle for this batch
                x_change = abs(batch[-1]["x"] - batch[0]["x"])
                y_change = abs(batch[-1]["y"] - batch[0]["y"])
                
                # Calculate angle change properly considering angle wrapping
                start_angle = batch[0]["theta"]
                end_angle = batch[-1]["theta"]
                angle_diff = normalize_angle(end_angle - start_angle)
                angle_change = abs(angle_diff)
                
                x_changes.append(x_change)
                y_changes.append(y_change)
                angle_changes.append(angle_change)
        
        # Calculate statistics
        print(f"X changes:")
        print(f"  Max: {max(x_changes):.6f}")
        print(f"  Min: {min(x_changes):.6f}")
        print(f"  Mean: {np.mean(x_changes):.6f}")
        print(f"  Std: {np.std(x_changes):.6f}")
        
        print(f"\nY changes:")
        print(f"  Max: {max(y_changes):.6f}")
        print(f"  Min: {min(y_changes):.6f}")
        print(f"  Mean: {np.mean(y_changes):.6f}")
        print(f"  Std: {np.std(y_changes):.6f}")
        
        print(f"\nAngle changes (radians):")
        print(f"  Max: {max(angle_changes):.6f}")
        print(f"  Min: {min(angle_changes):.6f}")
        print(f"  Mean: {np.mean(angle_changes):.6f}")
        print(f"  Std: {np.std(angle_changes):.6f}")

        # Convert angle changes to degrees for better readability
        angle_changes_deg = [np.degrees(angle) for angle in angle_changes]
        print(f"\nAngle changes (degrees):")
        print(f"  Max: {max(angle_changes_deg):.2f}°")
        print(f"  Min: {min(angle_changes_deg):.2f}°")
        print(f"  Mean: {np.mean(angle_changes_deg):.2f}°")
        print(f"  Std: {np.std(angle_changes_deg):.2f}°")

def analyze_movement(positions):
    """Analyze robot movement for all frames."""
    print(f"\nAnalyzing all {len(positions)} frames:")
    
    # Analyze movement space
    movement_space = analyze_movement_space(positions)
    
    # Analyze frame similarity
    similarity_analysis = analyze_frame_similarity(positions)
    
    # Find sequences
    sequences = find_sequence_boundaries(positions)
    
    # Analyze sequences
    analyze_sequences(positions, sequences)
    
    # Analyze batch changes
    analyze_batch_changes(positions)
    
    # Original movement analysis
    movements = []
    thetas = []
    x_coords = []
    y_coords = []
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        dx = curr_pos["x"] - prev_pos["x"]
        dy = curr_pos["y"] - prev_pos["y"]
        movement = np.sqrt(dx**2 + dy**2)
        movements.append(movement)
        
        thetas.append(curr_pos["theta"])
        x_coords.append(curr_pos["x"])
        y_coords.append(curr_pos["y"])
    
    print("\nOverall Statistics:")
    print(f"Total frames: {len(positions)}")
    print(f"Total distance traveled: {sum(movements):.3f}")
    print(f"Average movement per frame: {np.mean(movements):.3f}")
    print(f"Max movement: {max(movements):.3f}")
    print(f"Min movement: {min(movements):.3f}")
    print(f"Std movement: {np.std(movements):.3f}")
    
    print("\nPosition Statistics:")
    print(f"X range: [{min(x_coords):.3f}, {max(x_coords):.3f}]")
    print(f"Y range: [{min(y_coords):.3f}, {max(y_coords):.3f}]")
    print(f"Average X: {np.mean(x_coords):.3f}")
    print(f"Average Y: {np.mean(y_coords):.3f}")
    
    print("\nTheta Statistics:")
    print(f"Average theta: {np.mean(thetas):.3f}")
    print(f"Max theta: {max(thetas):.3f}")
    print(f"Min theta: {min(thetas):.3f}")
    print(f"Std theta: {np.std(thetas):.3f}")

def main():
    # Find the latest robot positions file
    output_dir = Path("output/robot_trajectory")
    position_files = list(output_dir.glob("robot_positions_*.json"))
    if not position_files:
        print("No robot position files found!")
        return
    
    latest_file = max(position_files, key=lambda x: x.stat().st_mtime)
    print(f"Processing file: {latest_file}")
    
    # Load and analyze data
    positions = load_robot_positions(latest_file)
    analyze_movement(positions)

if __name__ == "__main__":
    main() 