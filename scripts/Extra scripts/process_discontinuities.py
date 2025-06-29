import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan

# Get the project root (parent of the scripts folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output/post_process_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Spike detection parameter
EPSILON = 0.25  # Threshold for detecting significant distance changes

def find_latest_log_file(log_dir):
    """Find the latest log file in the specified directory."""
    log_files = list(Path(log_dir).glob("yolo1d_scan_*.jsonl"))
    if not log_files:
        raise FileNotFoundError("No log files found in the specified directory")
    return max(log_files, key=lambda x: x.stat().st_mtime)

def process_frame(scan):
    """Process a single frame and return binary discontinuity list"""
    distances = scan.raw_scan
    
    # Calculate jumps between consecutive points
    jumps = np.abs(np.diff(distances))
    
    # Create binary list where 1 indicates discontinuity
    discontinuities = (jumps > EPSILON).astype(int).tolist()
    
    return discontinuities

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = find_latest_log_file(log_dir)
    print(f"Processing log file: {latest_log}")
    
    raw_log_name = Path(latest_log).stem.replace('.jsonl', '')
    
    with open(latest_log, 'r') as f:
        lines = f.readlines()
    
    # Process each frame and write directly to output file
    output_file = OUTPUT_DIR / f"discontinuities_{raw_log_name}.jsonl"
    total_frames = len(lines)
    
    with open(output_file, 'w') as f:
        for idx, line in enumerate(lines):
            if idx % max(1, total_frames // 50) == 0 or idx == total_frames - 1:
                percent = (idx + 1) / total_frames * 100
                print(f"Processing frame {idx + 1}/{total_frames} ({percent:.1f}%)", end='\r')
            
            data = json.loads(line)
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            discontinuities = process_frame(scan)
            
            # Write each frame as a single line
            frame_data = {
                'frame_id': scan.frame_id,
                'discontinuities': discontinuities
            }
            f.write(json.dumps(frame_data) + '\n')
    
    print()  # Newline after progress bar
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 