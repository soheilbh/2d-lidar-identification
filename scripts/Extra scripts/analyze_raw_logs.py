import os
import json
import glob
from datetime import datetime
from collections import defaultdict

def get_latest_log_file(logs_dir):
    """Get the most recent log file from the raw_logs directory."""
    log_files = glob.glob(os.path.join(logs_dir, 'yolo1d_scan_*.jsonl'))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {logs_dir}")
    return max(log_files, key=os.path.getmtime)

def analyze_log_file(log_file):
    """Analyze a log file and count frames per scenario."""
    print(f"\nAnalyzing log file: {os.path.basename(log_file)}")
    print("-" * 50)
    
    # Statistics
    total_frames = 0
    frames_per_scenario = defaultdict(int)
    scenarios_seen = set()
    
    # Read the log file
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_id = data.get('frame_id', 0)
                scenario_number = data.get('scenario_number', 0)
                
                # Update statistics
                total_frames += 1
                frames_per_scenario[scenario_number] += 1
                scenarios_seen.add(scenario_number)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
                continue
    
    # Print results
    print(f"Total frames: {total_frames}")
    print(f"Number of scenarios: {len(scenarios_seen)}")
    print("\nFrames per scenario:")
    print("-" * 30)
    
    # Sort scenarios by number
    for scenario in sorted(scenarios_seen):
        frames = frames_per_scenario[scenario]
        print(f"Scenario {scenario}: {frames} frames")
    
    # Print summary
    print("\nSummary:")
    print(f"Average frames per scenario: {total_frames/len(scenarios_seen):.1f}")
    print(f"Min frames in a scenario: {min(frames_per_scenario.values())}")
    print(f"Max frames in a scenario: {max(frames_per_scenario.values())}")

def main():
    # Get the raw_logs directory path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, 'raw_logs')
    
    try:
        # Get the latest log file
        latest_log = get_latest_log_file(logs_dir)
        
        # Analyze the log file
        analyze_log_file(latest_log)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 