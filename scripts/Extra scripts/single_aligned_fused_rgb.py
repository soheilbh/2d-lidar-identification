import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from pathlib import Path
from bot3.utils.lidar_utils import LidarScan
from scipy import ndimage
import matplotlib.pyplot as plt
from collections import deque

class LidarBuffer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)
        self.poses = deque(maxlen=buffer_size)
        self.threshold = int(np.ceil(buffer_size / 2))
    
    def add_frame(self, raw_scan, angles, pose):
        binary_array = np.zeros((64, 360), dtype=np.uint8)
        angle_indices = (np.round(np.degrees(angles)) % 360).astype(np.uint16)
        distance_indices = np.clip((np.array(raw_scan) * 15.75), 0, 63).astype(np.uint8)
        binary_array[distance_indices, angle_indices] = 1
        self.binary_arrays.append(binary_array)
        self.poses.append(pose)
    
    def get_rgb_image(self):
        if len(self.binary_arrays) < self.buffer_size:
            return None
        angles = [pose[2] for pose in self.poses]
        reference_angle = angles[-1]
        shifts = ((np.degrees(reference_angle - np.array(angles))) % 360).astype(np.uint16)
        density_array = np.sum([np.roll(arr, shift, axis=1) for arr, shift in zip(self.binary_arrays, shifts)], axis=0).astype(np.uint8)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        mask = ndimage.binary_dilation(density_array >= self.threshold, structure=structure).astype(np.uint8)
        binary = (density_array >= self.threshold).astype(np.uint8)
        sobel_x = ndimage.sobel(binary, axis=1).astype(np.uint8)
        sobel_y = ndimage.sobel(binary, axis=0).astype(np.uint8)
        edges = np.clip(np.abs(sobel_x) + np.abs(sobel_y), 0, 255).astype(np.uint8)
        edges = (edges > 0).astype(np.uint8) * 255
        rgb_image = np.zeros((64, 384, 3), dtype=np.uint8)
        rgb_image[:, :360, 0] = mask * 255
        rgb_image[:, :360, 1] = edges
        rgb_image[:, :360, 2] = mask * (density_array / self.buffer_size * 255).astype(np.uint8)
        return rgb_image

def main(buffer_size=5):
    buffer = LidarBuffer(buffer_size=buffer_size)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.abspath(os.path.join(script_dir, '../raw_logs'))
    latest_log = max(Path(log_dir).glob("yolo1d_scan_*.jsonl"), key=lambda x: x.stat().st_mtime)
    raw_log_name = os.path.splitext(os.path.basename(latest_log))[0]
    output_dir = Path("output/visualizations") / f"single_aligned_fused_rgb_{raw_log_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with open(latest_log, 'r') as f:
        for line in f:
            data = json.loads(line)
            if "status" in data or "raw_scan" not in data:
                continue
            scan = LidarScan(data["raw_scan"], data["pose"], frame_id=data.get("frame_id"))
            buffer.add_frame(scan.raw_scan, scan.angles, scan.pose)
            frame_count += 1
            if frame_count >= buffer_size:
                break
    rgb_image = buffer.get_rgb_image()
    if rgb_image is not None:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(rgb_image, aspect='equal', interpolation='nearest')
        # ax.set_title('2D LiDAR Aligned Fused RGB (5 frames)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Distance (bins)', fontsize=12)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 359, 383])
        ax.set_yticks([0, 16, 32, 48, 63])
        ax.set_xlim(0, 383)
        ax.set_ylim(63, 0)
        plt.tight_layout()
        output_path = output_dir / "single_aligned_fused_rgb.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Single aligned fused RGB image saved to: {output_path}")
    else:
        print("Not enough valid frames for aligned fusion.")

if __name__ == "__main__":
    main(buffer_size=5) 